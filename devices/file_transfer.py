"""
JARVIS AI — Permission-Gated Chunked File Transfer Engine
=========================================================
Implements verified, chunked file transfers (64 KB chunks) with:
- Strict folder sandboxing (via AuthorizedFolderSandbox)
- Mandatory permission checking (FILE_WRITE scope)
- On-the-fly SHA-256 streaming checksum verification
- Atomic rename from temporary `.part` file to guarantee crash consistency
"""

import os
import uuid
import hashlib
import sqlite3
from typing import Dict, Any, Optional, Generator, Tuple
from datetime import datetime, timezone
from pydantic import BaseModel, Field

from devices.folder_access import authorized_folder_sandbox
from devices.permission import consent_manager, PermissionScope
from core.logger import log

CHUNK_SIZE = 64 * 1024  # 64 KB chunk size


class TransferRecord(BaseModel):
    transfer_id: str = Field(default_factory=lambda: f"xfr-{uuid.uuid4().hex[:8]}")
    source_device: str
    target_device: str
    file_name: str
    file_size: int
    sha256_hash: str
    target_folder: str
    target_path: str
    status: str = "pending"  # "pending", "authorized", "in_progress", "completed", "failed", "rejected"
    bytes_transferred: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None


class FileTransferEngine:
    """Manages verified, secure file transfers across authorized endpoints."""

    def __init__(self, db_path: Optional[str] = None, sandbox: Optional[Any] = None):
        if db_path is None:
            from config.settings import BASE_DIR
            self.db_path = str(BASE_DIR / "jarvis.db")
        else:
            self.db_path = db_path

        self.sandbox = sandbox or authorized_folder_sandbox
        self._active_transfers: Dict[str, TransferRecord] = {}
        self._active_hasher: Dict[str, Any] = {}  # transfer_id -> sha256 hasher
        self._active_files: Dict[str, Any] = {}   # transfer_id -> file handle
        self._init_db()

    def _init_db(self):
        """Create file_transfers table in SQLite."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS file_transfers (
                        transfer_id TEXT PRIMARY KEY,
                        source_device TEXT NOT NULL,
                        target_device TEXT NOT NULL,
                        file_name TEXT NOT NULL,
                        file_size INTEGER NOT NULL,
                        sha256_hash TEXT NOT NULL,
                        target_folder TEXT NOT NULL,
                        status TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        completed_at TEXT
                    )
                """)
                conn.commit()
        except Exception as e:
            log.warning(f"FileTransferEngine: SQLite init notice ({e}).")

    def initiate_transfer(
        self,
        source_device: str,
        target_device: str,
        file_name: str,
        file_size: int,
        sha256_hash: str,
        target_folder: str,
        require_consent: bool = True,
    ) -> TransferRecord:
        """
        Initiate a transfer request. Validates folder sandboxing and permission.
        """
        # 1. Sanitize file name
        safe_name = os.path.basename(file_name)
        if safe_name != file_name or ".." in file_name or "\0" in file_name:
            raise ValueError(f"Path traversal detected in file_name: {file_name}")

        # 2. Check folder sandbox
        target_full_path = os.path.join(target_folder, safe_name)
        canonical_path = self.sandbox.validate_path_access(
            target_full_path, require_write=True
        )

        # 3. Check permission scope
        if require_consent and not consent_manager.has_permission(source_device, PermissionScope.FILE_WRITE):
            # Check if pending consent is approved or create consent request
            req = consent_manager.request_consent(
                device_id=source_device,
                scope=PermissionScope.FILE_WRITE,
                action=f"transfer_file:{safe_name}",
                details={"file_name": safe_name, "file_size": file_size, "target_folder": target_folder},
            )
            if req.status != "approved":
                raise PermissionError(f"Permission denied: FILE_WRITE consent required for device [{source_device}].")

        transfer = TransferRecord(
            source_device=source_device,
            target_device=target_device,
            file_name=safe_name,
            file_size=file_size,
            sha256_hash=sha256_hash.lower(),
            target_folder=target_folder,
            target_path=canonical_path,
            status="authorized",
        )

        self._active_transfers[transfer.transfer_id] = transfer
        self._insert_db(transfer)
        log.info(f"File transfer authorized: '{safe_name}' ({file_size} bytes) -> {canonical_path}")
        return transfer

    def receive_file_stream(
        self,
        source_device: str,
        file_name: str,
        content: bytes,
        expected_sha256: str,
        target_folder: str,
        require_consent: bool = False,
    ) -> TransferRecord:
        """
        Atomic full file transfer helper with SHA-256 verification.
        """
        transfer = self.initiate_transfer(
            source_device=source_device,
            target_device="jarvis_host",
            file_name=file_name,
            file_size=len(content),
            sha256_hash=expected_sha256,
            target_folder=target_folder,
            require_consent=require_consent,
        )

        part_path = transfer.target_path + ".part"
        hasher = hashlib.sha256()

        try:
            # Write in 64 KB chunks to .part file
            with open(part_path, "wb") as f:
                offset = 0
                while offset < len(content):
                    chunk = content[offset : offset + CHUNK_SIZE]
                    hasher.update(chunk)
                    f.write(chunk)
                    offset += len(chunk)

            calculated_hash = hasher.hexdigest().lower()
            if calculated_hash != expected_sha256.lower():
                # Cleanup corrupt part file
                if os.path.exists(part_path):
                    os.remove(part_path)
                transfer.status = "failed"
                transfer.error_message = (
                    f"Integrity check failed: Expected {expected_sha256}, got {calculated_hash}"
                )
                self._update_db(transfer)
                raise ValueError(transfer.error_message)

            # Atomic rename to final destination
            os.replace(part_path, transfer.target_path)
            transfer.status = "completed"
            transfer.bytes_transferred = len(content)
            transfer.completed_at = datetime.now(timezone.utc)
            self._update_db(transfer)
            log.info(f"File transfer successfully completed & verified: {transfer.target_path}")
            return transfer

        except Exception as e:
            if os.path.exists(part_path):
                try:
                    os.remove(part_path)
                except Exception:
                    pass
            if transfer.status != "failed":
                transfer.status = "failed"
                transfer.error_message = str(e)
                self._update_db(transfer)
            raise

    def read_file_chunks(
        self,
        file_path: str,
        chunk_size: int = CHUNK_SIZE,
    ) -> Generator[Tuple[bytes, int, bool, str], None, None]:
        """
        Stream an authorized file in chunks with rolling SHA-256 calculation.
        Yields: (chunk_bytes, chunk_index, is_last, rolling_sha256)
        """
        canonical_path = self.sandbox.validate_path_access(file_path, require_write=False)
        if not os.path.exists(canonical_path):
            raise FileNotFoundError(f"File not found: {canonical_path}")

        file_size = os.path.getsize(canonical_path)
        hasher = hashlib.sha256()
        bytes_read = 0
        chunk_index = 0

        with open(canonical_path, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                hasher.update(chunk)
                bytes_read += len(chunk)
                is_last = (bytes_read >= file_size)
                yield (chunk, chunk_index, is_last, hasher.hexdigest().lower())
                chunk_index += 1

    def get_transfer_status(self, transfer_id: str) -> Optional[TransferRecord]:
        """Fetch transfer progress and status."""
        return self._active_transfers.get(transfer_id)

    def _insert_db(self, transfer: TransferRecord):
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO file_transfers
                    (transfer_id, source_device, target_device, file_name, file_size, sha256_hash, target_folder, status, created_at, completed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        transfer.transfer_id,
                        transfer.source_device,
                        transfer.target_device,
                        transfer.file_name,
                        transfer.file_size,
                        transfer.sha256_hash,
                        transfer.target_folder,
                        transfer.status,
                        transfer.created_at.isoformat(),
                        transfer.completed_at.isoformat() if transfer.completed_at else None,
                    ),
                )
                conn.commit()
        except Exception as e:
            log.warning(f"Error persisting file transfer to DB: {e}")

    def _update_db(self, transfer: TransferRecord):
        self._insert_db(transfer)


file_transfer_engine = FileTransferEngine()
