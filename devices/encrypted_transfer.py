"""
JARVIS AI — Authenticated Encrypted File Transfer Engine
========================================================
Combines AES-256-GCM authenticated chunk encryption with strict directory
sandboxing. Guarantees end-to-end confidentiality and tamper resistance.
"""

import os
import hashlib
from typing import Dict, Any, Generator, Tuple, Optional
from datetime import datetime, timezone

from devices.crypto_channel import CryptoChannel
from devices.folder_access import authorized_folder_sandbox
from core.logger import log

ENCRYPTED_CHUNK_SIZE = 64 * 1024  # 64 KB chunk size before framing


class EncryptedTransferEngine:
    """Manages AEAD-encrypted chunked file transfers over untrusted LAN."""

    def __init__(self, sandbox=authorized_folder_sandbox):
        self.sandbox = sandbox

    def send_encrypted_file(
        self,
        file_path: str,
        channel: CryptoChannel,
    ) -> Generator[Tuple[bytes, int, bool, str], None, None]:
        """
        Reads an authorized file, encrypts chunks with AES-256-GCM,
        and yields (encrypted_packet, chunk_index, is_last, plain_sha256).
        """
        canonical_path = self.sandbox.validate_path_access(file_path, require_write=False)
        if not os.path.exists(canonical_path):
            raise FileNotFoundError(f"Source file not found: {canonical_path}")

        file_size = os.path.getsize(canonical_path)
        hasher = hashlib.sha256()
        bytes_read = 0
        chunk_idx = 0

        with open(canonical_path, "rb") as f:
            while True:
                plain_chunk = f.read(ENCRYPTED_CHUNK_SIZE)
                if not plain_chunk:
                    break
                hasher.update(plain_chunk)
                bytes_read += len(plain_chunk)
                is_last = (bytes_read >= file_size)

                # Encrypt chunk via CryptoChannel (includes nonce and AEAD tag)
                encrypted_packet = channel.encrypt_payload(plain_chunk)
                yield (encrypted_packet, chunk_idx, is_last, hasher.hexdigest().lower())
                chunk_idx += 1

    def receive_encrypted_file(
        self,
        file_name: str,
        target_folder: str,
        encrypted_packets: List_or_iterable,
        expected_sha256: str,
        channel: CryptoChannel,
    ) -> Dict[str, Any]:
        """
        Decrypts stream of encrypted chunks, verifies AEAD tags on every chunk,
        verifies cumulative SHA-256 manifest, and atomically writes to target.
        """
        # 1. Sanitize file name
        safe_name = os.path.basename(file_name)
        if safe_name != file_name or ".." in file_name or "\0" in file_name:
            raise ValueError(f"Path traversal detected in file_name: {file_name}")

        # 2. Check directory sandbox
        target_path = os.path.join(target_folder, safe_name)
        canonical_path = self.sandbox.validate_path_access(target_path, require_write=True)
        part_path = canonical_path + ".part"

        hasher = hashlib.sha256()
        bytes_written = 0

        try:
            with open(part_path, "wb") as f:
                for packet in encrypted_packets:
                    # Decrypt with AEAD tag check and replay protection
                    plain_chunk = channel.decrypt_payload(packet)
                    hasher.update(plain_chunk)
                    f.write(plain_chunk)
                    bytes_written += len(plain_chunk)

            calculated_hash = hasher.hexdigest().lower()
            if calculated_hash != expected_sha256.lower():
                if os.path.exists(part_path):
                    os.remove(part_path)
                raise ValueError(
                    f"Encrypted transfer checksum mismatch: Expected {expected_sha256}, got {calculated_hash}"
                )

            # Atomic rename upon successful integrity seal verification
            os.replace(part_path, canonical_path)
            log.info(f"Encrypted file transfer verified and saved: {canonical_path} ({bytes_written} bytes)")

            return {
                "success": True,
                "file_path": canonical_path,
                "bytes_transferred": bytes_written,
                "sha256": calculated_hash,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            if os.path.exists(part_path):
                try:
                    os.remove(part_path)
                except Exception:
                    pass
            log.warning(f"Encrypted transfer aborted: {e}")
            raise


# Python type alias helper
List_or_iterable = Any

encrypted_transfer_engine = EncryptedTransferEngine()
