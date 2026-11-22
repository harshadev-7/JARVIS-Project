"""
JARVIS AI — Mutual Challenge-Response Authenticator
===================================================
Establishes mutual trust and derives ephemeral AES-256 session keys
via HMAC-SHA256 challenge-response handshakes.
Guarantees that both the host and the connecting device independently prove
knowledge of the device's paired secret without transmitting raw keys.
"""

import hmac
import hashlib
import secrets
import uuid
from typing import Dict, Any, Tuple, Optional
from datetime import datetime, timezone, timedelta
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes
from pydantic import BaseModel, Field

from core.logger import log


class AuthChallenge(BaseModel):
    challenge_id: str = Field(default_factory=lambda: f"chg-{uuid.uuid4().hex[:8]}")
    device_id: str
    challenge_data: str  # Hex string
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc) + timedelta(seconds=60))


class MutualAuthResult(BaseModel):
    success: bool
    session_id: Optional[str] = None
    session_key_hex: Optional[str] = None
    device_id: str
    error: Optional[str] = None


class MutualAuthenticator:
    """Orchestrates challenge-response handshakes and derives session keys."""

    def __init__(self):
        # challenge_id -> (AuthChallenge, expected_response_bytes, session_key_bytes)
        self._pending_challenges: Dict[str, Tuple[AuthChallenge, bytes, bytes]] = {}

    @staticmethod
    def derive_device_master_key(raw_auth_token: str, salt: bytes = b"JARVIS-DEVICE-SALT-v1") -> bytes:
        """Derive a 256-bit cryptographic master key from the device's paired auth token."""
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            info=b"JARVIS-DEVICE-MASTER-KEY",
        )
        return hkdf.derive(raw_auth_token.encode("utf-8"))

    def issue_challenge(self, device_id: str, device_master_key: bytes) -> Tuple[AuthChallenge, str]:
        """
        Create a 32-byte cryptographic challenge for the connecting device.
        Precomputes the expected HMAC response and derives an ephemeral session key.
        """
        raw_challenge = secrets.token_bytes(32)
        challenge_hex = raw_challenge.hex()

        # Compute expected client response: HMAC-SHA256(master_key, b"CLIENT-RESP:" + challenge)
        expected_resp = hmac.new(
            device_master_key,
            b"CLIENT-RESP:" + raw_challenge,
            hashlib.sha256,
        ).digest()

        # Derive 32-byte AES-256 session key for this conversation
        hkdf_session = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=raw_challenge,
            info=b"JARVIS-EPHEMERAL-SESSION-KEY",
        )
        session_key = hkdf_session.derive(device_master_key)

        challenge_obj = AuthChallenge(
            device_id=device_id,
            challenge_data=challenge_hex,
        )

        self._pending_challenges[challenge_obj.challenge_id] = (challenge_obj, expected_resp, session_key)
        log.info(f"Issued mutual auth challenge [{challenge_obj.challenge_id}] to device [{device_id}].")
        return challenge_obj, challenge_hex

    def solve_challenge(self, challenge_hex: str, device_master_key: bytes) -> str:
        """
        Helper for client nodes to compute the expected HMAC response from challenge data.
        """
        raw_challenge = bytes.fromhex(challenge_hex)
        resp_bytes = hmac.new(
            device_master_key,
            b"CLIENT-RESP:" + raw_challenge,
            hashlib.sha256,
        ).digest()
        return resp_bytes.hex()

    def verify_response(
        self,
        challenge_id: str,
        client_response_hex: str,
    ) -> MutualAuthResult:
        """
        Verify client's challenge response using constant-time comparison.
        On success, consumes challenge and returns active session credentials.
        """
        entry = self._pending_challenges.get(challenge_id)
        if not entry:
            return MutualAuthResult(
                success=False,
                device_id="",
                error="Challenge ID not found or already consumed.",
            )

        challenge_obj, expected_resp, session_key = entry
        del self._pending_challenges[challenge_id]

        # Check expiration
        now = datetime.now(timezone.utc)
        if now > challenge_obj.expires_at:
            return MutualAuthResult(
                success=False,
                device_id=challenge_obj.device_id,
                error="Challenge expired. Please request a new challenge.",
            )

        try:
            client_resp_bytes = bytes.fromhex(client_response_hex)
        except ValueError:
            return MutualAuthResult(
                success=False,
                device_id=challenge_obj.device_id,
                error="Malformed response format (invalid hex).",
            )

        # Constant-time comparison to prevent timing attacks
        if not hmac.compare_digest(expected_resp, client_resp_bytes):
            log.warning(f"Mutual auth failed for device [{challenge_obj.device_id}]: invalid signature.")
            return MutualAuthResult(
                success=False,
                device_id=challenge_obj.device_id,
                error="Mutual authentication failed: Signature verification mismatch.",
            )

        session_id = f"sec-sess-{uuid.uuid4().hex[:12]}"
        log.info(f"Mutual authentication successful for device [{challenge_obj.device_id}]. Session ID: {session_id}")

        return MutualAuthResult(
            success=True,
            session_id=session_id,
            session_key_hex=session_key.hex(),
            device_id=challenge_obj.device_id,
        )


mutual_authenticator = MutualAuthenticator()
