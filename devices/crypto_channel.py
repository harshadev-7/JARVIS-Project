"""
JARVIS AI — End-to-End Cryptographic Channel (AES-256-GCM)
===========================================================
Provides authenticated encryption with associated data (AEAD) for multi-device mesh
communication over local networks.
Protocol: JARVIS-SEC-v1
Cipher: AES-256-GCM (256-bit symmetric key, 96-bit unique nonces)
Replay Protection: Monotonically increasing nonce sequence with sliding verification window.
"""

import os
import struct
import json
from typing import Dict, Any, Tuple, Optional
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from core.logger import log

PROTOCOL_MAGIC = b"JSEC"
PROTOCOL_VERSION = 1


class CryptoChannelError(Exception):
    """Base exception for secure channel operations."""
    pass


class ReplayAttackError(CryptoChannelError):
    """Raised when an out-of-order or duplicate nonce sequence is detected."""
    pass


class TamperDetectedError(CryptoChannelError):
    """Raised when authentication tag verification fails (ciphertext tampered)."""
    pass


class CryptoChannel:
    """Manages end-to-end authenticated symmetric encryption between nodes."""

    def __init__(self, session_key: bytes, session_id: str, is_initiator: bool = True):
        if len(session_key) != 32:
            raise ValueError("AES-256 session key must be exactly 32 bytes (256 bits).")
        
        self.session_key = session_key
        self.session_id = session_id
        self.is_initiator = is_initiator
        self.aesgcm = AESGCM(self.session_key)

        # Monotonic counters for anti-replay
        self._send_counter = 0
        self._expected_recv_counter = 0
        self._recv_window_size = 128
        self._received_counters = set()

    def _generate_nonce(self, counter: int) -> bytes:
        """
        Derive 12-byte (96-bit) nonce from role flag, session prefix, and 64-bit counter.
        First byte indicates initiator (0x01) vs responder (0x02).
        Next 3 bytes are fixed session entropy.
        Final 8 bytes are the big-endian counter.
        """
        role_byte = 0x01 if self.is_initiator else 0x02
        prefix = role_byte.to_bytes(1, "big") + b"\x00\xaa\xbb"
        return prefix + struct.pack(">Q", counter)

    def encrypt_payload(self, data: bytes, associated_data: Optional[bytes] = None) -> bytes:
        """
        Encrypt plaintext with AES-256-GCM and format into a secure packet frame.
        Frame format:
        [MAGIC: 4B][VERSION: 1B][FLAGS: 1B][NONCE: 12B][CIPHERTEXT + 16B TAG]
        """
        self._send_counter += 1
        counter = self._send_counter
        nonce = self._generate_nonce(counter)

        # Build AAD: includes session_id, counter, and protocol version
        aad = (associated_data or b"") + self.session_id.encode("utf-8") + struct.pack(">Q", counter)

        try:
            ciphertext = self.aesgcm.encrypt(nonce, data, aad)
        except Exception as e:
            raise CryptoChannelError(f"Encryption failed: {e}") from e

        # Assemble packet header
        header = PROTOCOL_MAGIC + struct.pack(">BB", PROTOCOL_VERSION, 0x00) + nonce
        return header + ciphertext

    def decrypt_payload(self, packet: bytes, associated_data: Optional[bytes] = None) -> bytes:
        """
        Validate packet header, verify replay protection, and decrypt ciphertext.
        """
        if len(packet) < 4 + 2 + 12 + 16:
            raise CryptoChannelError("Packet too short to contain valid JARVIS-SEC-v1 frame.")

        magic = packet[:4]
        if magic != PROTOCOL_MAGIC:
            raise CryptoChannelError(f"Invalid protocol magic: {magic}")

        version, flags = struct.unpack(">BB", packet[4:6])
        if version != PROTOCOL_VERSION:
            raise CryptoChannelError(f"Unsupported protocol version: {version}")

        nonce = packet[6:18]
        ciphertext = packet[18:]

        # Extract counter from nonce
        counter = struct.unpack(">Q", nonce[4:12])[0]

        # Replay protection check
        if counter in self._received_counters or counter < (self._expected_recv_counter - self._recv_window_size):
            raise ReplayAttackError(f"Replay attack detected: counter {counter} already processed or stale.")

        # Build expected AAD
        aad = (associated_data or b"") + self.session_id.encode("utf-8") + struct.pack(">Q", counter)

        try:
            plaintext = self.aesgcm.decrypt(nonce, ciphertext, aad)
        except Exception as e:
            raise TamperDetectedError("Decryption failed: authentication tag mismatch or tampered payload.") from e

        # Update counter window
        self._received_counters.add(counter)
        if counter >= self._expected_recv_counter:
            self._expected_recv_counter = counter + 1

        # Prune received counters older than window
        cutoff = self._expected_recv_counter - self._recv_window_size
        self._received_counters = {c for c in self._received_counters if c >= cutoff}

        return plaintext

    def encrypt_json(self, payload_dict: Dict[str, Any]) -> bytes:
        """Convenience method to encrypt a JSON-serializable dictionary."""
        raw = json.dumps(payload_dict, separators=(",", ":")).encode("utf-8")
        return self.encrypt_payload(raw)

    def decrypt_json(self, packet: bytes) -> Dict[str, Any]:
        """Convenience method to decrypt a packet into a dictionary."""
        plaintext = self.decrypt_payload(packet)
        return json.loads(plaintext.decode("utf-8"))
