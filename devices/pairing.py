"""
JARVIS AI — Cryptographic Pairing Handshake Engine
==================================================
Manages 6-digit one-time PIN generation, TTL enforcement (5-minute window),
and initial mutual authentication between JARVIS and connecting devices.
"""

import uuid
import secrets
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timezone, timedelta
from pydantic import BaseModel, Field

from devices.device_registry.registry import (
    DeviceRecord,
    DeviceType,
    DeviceStatus,
    device_registry,
)
from devices.auth import token_authenticator
from core.logger import log


class PairingPinRecord(BaseModel):
    pin: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime
    is_consumed: bool = False


class PairingEngine:
    """Coordinates one-time 6-digit PIN pairing handshakes with strict TTL."""

    def __init__(self, default_ttl_seconds: int = 300):
        self.default_ttl = default_ttl_seconds
        self._pins: Dict[str, PairingPinRecord] = {}

    def generate_pairing_pin(self, ttl_seconds: Optional[int] = None) -> str:
        """
        Generate a cryptographically secure 6-digit one-time PIN.
        TTL defaults to 300 seconds (5 minutes).
        """
        ttl = ttl_seconds or self.default_ttl
        pin = f"{secrets.randbelow(900000) + 100000}"
        now = datetime.now(timezone.utc)
        record = PairingPinRecord(
            pin=pin,
            created_at=now,
            expires_at=now + timedelta(seconds=ttl),
            is_consumed=False,
        )
        self._pins[pin] = record
        # Synchronize with legacy device_registry for backward compatibility
        device_registry._pairing_codes[pin] = {
            "created_at": now,
        }
        log.info(f"Generated 6-digit device pairing PIN: {pin} (valid for {ttl}s)")
        return pin

    def verify_and_pair(
        self,
        pin: str,
        name: str,
        device_type: DeviceType,
        ip_address: Optional[str] = None,
        capabilities: Optional[List[str]] = None,
    ) -> Tuple[DeviceRecord, str]:
        """
        Verify the PIN, consume it immediately, and return (DeviceRecord, raw_token).
        
        Raises:
            ValueError: If PIN is invalid, already consumed, or expired.
        """
        now = datetime.now(timezone.utc)
        record = self._pins.get(pin)

        # Fallback check on legacy registry codes
        if not record and pin in device_registry._pairing_codes:
            record = PairingPinRecord(
                pin=pin,
                created_at=device_registry._pairing_codes[pin]["created_at"],
                expires_at=now + timedelta(seconds=self.default_ttl),
                is_consumed=False,
            )
            self._pins[pin] = record

        if not record:
            raise ValueError("Invalid pairing PIN: Code does not exist.")

        if record.is_consumed:
            raise ValueError("Invalid pairing PIN: Code has already been consumed.")

        if now > record.expires_at:
            del self._pins[pin]
            if pin in device_registry._pairing_codes:
                del device_registry._pairing_codes[pin]
            raise ValueError("Expired pairing PIN: The 5-minute pairing window has elapsed.")

        # Immediately consume PIN to enforce one-time usage
        record.is_consumed = True
        del self._pins[pin]
        if pin in device_registry._pairing_codes:
            del device_registry._pairing_codes[pin]

        # Generate credentials
        dev_id = f"dev-{uuid.uuid4().hex[:8]}"
        raw_token = secrets.token_urlsafe(32)

        caps = capabilities or ["notification", "remote_task", "file_transfer"]
        device_record = DeviceRecord(
            device_id=dev_id,
            name=name,
            device_type=device_type,
            auth_token=raw_token,
            status=DeviceStatus.ONLINE,
            capabilities=caps,
        )

        # Register in DeviceRegistry (Phase 7 backward-compat)
        device_registry._devices[dev_id] = device_record

        # Register in TokenAuthenticator (Phase 21 persistent security)
        token_authenticator.register_device(
            device_id=dev_id,
            raw_token=raw_token,
            name=name,
            device_type=device_type.value,
            ip_address=ip_address,
            capabilities=caps,
        )

        log.info(f"Device successfully paired: '{name}' ({device_type.value}) -> ID: {dev_id}")
        return device_record, raw_token

    def clean_expired_pins(self):
        """Remove expired PINs from memory."""
        now = datetime.now(timezone.utc)
        expired = [p for p, rec in self._pins.items() if now > rec.expires_at or rec.is_consumed]
        for p in expired:
            del self._pins[p]
            if p in device_registry._pairing_codes:
                del device_registry._pairing_codes[p]


pairing_engine = PairingEngine()
