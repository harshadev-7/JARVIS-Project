"""
JARVIS AI — Unified Device Manager & Mesh Coordinator
=====================================================
Integrates all Phase 7 and Phase 21 multi-device subsystems into a single,
cohesive coordinator:
- PairingEngine (6-digit cryptographic handshake)
- LocalDiscoveryEngine (LAN UDP beaconing & TCP probing)
- TokenAuthenticator (SHA-256 tokens & session telemetry)
- AuthorizedFolderSandbox (Directory whitelisting & traversal shielding)
- ConsentManager (Zero-Trust scope validation & two-phase consent)
- FileTransferEngine (Chunked 64 KB transfers & streaming SHA-256 verification)
- NotificationSyncEngine (Approved category filtering & offline queuing)
- DeviceRegistry (Backwards compatibility with Phase 7 remote task dispatch)
"""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone

from devices.device_registry.registry import (
    DeviceRecord,
    DeviceType,
    DeviceStatus,
    RemoteTaskPayload,
    device_registry,
)
from devices.folder_access import authorized_folder_sandbox, AuthorizedFolderSandbox
from devices.permission import consent_manager, PermissionScope, ConsentManager, ConsentRequest
from devices.auth import token_authenticator, TokenAuthenticator
from devices.discovery import discovery_engine, LocalDiscoveryEngine
from devices.pairing import pairing_engine, PairingEngine
from devices.file_transfer import file_transfer_engine, FileTransferEngine, TransferRecord
from devices.notification_sync import (
    notification_sync_engine,
    NotificationSyncEngine,
    NotificationItem,
)
from devices.crypto_channel import CryptoChannel
from devices.mutual_auth import mutual_authenticator, MutualAuthenticator, AuthChallenge, MutualAuthResult
from devices.session_manager import session_manager, NetworkSessionManager, SessionRecord
from devices.command_whitelist import command_gatekeeper, CommandWhitelistGatekeeper, RemoteCommandRequest, RemoteCommandResult
from devices.connection_monitor import connection_monitor, ConnectionMonitor, PeerMetrics
from devices.encrypted_transfer import encrypted_transfer_engine, EncryptedTransferEngine
from core.logger import log


class DeviceManager:
    """Unified coordinator for all authorized device integration services."""

    def __init__(
        self,
        registry=device_registry,
        sandbox=authorized_folder_sandbox,
        consent=consent_manager,
        auth=token_authenticator,
        discovery=discovery_engine,
        pairing=pairing_engine,
        transfer=file_transfer_engine,
        notification=notification_sync_engine,
        mutual_auth=mutual_authenticator,
        session_mgr=session_manager,
        command_gate=command_gatekeeper,
        monitor=connection_monitor,
        encrypted_transfer=encrypted_transfer_engine,
    ):
        self.registry = registry
        self.sandbox = sandbox
        self.consent = consent
        self.auth = auth
        self.discovery = discovery
        self.pairing = pairing
        self.transfer = transfer
        self.notification = notification
        self.mutual_auth = mutual_auth
        self.session_mgr = session_mgr
        self.command_gate = command_gate
        self.monitor = monitor
        self.encrypted_transfer = encrypted_transfer

    # ── 1. Pairing & Authentication ──────────────────────────────────────────

    def generate_pairing_pin(self, ttl_seconds: int = 300) -> str:
        """Create a 6-digit one-time cryptographic pairing PIN with 5-min TTL."""
        return self.pairing.generate_pairing_pin(ttl_seconds=ttl_seconds)

    def pair_device(
        self,
        pairing_code: str,
        name: str,
        device_type: DeviceType,
        ip_address: Optional[str] = None,
        capabilities: Optional[List[str]] = None,
    ) -> Tuple[DeviceRecord, str]:
        """Verify pairing code and register device, returning (DeviceRecord, raw_token)."""
        record, raw_token = self.pairing.verify_and_pair(
            pin=pairing_code,
            name=name,
            device_type=device_type,
            ip_address=ip_address,
            capabilities=capabilities,
        )
        return record, raw_token

    def authenticate_token(self, raw_token: str) -> Optional[str]:
        """Authenticate bearer token, returning device_id if valid."""
        return self.auth.authenticate_token(raw_token)

    def disconnect_device(self, device_id: str) -> bool:
        """Revoke device authorization across all subsystems."""
        self.auth.revoke_device(device_id)
        self.consent.revoke_all(device_id)
        return self.registry.disconnect_device(device_id)

    # ── 2. Device Status & Heartbeat ─────────────────────────────────────────

    def heartbeat(
        self,
        device_id: str,
        battery_level: Optional[float] = None,
        ip_address: Optional[str] = None,
    ) -> bool:
        """Update presence and telemetry."""
        reg_ok = self.registry.heartbeat(device_id, battery_level=battery_level)
        self.auth.update_heartbeat(device_id, battery_level=battery_level, ip_address=ip_address)
        return reg_ok

    def list_devices(self) -> List[DeviceRecord]:
        """List all registered devices with current online/offline status."""
        self.auth.check_timeouts(timeout_seconds=60)
        return self.registry.list_devices()

    def find_device(self, name_or_id: str) -> Optional[DeviceRecord]:
        """Find device by ID or partial name match."""
        return self.registry._devices.get(name_or_id) or self.registry.find_device_by_name(name_or_id)

    # ── 3. Local LAN Discovery ───────────────────────────────────────────────

    def scan_network(self) -> List[Dict[str, Any]]:
        """Return list of nearby discovered devices."""
        return self.discovery.get_discovered_devices()

    def broadcast_presence(self, host_port: int = 8000) -> bool:
        """Broadcast J.A.R.V.I.S. host beacon on local subnet."""
        return self.discovery.broadcast_presence(
            device_id="jarvis-core",
            name="JARVIS-Host",
            device_type="windows",
            service_port=host_port,
        )

    # ── 4. Folder Sandboxing ─────────────────────────────────────────────────

    def authorize_folder(self, folder_path: str, access_level: str = "read_only") -> Dict[str, Any]:
        """Whitelist directory for device file access."""
        return self.sandbox.authorize_folder(folder_path, access_level=access_level)

    def revoke_folder(self, folder_id_or_path: str) -> bool:
        """Revoke directory authorization."""
        return self.sandbox.revoke_folder(folder_id_or_path)

    def list_authorized_folders(self) -> List[Dict[str, Any]]:
        """List all whitelisted directories."""
        return self.sandbox.list_authorized_folders()

    # ── 5. Permission & Consent ──────────────────────────────────────────────

    def grant_permission(self, device_id: str, scope: PermissionScope):
        """Grant permission scope to device."""
        self.consent.grant_scope(device_id, scope)

    def has_permission(self, device_id: str, scope: PermissionScope) -> bool:
        """Check if device has scope."""
        return self.consent.has_permission(device_id, scope)

    def request_consent(
        self, device_id: str, scope: PermissionScope, action: str, details: Optional[Dict[str, Any]] = None
    ) -> ConsentRequest:
        """Request affirmative user consent for consequential operation."""
        return self.consent.request_consent(device_id, scope, action, details)

    # ── 6. Verified File Transfers ───────────────────────────────────────────

    def transfer_file(
        self,
        source_device: str,
        file_name: str,
        content: bytes,
        expected_sha256: str,
        target_folder: str,
        require_consent: bool = False,
    ) -> TransferRecord:
        """Receive and verify a streamed file."""
        return self.transfer.receive_file_stream(
            source_device=source_device,
            file_name=file_name,
            content=content,
            expected_sha256=expected_sha256,
            target_folder=target_folder,
            require_consent=require_consent,
        )

    # ── 7. Notification Synchronization ──────────────────────────────────────

    def send_notification(
        self,
        source_device: str,
        target_device: str,
        title: str,
        message: str,
        category: str = "system_event",
        priority: str = "normal",
    ) -> NotificationItem:
        """Dispatch filtered notification to target device."""
        target_dev = self.find_device(target_device)
        is_online = (target_dev.status == DeviceStatus.ONLINE) if target_dev else True
        target_id = target_dev.device_id if target_dev else target_device

        return self.notification.send_notification(
            source_device=source_device,
            target_device=target_id,
            title=title,
            message=message,
            category=category,
            priority=priority,
            is_target_online=is_online,
        )

    def get_queued_notifications(self, device_id: str) -> List[NotificationItem]:
        """Fetch pending notifications for device upon reconnect."""
        return self.notification.get_queued_notifications(device_id)

    # ── 8. Remote Task Dispatching (Phase 7 Backwards-Compatibility) ──────────

    def dispatch_remote_task(
        self, target_name_or_id: str, action: str, parameters: Optional[Dict[str, Any]] = None
    ) -> RemoteTaskPayload:
        """Dispatch a task to a registered remote device."""
        return self.registry.dispatch_remote_task(
            target_name_or_id=target_name_or_id, action=action, parameters=parameters
        )

    # ── 9. Phase 22: Secure Networking & Mutual Authentication ───────────────

    def issue_auth_challenge(self, device_id: str) -> Tuple[AuthChallenge, str]:
        """Issue an ephemeral cryptographic challenge to a paired device."""
        dev = self.find_device(device_id)
        if not dev:
            raise LookupError(f"Device '{device_id}' is not registered.")
        master_key = self.mutual_auth.derive_device_master_key(dev.auth_token)
        return self.mutual_auth.issue_challenge(dev.device_id, master_key)

    def verify_auth_response(self, challenge_id: str, client_response_hex: str) -> MutualAuthResult:
        """Verify client challenge response, register active session, and return ticket."""
        res = self.mutual_auth.verify_response(challenge_id, client_response_hex)
        if res.success and res.session_id and res.session_key_hex:
            session_key = bytes.fromhex(res.session_key_hex)
            self.session_mgr.register_session(
                session_id=res.session_id,
                device_id=res.device_id,
                session_key=session_key,
            )
        return res

    async def execute_remote_command(
        self,
        command_name: str,
        device_id: str,
        session_id: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> RemoteCommandResult:
        """Verify session validity and execute safe whitelisted remote command."""
        session = self.session_mgr.get_session(session_id)
        if not session or session.device_id != device_id:
            return RemoteCommandResult(
                success=False,
                command_name=command_name,
                error="Authentication failed: Invalid or expired network session.",
            )

        self.session_mgr.touch_session(session_id)
        req = RemoteCommandRequest(
            command_name=command_name,
            device_id=device_id,
            session_id=session_id,
            parameters=parameters or {},
        )
        return await self.command_gate.execute_command(req)

    def record_peer_heartbeat(self, device_id: str, rtt_ms: Optional[float] = None) -> PeerMetrics:
        """Record live latency and presence telemetry for connected peer."""
        self.heartbeat(device_id)
        return self.monitor.record_heartbeat(device_id, rtt_ms=rtt_ms)

    def get_peer_telemetry(self) -> List[PeerMetrics]:
        """List live connection metrics across active peer network."""
        return self.monitor.list_all_metrics()

    def revoke_session(self, session_id: str) -> bool:
        """Instantly terminate an active encrypted session."""
        return self.session_mgr.revoke_session(session_id)


device_manager = DeviceManager()
