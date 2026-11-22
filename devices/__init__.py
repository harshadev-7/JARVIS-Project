"""
JARVIS AI — Authorized Device Integration Subsystem
===================================================
Public API exporting all device networking, security, folder sandboxing,
pairing, and synchronization services.
"""

from devices.device_registry.registry import (
    DeviceRegistry,
    device_registry,
    DeviceRecord,
    DeviceType,
    DeviceStatus,
    RemoteTaskPayload,
)

from devices.folder_access import (
    AuthorizedFolderSandbox,
    authorized_folder_sandbox,
)

from devices.permission import (
    PermissionScope,
    ConsentStatus,
    ConsentRequest,
    ConsentManager,
    consent_manager,
)

from devices.auth import (
    TokenAuthenticator,
    token_authenticator,
    DeviceSession,
)

from devices.discovery import (
    LocalDiscoveryEngine,
    discovery_engine,
)

from devices.pairing import (
    PairingEngine,
    pairing_engine,
    PairingPinRecord,
)

from devices.file_transfer import (
    FileTransferEngine,
    file_transfer_engine,
    TransferRecord,
)

from devices.notification_sync import (
    NotificationSyncEngine,
    notification_sync_engine,
    NotificationItem,
)

from devices.device_manager import (
    DeviceManager,
    device_manager,
)

from devices.device_agent import (
    DeviceAgent,
)

from devices.crypto_channel import (
    CryptoChannel,
    CryptoChannelError,
    ReplayAttackError,
    TamperDetectedError,
)

from devices.mutual_auth import (
    MutualAuthenticator,
    mutual_authenticator,
    AuthChallenge,
    MutualAuthResult,
)

from devices.session_manager import (
    NetworkSessionManager,
    session_manager,
    SessionRecord,
)

from devices.command_whitelist import (
    CommandWhitelistGatekeeper,
    command_gatekeeper,
    RemoteCommandRequest,
    RemoteCommandResult,
)

from devices.connection_monitor import (
    ConnectionMonitor,
    connection_monitor,
    PeerMetrics,
)

from devices.encrypted_transfer import (
    EncryptedTransferEngine,
    encrypted_transfer_engine,
)

__all__ = [
    # Core Coordinator
    "DeviceManager",
    "device_manager",
    # Agent
    "DeviceAgent",
    # Legacy & Mesh Registry
    "DeviceRegistry",
    "device_registry",
    "DeviceRecord",
    "DeviceType",
    "DeviceStatus",
    "RemoteTaskPayload",
    # Sandboxing & Permissions
    "AuthorizedFolderSandbox",
    "authorized_folder_sandbox",
    "PermissionScope",
    "ConsentStatus",
    "ConsentRequest",
    "ConsentManager",
    "consent_manager",
    # Auth & Pairing
    "TokenAuthenticator",
    "token_authenticator",
    "DeviceSession",
    "PairingEngine",
    "pairing_engine",
    "PairingPinRecord",
    # LAN Discovery
    "LocalDiscoveryEngine",
    "discovery_engine",
    # File Transfers & Notifications
    "FileTransferEngine",
    "file_transfer_engine",
    "TransferRecord",
    "NotificationSyncEngine",
    "notification_sync_engine",
    "NotificationItem",
    # Phase 22 Secure Networking
    "CryptoChannel",
    "CryptoChannelError",
    "ReplayAttackError",
    "TamperDetectedError",
    "MutualAuthenticator",
    "mutual_authenticator",
    "AuthChallenge",
    "MutualAuthResult",
    "NetworkSessionManager",
    "session_manager",
    "SessionRecord",
    "CommandWhitelistGatekeeper",
    "command_gatekeeper",
    "RemoteCommandRequest",
    "RemoteCommandResult",
    "ConnectionMonitor",
    "connection_monitor",
    "PeerMetrics",
    "EncryptedTransferEngine",
    "encrypted_transfer_engine",
]
