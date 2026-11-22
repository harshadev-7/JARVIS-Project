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
]
