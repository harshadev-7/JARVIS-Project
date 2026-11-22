"""
JARVIS AI — Phase 21 Authorized Device Integration Test Suite
=============================================================
Comprehensive unit and integration tests covering:
1. AuthorizedFolderSandbox (whitelisting, system folder rejection, path traversal attacks)
2. ConsentManager & PermissionScope (zero-trust two-phase consent and auditing)
3. TokenAuthenticator (SHA-256 bearer tokens, sessions, revocation, timeouts)
4. LocalDiscoveryEngine (UDP beaconing format, packet parsing, TCP reachability)
5. PairingEngine (6-digit one-time cryptographic PIN, TTL, single-use guarantee)
6. FileTransferEngine (64 KB chunking, SHA-256 integrity, atomic rename, corrupt hash rejection)
7. NotificationSyncEngine (category whitelist, security blocking of OTP/banking, offline queuing)
8. DeviceManager (unified coordinator across all subsystems)
9. DeviceAgent (natural language device orchestration)
10. FastAPI Device Endpoints (REST API verification)
"""

import os
import shutil
import hashlib
import tempfile
import pytest
from datetime import datetime, timezone, timedelta
from fastapi.testclient import TestClient

from devices.folder_access import AuthorizedFolderSandbox
from devices.permission import (
    ConsentManager,
    PermissionScope,
    ConsentStatus,
)
from devices.auth import TokenAuthenticator
from devices.discovery import LocalDiscoveryEngine, DISCOVERY_PROTOCOL
from devices.pairing import PairingEngine
from devices.file_transfer import FileTransferEngine, CHUNK_SIZE
from devices.notification_sync import NotificationSyncEngine
from devices.device_manager import DeviceManager
from devices.device_agent import DeviceAgent
from devices.device_registry.registry import DeviceType, DeviceStatus, DeviceRegistry
from api.app import app


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def temp_dir():
    """Create a temporary directory for sandboxing tests and clean up after."""
    d = tempfile.mkdtemp(prefix="jarvis_test_sandbox_")
    yield os.path.realpath(d)
    if os.path.exists(d):
        shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def temp_db(temp_dir):
    """Temporary SQLite database path."""
    return os.path.join(temp_dir, "test_jarvis.db")


# ── 1. AuthorizedFolderSandbox Tests ─────────────────────────────────────────

def test_folder_sandbox_whitelisting(temp_dir, temp_db):
    """Authorizing a safe directory should succeed and record access level."""
    sandbox = AuthorizedFolderSandbox(db_path=temp_db)
    record = sandbox.authorize_folder(temp_dir, access_level="read_write")

    assert record["folder_id"].startswith("fld-")
    assert record["path"].lower() == temp_dir.lower()
    assert record["access_level"] == "read_write"

    # Listing should show the folder
    folders = sandbox.list_authorized_folders()
    assert len(folders) >= 1
    assert any(f["path"].lower() == temp_dir.lower() for f in folders)


def test_folder_sandbox_system_root_protection(temp_db):
    """Attempting to authorize critical system directories must be rejected."""
    sandbox = AuthorizedFolderSandbox(db_path=temp_db)

    # Windows system root tests
    with pytest.raises(PermissionError):
        sandbox.authorize_folder("C:\\Windows")

    with pytest.raises(PermissionError):
        sandbox.authorize_folder("C:\\Program Files")

    # Unix system root tests
    with pytest.raises(PermissionError):
        sandbox.authorize_folder("/etc")


def test_folder_sandbox_path_traversal_rejection(temp_dir, temp_db):
    """Directory traversal sequences like '..' and null bytes must raise ValueError."""
    sandbox = AuthorizedFolderSandbox(db_path=temp_db)
    sandbox.authorize_folder(temp_dir, access_level="read_write")

    # Traversal sequence
    with pytest.raises(ValueError, match="Path traversal attack detected"):
        sandbox.validate_path_access(os.path.join(temp_dir, "..", "secret.txt"))

    # Null byte attack
    with pytest.raises(ValueError, match="null byte in path"):
        sandbox.validate_path_access(os.path.join(temp_dir, "test\0.txt"))


def test_folder_sandbox_unauthorized_path_access(temp_dir, temp_db):
    """Accessing paths outside any authorized folder must raise PermissionError."""
    sandbox = AuthorizedFolderSandbox(db_path=temp_db)
    # temp_dir is NOT authorized yet
    unauthorized_file = os.path.join(temp_dir, "data.txt")

    with pytest.raises(PermissionError, match="outside all authorized folder sandboxes"):
        sandbox.validate_path_access(unauthorized_file)


def test_folder_sandbox_read_write_enforcement(temp_dir, temp_db):
    """Write operations in read_only directories must be rejected."""
    sandbox = AuthorizedFolderSandbox(db_path=temp_db)
    sandbox.authorize_folder(temp_dir, access_level="read_only")

    target_file = os.path.join(temp_dir, "write_test.txt")
    # Read should succeed
    path = sandbox.validate_path_access(target_file, require_write=False)
    assert path.lower() == os.path.realpath(target_file).lower()

    # Write should raise PermissionError
    with pytest.raises(PermissionError, match="read_only"):
        sandbox.validate_path_access(target_file, require_write=True)


def test_folder_sandbox_revocation(temp_dir, temp_db):
    """Revoking a folder should immediately deny subsequent file access."""
    sandbox = AuthorizedFolderSandbox(db_path=temp_db)
    record = sandbox.authorize_folder(temp_dir, access_level="read_write")
    fid = record["folder_id"]

    assert sandbox.is_path_authorized(os.path.join(temp_dir, "test.txt")) is True
    revoked = sandbox.revoke_folder(fid)
    assert revoked is True
    assert sandbox.is_path_authorized(os.path.join(temp_dir, "test.txt")) is False


# ── 2. ConsentManager & PermissionScope Tests ─────────────────────────────────

def test_consent_manager_grant_and_revoke():
    """Testing granting and revoking scopes for devices."""
    mgr = ConsentManager()
    dev_id = "dev-test-123"

    assert mgr.has_permission(dev_id, PermissionScope.FILE_WRITE) is False
    mgr.grant_scope(dev_id, PermissionScope.FILE_WRITE)
    assert mgr.has_permission(dev_id, PermissionScope.FILE_WRITE) is True

    mgr.revoke_scope(dev_id, PermissionScope.FILE_WRITE)
    assert mgr.has_permission(dev_id, PermissionScope.FILE_WRITE) is False


def test_consent_manager_request_and_approval():
    """Testing two-phase consent requests and manual approval."""
    mgr = ConsentManager()
    dev_id = "dev-phone-456"

    # Request consent when scope not granted
    req = mgr.request_consent(dev_id, PermissionScope.FILE_WRITE, action="transfer:photo.jpg")
    assert req.status == ConsentStatus.PENDING
    assert req.request_id in [r.request_id for r in mgr.get_pending_requests()]

    # Approve consent
    approved = mgr.approve_consent(req.request_id, persist_scope=True)
    assert approved is True
    assert mgr.has_permission(dev_id, PermissionScope.FILE_WRITE) is True
    assert len(mgr.get_pending_requests()) == 0


def test_consent_manager_rejection():
    """Testing consent rejection."""
    mgr = ConsentManager()
    dev_id = "dev-phone-789"

    req = mgr.request_consent(dev_id, PermissionScope.APP_CONTROL, action="launch:game")
    rejected = mgr.reject_consent(req.request_id)
    assert rejected is True
    assert mgr.has_permission(dev_id, PermissionScope.APP_CONTROL) is False


def test_consent_manager_audit_log():
    """Audit log should capture security events."""
    mgr = ConsentManager()
    dev_id = "dev-audit"
    mgr.grant_scope(dev_id, PermissionScope.DEVICE_CONTROL)
    mgr.revoke_all(dev_id)

    log_entries = mgr.get_audit_log()
    assert len(log_entries) >= 2
    events = [entry["event"] for entry in log_entries]
    assert "GRANT_SCOPE" in events
    assert "REVOKE_ALL" in events


# ── 3. TokenAuthenticator Tests ──────────────────────────────────────────────

def test_token_auth_lifecycle(temp_db):
    """Register device, authenticate token, and update telemetry."""
    auth = TokenAuthenticator(db_path=temp_db)
    dev_id = "dev-auth-test"
    raw_token = "secure-random-bearer-token-12345"

    token_hash = auth.register_device(
        device_id=dev_id,
        raw_token=raw_token,
        name="Harsh Phone",
        device_type="android",
        ip_address="192.168.1.50",
    )
    assert token_hash == hashlib.sha256(raw_token.encode("utf-8")).hexdigest()

    # Authenticate valid token
    authed_id = auth.authenticate_token(raw_token)
    assert authed_id == dev_id

    # Authenticate invalid token
    assert auth.authenticate_token("invalid-token") is None

    # Heartbeat
    updated = auth.update_heartbeat(dev_id, battery_level=92.5)
    assert updated is True
    session = auth.get_session(dev_id)
    assert session.battery_level == 92.5


def test_token_auth_revocation(temp_db):
    """Revoking a device must terminate authentication immediately."""
    auth = TokenAuthenticator(db_path=temp_db)
    dev_id = "dev-revoke-test"
    raw_token = "token-to-be-revoked-xyz"

    auth.register_device(dev_id, raw_token, name="Tablet", device_type="android")
    assert auth.authenticate_token(raw_token) == dev_id

    # Revoke
    revoked = auth.revoke_device(dev_id)
    assert revoked is True
    # Subsequent auth must fail
    assert auth.authenticate_token(raw_token) is None


# ── 4. LocalDiscoveryEngine Tests ─────────────────────────────────────────────

def test_discovery_packet_generation_and_parsing():
    """Discovery beacon payload must adhere to JARVIS-DISC-v1 protocol format."""
    engine = LocalDiscoveryEngine()
    payload = engine.create_beacon_payload("dev-1", "Laptop", "windows", 8000)

    assert payload["proto"] == DISCOVERY_PROTOCOL
    assert payload["device_id"] == "dev-1"
    assert payload["name"] == "Laptop"
    assert "nonce" in payload
    assert "timestamp" in payload

    import json
    raw_bytes = json.dumps(payload).encode("utf-8")
    parsed = engine.parse_packet(raw_bytes, ("192.168.1.100", 54321))

    assert parsed is not None
    assert parsed["device_id"] == "dev-1"
    assert parsed["ip"] == "192.168.1.100"

    discovered = engine.get_discovered_devices()
    assert len(discovered) == 1
    assert discovered[0]["device_id"] == "dev-1"


def test_discovery_invalid_packet_ignored():
    """Packets lacking JARVIS-DISC-v1 protocol header must be ignored."""
    engine = LocalDiscoveryEngine()
    import json
    bad_bytes = json.dumps({"proto": "UNKNOWN-PROTO", "device_id": "bad"}).encode("utf-8")
    assert engine.parse_packet(bad_bytes, ("192.168.1.101", 54321)) is None


# ── 5. PairingEngine Tests ───────────────────────────────────────────────────

def test_pairing_handshake_flow():
    """Generating 6-digit PIN and verifying pairing handshake."""
    engine = PairingEngine(default_ttl_seconds=300)
    pin = engine.generate_pairing_pin()

    assert len(pin) == 6
    assert pin.isdigit()

    record, token = engine.verify_and_pair(
        pin=pin,
        name="Companion Tablet",
        device_type=DeviceType.ANDROID,
        capabilities=["display", "file_transfer"],
    )
    assert record.device_id.startswith("dev-")
    assert record.name == "Companion Tablet"
    assert len(token) >= 32

    # Single-use guarantee: PIN cannot be reused
    with pytest.raises(ValueError, match="already been consumed|does not exist"):
        engine.verify_and_pair(pin, "Second Phone", DeviceType.ANDROID)


def test_pairing_expired_pin():
    """Expired PINs must be rejected."""
    engine = PairingEngine(default_ttl_seconds=-10)  # already expired
    pin = engine.generate_pairing_pin(ttl_seconds=-1)

    with pytest.raises(ValueError, match="Expired pairing PIN"):
        engine.verify_and_pair(pin, "Phone", DeviceType.ANDROID)


# ── 6. FileTransferEngine Tests ──────────────────────────────────────────────

def test_file_transfer_successful(temp_dir, temp_db):
    """File transfer with valid SHA-256 and authorized folder must succeed."""
    sandbox = AuthorizedFolderSandbox(db_path=temp_db)
    sandbox.authorize_folder(temp_dir, access_level="read_write")

    transfer_engine = FileTransferEngine(db_path=temp_db, sandbox=sandbox)

    content = b"JARVIS File Transfer Test Payload: " + b"X" * (CHUNK_SIZE + 1024)
    expected_sha256 = hashlib.sha256(content).hexdigest()
    file_name = "test_document.dat"

    record = transfer_engine.receive_file_stream(
        source_device="dev-client",
        file_name=file_name,
        content=content,
        expected_sha256=expected_sha256,
        target_folder=temp_dir,
        require_consent=False,
    )

    assert record.status == "completed"
    assert os.path.exists(record.target_path)
    # Read back and compare content
    with open(record.target_path, "rb") as f:
        saved_bytes = f.read()
    assert saved_bytes == content


def test_file_transfer_corrupt_checksum_rejection(temp_dir, temp_db):
    """Transfer with corrupt content/hash mismatch must fail and delete .part file."""
    sandbox = AuthorizedFolderSandbox(db_path=temp_db)
    sandbox.authorize_folder(temp_dir, access_level="read_write")

    transfer_engine = FileTransferEngine(db_path=temp_db, sandbox=sandbox)

    content = b"Original Content"
    fake_sha256 = "0" * 64
    file_name = "corrupt_file.dat"

    with pytest.raises(ValueError, match="Integrity check failed"):
        transfer_engine.receive_file_stream(
            source_device="dev-client",
            file_name=file_name,
            content=content,
            expected_sha256=fake_sha256,
            target_folder=temp_dir,
            require_consent=False,
        )

    # Neither .part nor final file should remain
    target_path = os.path.join(temp_dir, file_name)
    assert not os.path.exists(target_path)
    assert not os.path.exists(target_path + ".part")


def test_file_transfer_path_traversal_in_filename(temp_dir, temp_db):
    """File transfer with path traversal sequences in file_name must raise ValueError."""
    sandbox = AuthorizedFolderSandbox(db_path=temp_db)
    sandbox.authorize_folder(temp_dir, access_level="read_write")

    transfer_engine = FileTransferEngine(db_path=temp_db, sandbox=sandbox)

    with pytest.raises(ValueError, match="Path traversal detected"):
        transfer_engine.initiate_transfer(
            source_device="dev-client",
            target_device="host",
            file_name="../../escaped.txt",
            file_size=100,
            sha256_hash="abc",
            target_folder=temp_dir,
        )


# ── 7. NotificationSyncEngine Tests ──────────────────────────────────────────

def test_notification_approved_categories():
    """Approved categories (urgent, reminder, alert, system_event) must be dispatched."""
    notif_engine = NotificationSyncEngine()

    notif = notif_engine.send_notification(
        source_device="dev-phone",
        target_device="dev-laptop",
        title="Server Alert",
        message="CPU load exceeded 90%.",
        category="urgent",
        priority="high",
        is_target_online=True,
    )
    assert notif.category == "urgent"
    assert notif.delivered is True


def test_notification_blocked_categories():
    """Sensitive categories (otp, banking, credentials, password) must raise PermissionError."""
    notif_engine = NotificationSyncEngine()

    with pytest.raises(PermissionError, match="blocked by security policy"):
        notif_engine.send_notification(
            source_device="dev-phone",
            target_device="dev-laptop",
            title="Banking Alert",
            message="Your account balance was updated.",
            category="banking",
        )

    with pytest.raises(PermissionError, match="blocked by security policy"):
        notif_engine.send_notification(
            source_device="dev-phone",
            target_device="dev-laptop",
            title="Two Factor",
            message="Your code is 123456",
            category="otp",
        )


def test_notification_otp_leak_content_filter():
    """Notification body containing OTP / verification codes must be rejected."""
    notif_engine = NotificationSyncEngine()

    with pytest.raises(PermissionError, match="sensitive OTP/credentials detected"):
        notif_engine.send_notification(
            source_device="dev-phone",
            target_device="dev-laptop",
            title="System Reminder",
            message="Your verification code is 584920",
            category="reminder",
        )


def test_notification_offline_queuing():
    """Notifications sent to offline devices must be queued and delivered upon retrieval."""
    notif_engine = NotificationSyncEngine()
    offline_device = "dev-offline-tablet"

    # Send notification to offline device
    notif = notif_engine.send_notification(
        source_device="host",
        target_device=offline_device,
        title="Meeting Reminder",
        message="Standup in 10 minutes",
        category="reminder",
        is_target_online=False,
    )
    assert notif.delivered is False

    # Retrieve queued items
    queued = notif_engine.get_queued_notifications(offline_device, mark_delivered=True)
    assert len(queued) == 1
    assert queued[0].notification_id == notif.notification_id
    assert queued[0].delivered is True

    # Queue should now be empty
    assert len(notif_engine.get_queued_notifications(offline_device)) == 0


# ── 8. DeviceManager Unified Coordinator Tests ──────────────────────────────

def test_device_manager_unified_operations(temp_dir, temp_db):
    """DeviceManager coordinates pairing, heartbeat, folders, and dispatch."""
    dm = DeviceManager()

    # 1. Pairing
    pin = dm.generate_pairing_pin()
    record, token = dm.pair_device(pin, "Manager Test Phone", DeviceType.ANDROID)
    assert record.device_id.startswith("dev-")
    assert dm.authenticate_token(token) == record.device_id

    # 2. Heartbeat
    hb_ok = dm.heartbeat(record.device_id, battery_level=78.0)
    assert hb_ok is True

    # 3. Folder sandboxing
    fld = dm.authorize_folder(temp_dir, access_level="read_write")
    assert fld["path"].lower() == temp_dir.lower()

    # 4. Disconnect & Revoke
    disc = dm.disconnect_device(record.device_id)
    assert disc is True
    assert dm.authenticate_token(token) is None


# ── 9. DeviceAgent Natural Language Execution Tests ──────────────────────────

@pytest.mark.asyncio
async def test_device_agent_nlp_pairing():
    """DeviceAgent generates pairing code on natural language command."""
    agent = DeviceAgent()
    result = await agent.execute("pair my android phone with jarvis")
    assert result.success is True
    assert "pairing_pin" in result.data
    assert len(result.data["pairing_pin"]) == 6


@pytest.mark.asyncio
async def test_device_agent_nlp_list_devices():
    """DeviceAgent lists connected devices."""
    agent = DeviceAgent()
    result = await agent.execute("show connected devices")
    assert result.success is True
    assert "devices" in result.data


# ── 10. FastAPI REST Endpoints Tests ────────────────────────────────────────

@pytest.fixture
def client():
    return TestClient(app)


def test_api_pairing_and_folder_lifecycle(client, temp_dir):
    """Test full REST endpoint flow including folder authorization and notifications."""
    # 1. Generate PIN
    pin_res = client.post("/devices/pairing-code")
    assert pin_res.status_code == 200
    pin = pin_res.json()["pairing_code"]

    # 2. Pair device
    pair_res = client.post(
        "/devices/pair",
        json={
            "pairing_code": pin,
            "name": "API Companion",
            "device_type": "android",
            "capabilities": ["notification"],
        },
    )
    assert pair_res.status_code == 200
    device_id = pair_res.json()["device_id"]

    # 3. Heartbeat
    hb_res = client.post(
        "/devices/heartbeat",
        json={"device_id": device_id, "battery_level": 88.0},
    )
    assert hb_res.status_code == 200

    # 4. Folder authorization
    folder_res = client.post(
        "/devices/folders/authorize",
        json={"folder_path": temp_dir, "access_level": "read_write"},
    )
    assert folder_res.status_code == 200
    folder_id = folder_res.json()["folder_id"]

    # 5. List folders
    list_folders_res = client.get("/devices/folders")
    assert list_folders_res.status_code == 200
    assert any(f["folder_id"] == folder_id for f in list_folders_res.json())

    # 6. Send notification (approved category)
    notif_res = client.post(
        "/devices/notifications/send",
        json={
            "source_device": "host",
            "target_device": device_id,
            "title": "Build Finished",
            "message": "All 265 tests passed.",
            "category": "system_event",
        },
    )
    assert notif_res.status_code == 200

    # 7. Send notification (blocked category -> 403)
    blocked_res = client.post(
        "/devices/notifications/send",
        json={
            "source_device": "host",
            "target_device": device_id,
            "title": "Secret",
            "message": "otp 123456",
            "category": "otp",
        },
    )
    assert blocked_res.status_code == 403

    # 8. Revoke folder
    del_folder_res = client.delete(f"/devices/folders/{folder_id}")
    assert del_folder_res.status_code == 200

    # 9. Disconnect device
    del_dev_res = client.delete(f"/devices/{device_id}")
    assert del_dev_res.status_code == 200
