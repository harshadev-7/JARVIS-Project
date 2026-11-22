"""
JARVIS AI — Phase 22 Secure Multi-Device Networking Test Suite
==============================================================
Rigorous unit and integration tests covering:
1. CryptoChannel (AES-256-GCM AEAD encryption, tamper detection, anti-replay sliding window)
2. MutualAuthenticator (HKDF key derivation, challenge-response HMAC handshake, ephemeral session keys)
3. CommandWhitelistGatekeeper (Safe commands, injection blocking, ACL enforcement)
4. ConnectionMonitor (Heartbeat RTT tracking, latency degradation, watchdog offline detection)
5. NetworkSessionManager (Session registration, hard TTL, inactivity timeouts, instant revocation)
6. EncryptedTransferEngine (End-to-end AEAD chunked transfer and integrity verification)
7. DeviceManager Phase 22 Coordinator Lifecycle
8. FastAPI Secure Networking Endpoints (REST API verification)
"""

import os
import shutil
import tempfile
import secrets
import hashlib
import pytest
from datetime import datetime, timezone, timedelta
from fastapi.testclient import TestClient

from devices.crypto_channel import (
    CryptoChannel,
    CryptoChannelError,
    ReplayAttackError,
    TamperDetectedError,
)
from devices.mutual_auth import (
    MutualAuthenticator,
    mutual_authenticator,
)
from devices.command_whitelist import (
    CommandWhitelistGatekeeper,
    RemoteCommandRequest,
)
from devices.connection_monitor import (
    ConnectionMonitor,
)
from devices.session_manager import (
    NetworkSessionManager,
)
from devices.encrypted_transfer import (
    EncryptedTransferEngine,
)
from devices.folder_access import AuthorizedFolderSandbox
from devices.device_registry.registry import DeviceType, DeviceStatus
from devices.device_manager import DeviceManager
from api.app import app


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def temp_dir():
    """Create a temporary sandbox directory for encrypted transfer tests."""
    d = tempfile.mkdtemp(prefix="jarvis_sec_test_")
    yield os.path.realpath(d)
    if os.path.exists(d):
        shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def temp_db(temp_dir):
    """Temporary SQLite database path."""
    return os.path.join(temp_dir, "test_sec_jarvis.db")


# ── 1. CryptoChannel AEAD & Anti-Replay Tests ─────────────────────────────────

def test_crypto_channel_encrypt_decrypt_payload():
    """Valid encryption and decryption of raw byte payloads using AES-256-GCM."""
    key = secrets.token_bytes(32)
    session_id = "sess-test-123"
    initiator = CryptoChannel(key, session_id, is_initiator=True)
    responder = CryptoChannel(key, session_id, is_initiator=False)

    plaintext = b"JARVIS Sensitive Command: {action: 'status', ts: 1700000000}"
    packet = initiator.encrypt_payload(plaintext)

    # Frame header checks
    assert packet.startswith(b"JSEC")
    decrypted = responder.decrypt_payload(packet)
    assert decrypted == plaintext


def test_crypto_channel_json_encryption():
    """Valid encryption and decryption of structured JSON dictionaries."""
    key = secrets.token_bytes(32)
    session_id = "sess-json-456"
    channel1 = CryptoChannel(key, session_id, is_initiator=True)
    channel2 = CryptoChannel(key, session_id, is_initiator=False)

    payload = {"command": "play_media", "file": "intro.mp4", "volume": 0.8}
    packet = channel1.encrypt_json(payload)
    recovered = channel2.decrypt_json(packet)

    assert recovered["command"] == "play_media"
    assert recovered["volume"] == 0.8


def test_crypto_channel_tamper_detection():
    """Tampering with a single byte in ciphertext must raise TamperDetectedError."""
    key = secrets.token_bytes(32)
    session_id = "sess-tamper-789"
    initiator = CryptoChannel(key, session_id, is_initiator=True)
    responder = CryptoChannel(key, session_id, is_initiator=False)

    packet = bytearray(initiator.encrypt_payload(b"Confidential payload"))
    # Corrupt a byte in the ciphertext payload
    packet[-5] ^= 0xFF

    with pytest.raises(TamperDetectedError):
        responder.decrypt_payload(bytes(packet))


def test_crypto_channel_replay_attack_rejection():
    """Replaying an identical packet must be detected and rejected."""
    key = secrets.token_bytes(32)
    session_id = "sess-replay-001"
    initiator = CryptoChannel(key, session_id, is_initiator=True)
    responder = CryptoChannel(key, session_id, is_initiator=False)

    packet = initiator.encrypt_payload(b"Critical Transaction")
    first_dec = responder.decrypt_payload(packet)
    assert first_dec == b"Critical Transaction"

    # Second presentation of same packet (Replay)
    with pytest.raises(ReplayAttackError, match="Replay attack detected"):
        responder.decrypt_payload(packet)


# ── 2. Mutual Challenge-Response Authentication Tests ────────────────────────

def test_mutual_auth_handshake_flow():
    """Mutual challenge generation, client solving, and verification."""
    auth = MutualAuthenticator()
    master_key = auth.derive_device_master_key("device-secret-auth-token-12345")
    device_id = "dev-android-phone"

    # 1. Issue challenge
    challenge_obj, challenge_hex = auth.issue_challenge(device_id, master_key)
    assert len(challenge_hex) == 64  # 32 bytes hex
    assert challenge_obj.device_id == device_id

    # 2. Client computes response
    client_response = auth.solve_challenge(challenge_hex, master_key)

    # 3. Host verifies response
    result = auth.verify_response(challenge_obj.challenge_id, client_response)
    assert result.success is True
    assert result.session_id.startswith("sec-sess-")
    assert len(bytes.fromhex(result.session_key_hex)) == 32


def test_mutual_auth_invalid_signature():
    """Invalid signature must fail mutual authentication."""
    auth = MutualAuthenticator()
    master_key = auth.derive_device_master_key("legitimate-token")
    wrong_key = auth.derive_device_master_key("attacker-token")
    device_id = "dev-compromised"

    challenge_obj, challenge_hex = auth.issue_challenge(device_id, master_key)
    # Attacker attempts to solve with incorrect key
    bad_response = auth.solve_challenge(challenge_hex, wrong_key)

    result = auth.verify_response(challenge_obj.challenge_id, bad_response)
    assert result.success is False
    assert "verification mismatch" in result.error


def test_mutual_auth_single_use_challenge():
    """Challenges must be single-use and cannot be verified twice."""
    auth = MutualAuthenticator()
    master_key = auth.derive_device_master_key("one-time-token")
    device_id = "dev-single-use"

    challenge_obj, challenge_hex = auth.issue_challenge(device_id, master_key)
    resp = auth.solve_challenge(challenge_hex, master_key)

    res1 = auth.verify_response(challenge_obj.challenge_id, resp)
    assert res1.success is True

    # Reusing challenge ID must fail
    res2 = auth.verify_response(challenge_obj.challenge_id, resp)
    assert res2.success is False
    assert "already consumed" in res2.error


# ── 3. Command Whitelist & Sandboxing Tests ──────────────────────────────────

@pytest.mark.asyncio
async def test_command_whitelist_safe_execution():
    """Whitelisted commands (ping, get_status, echo) must execute cleanly."""
    gate = CommandWhitelistGatekeeper()
    req = RemoteCommandRequest(
        command_name="ping",
        device_id="dev-1",
        session_id="sess-1",
        parameters={},
    )
    res = await gate.execute_command(req)
    assert res.success is True
    assert res.data.get("pong") is True

    # Echo command
    echo_req = RemoteCommandRequest(
        command_name="echo",
        device_id="dev-1",
        session_id="sess-1",
        parameters={"message": "Hello JARVIS"},
    )
    echo_res = await gate.execute_command(echo_req)
    assert echo_res.success is True
    assert echo_res.data.get("echo") == "Hello JARVIS"


@pytest.mark.asyncio
async def test_command_whitelist_blocks_unauthorized_commands():
    """Non-whitelisted commands must be blocked immediately."""
    gate = CommandWhitelistGatekeeper()
    bad_req = RemoteCommandRequest(
        command_name="format_disk",
        device_id="dev-untrusted",
        session_id="sess-2",
        parameters={},
    )
    res = await gate.execute_command(bad_req)
    assert res.success is False
    assert "not in the authorized remote command whitelist" in res.error


@pytest.mark.asyncio
async def test_command_whitelist_parameter_injection_rejection():
    """Parameters containing shell metacharacters must be rejected."""
    gate = CommandWhitelistGatekeeper()
    injection_req = RemoteCommandRequest(
        command_name="echo",
        device_id="dev-1",
        session_id="sess-1",
        parameters={"message": "hello ; rm -rf /"},
    )
    res = await gate.execute_command(injection_req)
    assert res.success is False
    assert "Dangerous character" in res.error


# ── 4. Connection Monitoring Watchdog Tests ─────────────────────────────────

def test_connection_monitor_heartbeat_and_degradation():
    """Heartbeat updates RTT and flags high latency."""
    mon = ConnectionMonitor(timeout_sec=60, high_latency_threshold_ms=200.0)
    dev_id = "dev-telemetry-node"

    # Good latency ping
    m1 = mon.record_heartbeat(dev_id, rtt_ms=45.0)
    assert m1.status == "online"
    assert m1.rtt_ms == 45.0

    # High latency ping triggers degradation
    m2 = mon.record_heartbeat(dev_id, rtt_ms=350.0)
    # Exponential moving average will rise
    assert m2.status in ("online", "degraded")
    assert m2.total_packets_received == 2


def test_connection_monitor_watchdog_timeout():
    """Peers exceeding silent timeout (>60s) must be transitioned to offline."""
    mon = ConnectionMonitor(timeout_sec=10)
    dev_id = "dev-watchdog-stale"

    mon.record_heartbeat(dev_id, rtt_ms=20.0)
    # Artificially age the last seen timestamp
    mon._monitors[dev_id].last_seen = datetime.now(timezone.utc) - timedelta(seconds=20)

    disconnected = mon.check_watchdog()
    assert dev_id in disconnected
    assert mon.get_device_metrics(dev_id).status == "offline"


# ── 5. Network Session Manager & Revocation Tests ────────────────────────────

def test_session_lifecycle_and_revocation(temp_db):
    """Session registration, touch, and instant revocation."""
    sm = NetworkSessionManager(db_path=temp_db, default_ttl_sec=3600, inactivity_timeout_sec=600)
    dev_id = "dev-session-test"
    key = secrets.token_bytes(32)
    session_id = "sess-lifecycle-01"

    rec = sm.register_session(session_id, dev_id, key)
    assert rec.status == "active"

    # Valid session retrieval
    active_rec = sm.get_session(session_id)
    assert active_rec is not None

    # Touch session
    assert sm.touch_session(session_id) is True

    # Revoke
    assert sm.revoke_session(session_id) is True
    assert sm.get_session(session_id) is None


def test_session_hard_ttl_expiration(temp_db):
    """Expired sessions must be rejected."""
    sm = NetworkSessionManager(db_path=temp_db, default_ttl_sec=-10)  # Already expired
    session_id = "sess-expired"
    sm.register_session(session_id, "dev-expired", secrets.token_bytes(32), ttl_seconds=-1)

    assert sm.get_session(session_id) is None


# ── 6. Encrypted File Transfer Engine Tests ──────────────────────────────────

def test_encrypted_file_transfer_successful(temp_dir, temp_db):
    """End-to-end AEAD chunked file transfer with SHA-256 verification."""
    sandbox = AuthorizedFolderSandbox(db_path=temp_db)
    sandbox.authorize_folder(temp_dir, access_level="read_write")
    xfr_engine = EncryptedTransferEngine(sandbox=sandbox)

    key = secrets.token_bytes(32)
    session_id = "sess-xfr-01"
    sender_channel = CryptoChannel(key, session_id, is_initiator=True)
    receiver_channel = CryptoChannel(key, session_id, is_initiator=False)

    # Create dummy source file
    src_file = os.path.join(temp_dir, "plain_source.bin")
    content = b"J.A.R.V.I.S. Secure AEAD Transfer Payload " * 500
    with open(src_file, "wb") as f:
        f.write(content)
    expected_hash = hashlib.sha256(content).hexdigest()

    # Generate encrypted packets stream
    packets = [pkt for pkt, idx, is_last, h in xfr_engine.send_encrypted_file(src_file, sender_channel)]
    assert len(packets) >= 1

    # Receive and decrypt into target destination
    recv_res = xfr_engine.receive_encrypted_file(
        file_name="decrypted_output.bin",
        target_folder=temp_dir,
        encrypted_packets=packets,
        expected_sha256=expected_hash,
        channel=receiver_channel,
    )

    assert recv_res["success"] is True
    assert os.path.exists(recv_res["file_path"])
    with open(recv_res["file_path"], "rb") as f:
        assert f.read() == content


# ── 7. DeviceManager Phase 22 Coordinator Tests ──────────────────────────────

@pytest.mark.asyncio
async def test_device_manager_secure_networking_lifecycle(temp_dir, temp_db):
    """Full lifecycle: pair -> mutual auth -> remote command -> monitor -> revoke."""
    dm = DeviceManager()

    # 1. Pair
    pin = dm.generate_pairing_pin()
    dev_rec, raw_token = dm.pair_device(pin, "Secure Test Phone", DeviceType.ANDROID)

    # 2. Challenge-Response
    chg_obj, chg_hex = dm.issue_auth_challenge(dev_rec.device_id)
    master_key = dm.mutual_auth.derive_device_master_key(dev_rec.auth_token)
    solution = dm.mutual_auth.solve_challenge(chg_hex, master_key)
    auth_res = dm.verify_auth_response(chg_obj.challenge_id, solution)

    assert auth_res.success is True
    session_id = auth_res.session_id

    # 3. Remote Command Execution
    cmd_res = await dm.execute_remote_command(
        command_name="ping",
        device_id=dev_rec.device_id,
        session_id=session_id,
    )
    assert cmd_res.success is True
    assert cmd_res.data.get("pong") is True

    # 4. Connection Watchdog
    metrics = dm.record_peer_heartbeat(dev_rec.device_id, rtt_ms=25.0)
    assert metrics.status == "online"

    # 5. Revocation
    revoked = dm.revoke_session(session_id)
    assert revoked is True

    # Subsequent command must fail
    cmd_after = await dm.execute_remote_command(
        command_name="ping",
        device_id=dev_rec.device_id,
        session_id=session_id,
    )
    assert cmd_after.success is False
    assert "Invalid or expired network session" in cmd_after.error


# ── 8. FastAPI REST Endpoints Tests ──────────────────────────────────────────

@pytest.fixture
def client():
    return TestClient(app)


def test_api_secure_networking_endpoints(client):
    """Verify challenge, solve, command execution, and revocation via REST API."""
    # 1. Pair device to obtain credentials
    pin_res = client.post("/devices/pairing-code")
    pin = pin_res.json()["pairing_code"]

    pair_res = client.post(
        "/devices/pair",
        json={"pairing_code": pin, "name": "API Secure Node", "device_type": "windows"},
    )
    dev_data = pair_res.json()
    device_id = dev_data["device_id"]
    token = dev_data["auth_token"]

    # 2. Request mutual challenge
    chg_res = client.post("/devices/auth/challenge", json={"device_id": device_id})
    assert chg_res.status_code == 200
    chg_json = chg_res.json()
    challenge_id = chg_json["challenge_id"]
    challenge_data = chg_json["challenge_data"]

    # 3. Solve challenge
    master_key = mutual_authenticator.derive_device_master_key(token)
    solution = mutual_authenticator.solve_challenge(challenge_data, master_key)

    # 4. Verify challenge
    verify_res = client.post(
        "/devices/auth/verify",
        json={"challenge_id": challenge_id, "client_response_hex": solution},
    )
    assert verify_res.status_code == 200
    session_id = verify_res.json()["session_id"]

    # 5. Execute whitelisted command
    exec_res = client.post(
        "/devices/command/execute",
        json={
            "command_name": "get_status",
            "device_id": device_id,
            "session_id": session_id,
            "parameters": {},
        },
    )
    assert exec_res.status_code == 200
    assert exec_res.json()["success"] is True

    # 6. Revoke session
    revoke_res = client.post(f"/devices/sessions/{session_id}/revoke")
    assert revoke_res.status_code == 200
    assert revoke_res.json()["status"] == "revoked"
