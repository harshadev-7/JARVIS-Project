"""
JARVIS AI — Local Network Device Discovery Engine
=================================================
Implements zero-cloud, 100% offline LAN discovery via UDP broadcast beaconing
and direct TCP connectivity probes.
Protocol: JARVIS-DISC-v1
Default UDP Beacon Port: 54321
"""

import socket
import json
import time
import secrets
import threading
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
from core.logger import log

DISCOVERY_PROTOCOL = "JARVIS-DISC-v1"
DEFAULT_DISCOVERY_PORT = 54321


class LocalDiscoveryEngine:
    """Manages local subnet UDP beaconing and TCP reachability verification."""

    def __init__(self, port: int = DEFAULT_DISCOVERY_PORT):
        self.port = port
        self._discovered_devices: Dict[str, Dict[str, Any]] = {}
        self._listener_socket: Optional[socket.socket] = None
        self._listener_thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()

    def create_beacon_payload(
        self,
        device_id: str,
        name: str,
        device_type: str,
        service_port: int = 8000,
    ) -> Dict[str, Any]:
        """Generate a compliant discovery beacon payload."""
        return {
            "proto": DISCOVERY_PROTOCOL,
            "device_id": device_id,
            "name": name,
            "device_type": device_type,
            "port": service_port,
            "timestamp": int(time.time()),
            "nonce": secrets.token_hex(8),
        }

    def broadcast_presence(
        self,
        device_id: str,
        name: str,
        device_type: str,
        service_port: int = 8000,
        target_port: Optional[int] = None,
    ) -> bool:
        """
        Broadcast UDP discovery beacon onto the local network (255.255.255.255).
        """
        payload = self.create_beacon_payload(device_id, name, device_type, service_port)
        raw_data = json.dumps(payload).encode("utf-8")
        dest_port = target_port or self.port

        sock = None
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            sock.settimeout(1.0)
            sock.sendto(raw_data, ("<broadcast>", dest_port))
            log.info(f"Broadcasted presence beacon for '{name}' ({device_id}) on UDP port {dest_port}.")
            return True
        except Exception as e:
            log.warning(f"Failed to broadcast presence beacon: {e}")
            return False
        finally:
            if sock:
                sock.close()

    def parse_packet(self, data: bytes, addr: tuple) -> Optional[Dict[str, Any]]:
        """Parse and validate incoming discovery packet."""
        try:
            payload = json.loads(data.decode("utf-8"))
            if payload.get("proto") != DISCOVERY_PROTOCOL:
                return None
            
            ip = addr[0]
            dev_id = payload.get("device_id")
            if not dev_id:
                return None

            device_info = {
                "device_id": dev_id,
                "name": payload.get("name", "Unknown Device"),
                "device_type": payload.get("device_type", "unknown"),
                "ip": ip,
                "port": payload.get("port", 8000),
                "discovered_at": datetime.now(timezone.utc).isoformat(),
                "last_beacon_timestamp": payload.get("timestamp"),
            }

            with self._lock:
                self._discovered_devices[dev_id] = device_info

            return device_info
        except Exception:
            return None

    def start_listener(self, port: Optional[int] = None) -> bool:
        """Start background UDP listener thread."""
        if self._running:
            return True

        listen_port = port or self.port
        try:
            self._listener_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._listener_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            # Allow broadcast reception
            self._listener_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            self._listener_socket.bind(("", listen_port))
            self._listener_socket.settimeout(1.0)
            self._running = True

            self._listener_thread = threading.Thread(
                target=self._listen_loop,
                daemon=True,
                name="JARVIS-DiscoveryListener",
            )
            self._listener_thread.start()
            log.info(f"LAN Discovery listener started on UDP port {listen_port}.")
            return True
        except Exception as e:
            log.warning(f"Could not bind discovery socket on port {listen_port}: {e}")
            self.stop_listener()
            return False

    def _listen_loop(self):
        """Worker loop receiving UDP packets."""
        while self._running and self._listener_socket:
            try:
                data, addr = self._listener_socket.recvfrom(2048)
                self.parse_packet(data, addr)
            except socket.timeout:
                continue
            except Exception:
                if self._running:
                    time.sleep(0.1)
                break

    def stop_listener(self):
        """Stop background UDP listener."""
        self._running = False
        if self._listener_socket:
            try:
                self._listener_socket.close()
            except Exception:
                pass
            self._listener_socket = None

    def get_discovered_devices(self, max_age_seconds: int = 120) -> List[Dict[str, Any]]:
        """List all discovered peers seen within max_age_seconds."""
        now = time.time()
        with self._lock:
            active = []
            for dev in self._discovered_devices.values():
                ts = dev.get("last_beacon_timestamp") or 0
                if (now - ts) <= max_age_seconds:
                    active.append(dev)
            return active

    def register_simulated_peer(self, device_info: Dict[str, Any]):
        """Helper for registering a peer directly in test/mock environments."""
        with self._lock:
            self._discovered_devices[device_info["device_id"]] = device_info

    def probe_device(self, ip: str, port: int, timeout: float = 1.0) -> bool:
        """
        Direct TCP connection probe to test device reachability.
        Returns True if target responds to TCP handshake.
        """
        sock = None
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((ip, port))
            return result == 0
        except Exception:
            return False
        finally:
            if sock:
                sock.close()


discovery_engine = LocalDiscoveryEngine()
