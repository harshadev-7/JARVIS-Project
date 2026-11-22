"""
JARVIS AI — Connection Watchdog & Network Telemetry Monitor
===========================================================
Actively tracks peer health, round-trip latency (RTT), missed heartbeats,
and degradation across the multi-device network mesh.
Optimized for zero-stall asynchronous monitoring on Intel Core i3 systems.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from pydantic import BaseModel, Field

from core.logger import log


class PeerMetrics(BaseModel):
    device_id: str
    last_seen: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    rtt_ms: float = 0.0
    rolling_avg_rtt_ms: float = 0.0
    missed_heartbeats: int = 0
    status: str = "online"  # "online", "degraded", "offline"
    total_packets_received: int = 0


class ConnectionMonitor:
    """Monitors live network status, latency fluctuations, and peer reachability."""

    def __init__(self, timeout_sec: int = 60, high_latency_threshold_ms: float = 400.0):
        self.timeout_sec = timeout_sec
        self.high_latency_threshold = high_latency_threshold_ms
        self._monitors: Dict[str, PeerMetrics] = {}

    def record_heartbeat(
        self,
        device_id: str,
        rtt_ms: Optional[float] = None,
    ) -> PeerMetrics:
        """Record an incoming ping/heartbeat from a connected device."""
        now = datetime.now(timezone.utc)
        metrics = self._monitors.get(device_id)

        if not metrics:
            metrics = PeerMetrics(
                device_id=device_id,
                last_seen=now,
                rtt_ms=rtt_ms or 10.0,
                rolling_avg_rtt_ms=rtt_ms or 10.0,
                status="online",
                total_packets_received=1,
            )
            self._monitors[device_id] = metrics
        else:
            metrics.last_seen = now
            metrics.missed_heartbeats = 0
            metrics.total_packets_received += 1

            if rtt_ms is not None:
                metrics.rtt_ms = rtt_ms
                # Exponential moving average (alpha = 0.2)
                metrics.rolling_avg_rtt_ms = (0.2 * rtt_ms) + (0.8 * metrics.rolling_avg_rtt_ms)

            # Determine degradation
            if metrics.rolling_avg_rtt_ms > self.high_latency_threshold:
                metrics.status = "degraded"
            else:
                metrics.status = "online"

        return metrics

    def check_watchdog(self) -> List[str]:
        """
        Scan all monitored peers and mark offline if timeout exceeded.
        Returns list of newly disconnected device IDs.
        """
        now = datetime.now(timezone.utc)
        disconnected = []

        for dev_id, metrics in self._monitors.items():
            if metrics.status != "offline":
                elapsed = (now - metrics.last_seen).total_seconds()
                if elapsed > self.timeout_sec:
                    metrics.status = "offline"
                    metrics.missed_heartbeats += 1
                    disconnected.append(dev_id)
                    log.warning(f"Connection watchdog: Device [{dev_id}] marked OFFLINE (silent for {elapsed:.1f}s).")

        return disconnected

    def get_device_metrics(self, device_id: str) -> Optional[PeerMetrics]:
        """Fetch real-time metrics for a peer."""
        return self._monitors.get(device_id)

    def list_all_metrics(self) -> List[PeerMetrics]:
        """List telemetry across all registered device peers."""
        self.check_watchdog()
        return list(self._monitors.values())

    def reset_device(self, device_id: str):
        """Remove a device from active monitoring upon disconnect."""
        if device_id in self._monitors:
            del self._monitors[device_id]


connection_monitor = ConnectionMonitor()
