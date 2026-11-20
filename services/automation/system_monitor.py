"""
JARVIS AI — Windows System & Hardware Monitor
==============================================
Monitors CPU, RAM, Disk, Battery, Network, and Running Processes
using psutil and Windows APIs.
"""

import psutil
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class SystemMetrics(BaseModel):
    """Real-time system telemetry."""
    cpu_percent: float
    cpu_count: int
    ram_total_gb: float
    ram_used_gb: float
    ram_percent: float
    disk_total_gb: float
    disk_used_gb: float
    disk_percent: float
    battery_percent: Optional[float] = None
    battery_plugged_in: Optional[bool] = None
    network_bytes_sent_mb: float
    network_bytes_recv_mb: float
    top_processes: List[Dict[str, Any]] = Field(default_factory=list)


class SystemMonitor:
    """Monitors OS health and hardware performance."""

    def get_metrics(self, top_n_processes: int = 5) -> SystemMetrics:
        """Fetch current hardware and OS metrics."""
        cpu = psutil.cpu_percent(interval=None)
        cpu_count = psutil.cpu_count(logical=True) or 1

        ram = psutil.virtual_memory()
        disk = psutil.disk_usage("C:\\" if psutil.WINDOWS else "/")

        battery = psutil.sensors_battery()
        bat_percent = battery.percent if battery else None
        bat_plugged = battery.power_plugged if battery else None

        net = psutil.net_io_counters()

        # Extract top resource-consuming processes
        procs = []
        for p in psutil.process_iter(["pid", "name", "cpu_percent", "memory_percent"]):
            try:
                p_info = p.info
                if p_info["cpu_percent"] is not None and p_info["memory_percent"] is not None:
                    procs.append({
                        "pid": p_info["pid"],
                        "name": p_info["name"],
                        "cpu_percent": round(p_info["cpu_percent"], 1),
                        "memory_percent": round(p_info["memory_percent"], 1),
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        procs.sort(key=lambda x: (x["cpu_percent"], x["memory_percent"]), reverse=True)
        top_procs = procs[:top_n_processes]

        return SystemMetrics(
            cpu_percent=round(cpu, 1),
            cpu_count=cpu_count,
            ram_total_gb=round(ram.total / (1024 ** 3), 2),
            ram_used_gb=round(ram.used / (1024 ** 3), 2),
            ram_percent=round(ram.percent, 1),
            disk_total_gb=round(disk.total / (1024 ** 3), 2),
            disk_used_gb=round(disk.used / (1024 ** 3), 2),
            disk_percent=round(disk.percent, 1),
            battery_percent=bat_percent,
            battery_plugged_in=bat_plugged,
            network_bytes_sent_mb=round(net.bytes_sent / (1024 ** 2), 2),
            network_bytes_recv_mb=round(net.bytes_recv / (1024 ** 2), 2),
            top_processes=top_procs,
        )


system_monitor = SystemMonitor()
