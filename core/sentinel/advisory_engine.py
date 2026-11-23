"""JARVIS Phase 29 — Proactive Advisory Engine
=============================================
Evaluates telemetry, memory status, learning records, and security postures
to generate contextual, unprompted recommendations and early-warning alerts.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
import psutil

logger = logging.getLogger(__name__)


class AdvisorySeverity(str, Enum):
    INFO = "info"
    RECOMMENDATION = "recommendation"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class ProactiveAdvisory:
    """A proactive suggestion, warning, or operational insight."""
    id: str
    timestamp: float
    title: str
    description: str
    severity: AdvisorySeverity
    category: str  # hardware, learning, security, orchestration, media
    recommended_command: Optional[str] = None
    action_payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": round(self.timestamp, 3),
            "title": self.title,
            "description": self.description,
            "severity": self.severity.value,
            "category": self.category,
            "recommended_command": self.recommended_command,
            "action_payload": self.action_payload,
        }


class ProactiveAdvisoryEngine:
    """Evaluates contextual heuristics to anticipate user and system needs."""

    def __init__(self, ram_threshold_pct: float = 85.0, cpu_threshold_pct: float = 90.0):
        self.ram_threshold_pct = ram_threshold_pct
        self.cpu_threshold_pct = cpu_threshold_pct

    async def evaluate_heuristics(self) -> List[ProactiveAdvisory]:
        """Runs all rule checks and generates a list of active proactive advisories."""
        advisories: List[ProactiveAdvisory] = []
        now = time.time()

        # 1. Hardware Telemetry Heuristics
        try:
            mem = psutil.virtual_memory()
            cpu_pct = psutil.cpu_percent(interval=None)

            if mem.percent >= self.ram_threshold_pct:
                advisories.append(ProactiveAdvisory(
                    id=f"adv-ram-{uuid.uuid4().hex[:6]}",
                    timestamp=now,
                    title="High RAM Utilization Detected",
                    description=f"System RAM is at {mem.percent:.1f}% ({mem.used / (1024**3):.1f}GB / {mem.total / (1024**3):.1f}GB). Consider running maintenance or closing heavy tasks.",
                    severity=AdvisorySeverity.WARNING,
                    category="hardware",
                    recommended_command="/metrics",
                    action_payload={"ram_pct": mem.percent},
                ))

            if cpu_pct >= self.cpu_threshold_pct:
                advisories.append(ProactiveAdvisory(
                    id=f"adv-cpu-{uuid.uuid4().hex[:6]}",
                    timestamp=now,
                    title="Elevated CPU Load Detected",
                    description=f"CPU utilization spiked to {cpu_pct:.1f}%. Background tasks may experience throttling.",
                    severity=AdvisorySeverity.WARNING,
                    category="hardware",
                    recommended_command="/status",
                    action_payload={"cpu_pct": cpu_pct},
                ))
        except Exception as exc:
            logger.debug("Failed evaluating hardware telemetry: %s", exc)

        # 2. Adaptive Learning Feedback Heuristic
        try:
            from core.learning.adaptive_engine import adaptive_learning_engine
            staged_examples = getattr(adaptive_learning_engine, "staged_dataset", [])
            if len(staged_examples) >= 3:
                advisories.append(ProactiveAdvisory(
                    id=f"adv-learn-{uuid.uuid4().hex[:6]}",
                    timestamp=now,
                    title="Uncommitted Feedback Ready for Fine-Tuning",
                    description=f"You have {len(staged_examples)} staged learning corrections ready. Train candidate model on CPU to prevent repeating mistakes.",
                    severity=AdvisorySeverity.RECOMMENDATION,
                    category="learning",
                    recommended_command="/finetune-staged",
                    action_payload={"staged_count": len(staged_examples)},
                ))
        except Exception as exc:
            logger.debug("Failed evaluating adaptive learning state: %s", exc)

        # 3. Circuit Breaker Fault Heuristic
        try:
            from core.orchestrator.circuit_breaker import circuit_breaker_registry
            tripped = [
                name for name, breaker in circuit_breaker_registry.breakers.items()
                if breaker.state.value == "open"
            ]
            if tripped:
                advisories.append(ProactiveAdvisory(
                    id=f"adv-cb-{uuid.uuid4().hex[:6]}",
                    timestamp=now,
                    title="Circuit Breakers Tripped",
                    description=f"Circuit breakers currently OPEN for agents: {', '.join(tripped)}. Diverting calls to safe fallbacks.",
                    severity=AdvisorySeverity.WARNING,
                    category="orchestration",
                    recommended_command="/orchestrate status",
                    action_payload={"tripped_breakers": tripped},
                ))
        except Exception as exc:
            logger.debug("Failed evaluating circuit breakers: %s", exc)

        # 4. Subsystem Health Probing Heuristic
        try:
            from core.integration.subsystem_registry import subsystem_registry, SubsystemStatus
            degraded = [
                s.name for s in subsystem_registry.subsystems.values()
                if s.status == SubsystemStatus.DEGRADED or s.status == SubsystemStatus.OFFLINE
            ]
            if degraded:
                advisories.append(ProactiveAdvisory(
                    id=f"adv-health-{uuid.uuid4().hex[:6]}",
                    timestamp=now,
                    title="Subsystems Requiring Attention",
                    description=f"The following subsystems reported degraded state: {', '.join(degraded)}.",
                    severity=AdvisorySeverity.WARNING,
                    category="system",
                    recommended_command="/diagnostics",
                    action_payload={"degraded_subsystems": degraded},
                ))
        except Exception as exc:
            logger.debug("Failed evaluating subsystem health: %s", exc)

        return advisories


proactive_advisory_engine = ProactiveAdvisoryEngine()
