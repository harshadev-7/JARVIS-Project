"""JARVIS Phase 28 — Subsystem Registry & Health Probing Core
===========================================================
Catalogs and monitors the operational health, phase origins, and readiness
of all 28 specialized functional subsystems across the J.A.R.V.I.S. architecture.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class SubsystemStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    OFFLINE = "offline"
    INITIALIZING = "initializing"


@dataclass
class SubsystemInfo:
    """Descriptor and health status for a single J.A.R.V.I.S. subsystem."""
    subsystem_id: str
    name: str
    phase: int
    category: str  # intelligence, reasoning, memory, operational, security, media, orchestration, runtime
    status: SubsystemStatus = SubsystemStatus.INITIALIZING
    description: str = ""
    health_check_fn: Optional[Callable[[], bool]] = None
    last_checked: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subsystem_id": self.subsystem_id,
            "name": self.name,
            "phase": self.phase,
            "category": self.category,
            "status": self.status.value,
            "description": self.description,
            "last_checked": round(self.last_checked, 2) if self.last_checked else 0.0,
            "details": self.details,
        }


class SubsystemRegistry:
    """Central registry and health monitor for all 28 JARVIS subsystems."""

    def __init__(self):
        self.subsystems: Dict[str, SubsystemInfo] = {}
        self._register_canonical_subsystems()

    def _register_canonical_subsystems(self) -> None:
        """Initializes descriptors for all 28 project subsystems."""
        canonical_entries = [
            # Intelligence & Neural Core
            ("neural_transformer_core", "Custom Transformer Core", 1, "intelligence", "Decoder-only micro-transformer architecture with custom tokenizer."),
            ("deductive_solver", "Deductive Problem Solver", 5, "reasoning", "10-step first-principles deductive reasoning and problem solving engine."),
            ("code_analysis_engine", "Code Analysis & Linting", 6, "reasoning", "AST-based syntax validation and code structure inspection."),
            ("deterministic_math", "Deterministic Math Engine", 7, "reasoning", "Symbolic math evaluator with zero hallucinations."),
            ("logical_reasoning", "Logical Reasoning Engine", 8, "reasoning", "Constraint satisfaction and riddle solving heuristics."),
            
            # Swarm & Agent Core
            ("agent_swarm_registry", "15-Agent Swarm Registry", 9, "orchestration", "Central catalog of all 15 domain-specialized autonomous agents."),
            ("persistent_storage", "Persistent SQLite Storage", 10, "memory", "Relational database maintaining facts, learning records, and logs."),
            ("semantic_vector_store", "Semantic Vector Memory", 11, "memory", "TF-IDF vector store supporting contextual similarity lookup."),
            ("knowledge_answering", "Direct Knowledge Engine", 12, "intelligence", "Direct question answering without extraneous options."),
            ("research_pipeline", "Authentic Research Pipeline", 13, "intelligence", "Multi-source web search, scraping, and verification engine."),
            ("rag_retrieval_engine", "RAG Knowledge Retrieval", 14, "intelligence", "Retrieval-augmented offline generation engine."),
            
            # Operational & System Layer
            ("audio_speech_engine", "Audio & TTS Speech Engine", 15, "operational", "Neural & offline local speech synthesis."),
            ("desktop_automation", "Desktop Automation Engine", 16, "operational", "Safe application launching and window automation."),
            ("intent_classifier", "Intent Classifier & Router", 17, "intelligence", "High-speed deterministic intent classifier and task router."),
            ("personal_profile_memory", "Personal Profile Memory", 18, "memory", "Episodic user fact extraction, recall, and preferences."),
            
            # Media & Visual
            ("video_editing_suite", "Video Timeline & Reel Creator", 19, "media", "Non-destructive timeline assembly and viral reel creation."),
            ("autoclip_engine", "AutoClip Viral Highlight Extractor", 19, "media", "Long-to-short video slicing, subtitle snapping, and shorts compilation."),
            ("computer_vision", "Computer Vision & OCR", 20, "operational", "Text extraction from images and visual classification."),
            
            # Networking & Devices
            ("authorized_device_mesh", "Authorized Device Mesh", 21, "operational", "LAN UDP discovery and companion device registration."),
            ("encrypted_device_sync", "Encrypted Multi-Device Sync", 22, "security", "Cryptographic PIN pairing, mutual tokens, and safe file sync."),
            
            # Security & Research
            ("cybersecurity_intelligence", "Cybersecurity Threat Modeling", 23, "security", "Defensive intelligence, threat modeling, and CWE/CVE repository."),
            ("ethical_security_sandbox", "Ethical Security Sandbox", 24, "security", "Isolated static file quarantine and configuration auditing."),
            ("advanced_neural_research", "Advanced Neural Research", 25, "intelligence", "GQA, RoPE, KV-Cache, RMSNorm, and INT8 dynamic quantization."),
            ("adaptive_learning_engine", "Adaptive Learning Engine", 26, "intelligence", "User feedback analysis, learning audit log, and fine-tuning staging."),
            
            # Orchestration & Integration
            ("dag_task_engine", "DAG Task Dependency Engine", 27, "orchestration", "Directed Acyclic Graph scheduler with cycle detection."),
            ("circuit_breaker_resilience", "Circuit Breaker Fault Tolerance", 27, "orchestration", "Resilience state machine with retries and fallback diversion."),
            ("consensus_verification", "Multi-Agent Consensus Engine", 27, "orchestration", "Majority vote, unanimous, and weighted cross-agent verification."),
            ("master_system_runtime", "Master Runtime & Diagnostics", 28, "runtime", "Unified boot lifecycle, event bus, and automated self-tests."),
            ("self_learning_engine", "Self-Learning Engine", 29, "evolution", "Learns from approved feedback, logs mistakes, updates local knowledge, and evaluates learning progress."),
            ("self_reasoning_engine", "Self-Reasoning Improvement Engine", 30, "evolution", "Evaluates reasoning strategies, maintains benchmarks, optimizes task decomposition, and selects strategies based on accuracy."),
            ("sentinel_proactive_daemon", "Autonomous Sentinel Daemon", 29, "runtime", "Continuous background resource guarding, proactive advisories, and self-healing."),
            ("voice_conversational_assistant", "Conversational Voice Assistant", 30, "operational", "Full-duplex hands-free voice loop with wake-word detection, local STT, and neural speech."),
        ]

        for sub_id, name, phase, category, desc in canonical_entries:
            self.subsystems[sub_id] = SubsystemInfo(
                subsystem_id=sub_id,
                name=name,
                phase=phase,
                category=category,
                description=desc,
                status=SubsystemStatus.HEALTHY,
                last_checked=time.time(),
            )

    def register_health_check(self, subsystem_id: str, check_fn: Callable[[], bool]) -> None:
        """Associates a callable health check function with a subsystem."""
        if subsystem_id in self.subsystems:
            self.subsystems[subsystem_id].health_check_fn = check_fn

    def get_subsystem(self, subsystem_id: str) -> Optional[SubsystemInfo]:
        """Fetches metadata for a single subsystem."""
        return self.subsystems.get(subsystem_id)

    def list_subsystems(self, category: Optional[str] = None) -> List[SubsystemInfo]:
        """Lists all registered subsystems, optionally filtered by category."""
        if category:
            return [s for s in self.subsystems.values() if s.category == category]
        return list(self.subsystems.values())

    def probe_all_health(self) -> Dict[str, Any]:
        """Runs health checks across all registered subsystems and compiles overall health metrics."""
        now = time.time()
        healthy_count = 0
        degraded_count = 0
        offline_count = 0

        for sub in self.subsystems.values():
            sub.last_checked = now
            if sub.health_check_fn:
                try:
                    is_ok = sub.health_check_fn()
                    sub.status = SubsystemStatus.HEALTHY if is_ok else SubsystemStatus.DEGRADED
                except Exception as exc:
                    logger.warning("Health check failed for [%s]: %s", sub.subsystem_id, exc)
                    sub.status = SubsystemStatus.DEGRADED
                    sub.details["last_error"] = str(exc)

            if sub.status == SubsystemStatus.HEALTHY:
                healthy_count += 1
            elif sub.status == SubsystemStatus.DEGRADED:
                degraded_count += 1
            else:
                offline_count += 1

        total = len(self.subsystems)
        overall_status = (
            SubsystemStatus.HEALTHY.value
            if offline_count == 0 and degraded_count == 0
            else (SubsystemStatus.DEGRADED.value if offline_count == 0 else SubsystemStatus.OFFLINE.value)
        )

        return {
            "overall_status": overall_status,
            "total_subsystems": total,
            "healthy_count": healthy_count,
            "degraded_count": degraded_count,
            "offline_count": offline_count,
            "health_score_pct": round((healthy_count / max(1, total)) * 100, 1),
            "probed_at": now,
        }


# Global singleton registry
subsystem_registry = SubsystemRegistry()
