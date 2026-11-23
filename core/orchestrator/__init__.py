"""JARVIS Multi-Agent Orchestrator Module (Phase 27)
===================================================
Advanced Autonomous Agent Orchestration:
- Directed Acyclic Graph (DAG) task scheduling with topological staging and cycle detection
- Circuit Breaker state machine (CLOSED -> OPEN -> HALF_OPEN) with fallback routing
- Exponential backoff retry policies with jitter
- Multi-agent consensus voting (Majority Vote, Unanimous, Weighted Confidence)
- Supervisory goal decomposition and mission synthesis
- Hierarchical multi-agent coordinator
"""

from core.orchestrator.base_agent import (
    BaseAgent,
    AgentMessage,
    AgentTaskResult,
)
from core.orchestrator.agent_registry import (
    AgentRegistry,
    agent_registry,
)
from core.orchestrator.orchestrator import (
    OrchestrationMission,
    MultiAgentOrchestrator,
    orchestrator,
)
from core.orchestrator.task_graph import (
    TaskNode,
    TaskGraph,
    TaskStatus,
    RetryPolicy,
    CyclicDependencyError,
)
from core.orchestrator.circuit_breaker import (
    CircuitState,
    CircuitBreaker,
    CircuitBreakerRegistry,
    CircuitBreakerOpenError,
    circuit_breaker_registry,
)
from core.orchestrator.consensus_engine import (
    ConsensusStrategy,
    VerificationVote,
    ConsensusOutcome,
    ConsensusEngine,
    consensus_engine,
)
from core.orchestrator.supervisor_agent import (
    SupervisorAgent,
    supervisor_agent,
)
from core.orchestrator.hierarchical_orchestrator import (
    HierarchicalOrchestrator,
    hierarchical_orchestrator,
)

__all__ = [
    "BaseAgent",
    "AgentMessage",
    "AgentTaskResult",
    "AgentRegistry",
    "agent_registry",
    "OrchestrationMission",
    "MultiAgentOrchestrator",
    "orchestrator",
    "TaskNode",
    "TaskGraph",
    "TaskStatus",
    "RetryPolicy",
    "CyclicDependencyError",
    "CircuitState",
    "CircuitBreaker",
    "CircuitBreakerRegistry",
    "CircuitBreakerOpenError",
    "circuit_breaker_registry",
    "ConsensusStrategy",
    "VerificationVote",
    "ConsensusOutcome",
    "ConsensusEngine",
    "consensus_engine",
    "SupervisorAgent",
    "supervisor_agent",
    "HierarchicalOrchestrator",
    "hierarchical_orchestrator",
]
