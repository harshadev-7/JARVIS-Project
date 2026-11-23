"""JARVIS Phase 27 — Advanced Autonomous Agent Orchestration Unit Tests
=======================================================================
Validates:
1. Directed Acyclic Graph (DAG) construction, validation, and topological staging.
2. Cyclic dependency detection and self-dependency rejection.
3. Circuit breaker state transitions (CLOSED -> OPEN -> HALF_OPEN -> CLOSED).
4. Exponential retry backoff and fallback agent diversion.
5. Consensus voting strategies (Majority, Unanimous, Weighted Confidence).
6. SupervisorAgent objective decomposition and mission synthesis.
7. End-to-end hierarchical orchestration with context passing across stages.
8. Brain router /orchestrate intent classification.
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from core.brain.router import IntentRouter, IntentType
from core.orchestrator import (
    CircuitBreaker,
    CircuitBreakerOpenError,
    CircuitBreakerRegistry,
    CircuitState,
    ConsensusEngine,
    ConsensusOutcome,
    ConsensusStrategy,
    CyclicDependencyError,
    HierarchicalOrchestrator,
    OrchestrationMission,
    RetryPolicy,
    SupervisorAgent,
    TaskGraph,
    TaskNode,
    TaskStatus,
    VerificationVote,
    agent_registry,
    circuit_breaker_registry,
    consensus_engine,
    hierarchical_orchestrator,
    supervisor_agent,
)
from core.orchestrator.base_agent import AgentTaskResult


# 1. DAG Engine Tests
def test_dag_validation_and_staging():
    """Verify DAG topological staging: A -> (B, C) -> D."""
    graph = TaskGraph(mission_id="test-mis", objective="Build and deploy API")

    node_a = graph.add_node(task_id="A", assigned_agent="research", prompt="Research specs")
    node_b = graph.add_node(task_id="B", assigned_agent="programming", prompt="Code backend", dependencies=["A"])
    node_c = graph.add_node(task_id="C", assigned_agent="cybersecurity", prompt="Security audit", dependencies=["A"])
    node_d = graph.add_node(task_id="D", assigned_agent="task_planning", prompt="Deploy", dependencies=["B", "C"])

    # Validate DAG passes without error
    graph.validate_dag()

    # Stages should be: Stage 1: [A], Stage 2: [B, C], Stage 3: [D]
    stages = graph.get_execution_stages()
    assert len(stages) == 3
    assert [n.task_id for n in stages[0]] == ["A"]
    stage_2_ids = {n.task_id for n in stages[1]}
    assert stage_2_ids == {"B", "C"}
    assert [n.task_id for n in stages[2]] == ["D"]


def test_dag_cycle_detection():
    """Verify circular dependency detection raises CyclicDependencyError."""
    graph = TaskGraph(mission_id="test-cycle", objective="Loop test")
    graph.add_node(task_id="A", assigned_agent="research", prompt="Task A", dependencies=["C"])
    graph.add_node(task_id="B", assigned_agent="programming", prompt="Task B", dependencies=["A"])
    graph.add_node(task_id="C", assigned_agent="task_planning", prompt="Task C", dependencies=["B"])

    with pytest.raises(CyclicDependencyError):
        graph.validate_dag()


def test_dag_self_dependency():
    """Verify self-dependency raises CyclicDependencyError."""
    graph = TaskGraph(mission_id="test-self", objective="Self loop")
    graph.add_node(task_id="A", assigned_agent="research", prompt="Task A", dependencies=["A"])

    with pytest.raises(CyclicDependencyError):
        graph.validate_dag()


def test_dag_missing_dependency():
    """Verify referencing non-existent dependency raises ValueError."""
    graph = TaskGraph(mission_id="test-missing", objective="Missing dep")
    graph.add_node(task_id="A", assigned_agent="research", prompt="Task A", dependencies=["NON_EXISTENT"])

    with pytest.raises(ValueError, match="depends on non-existent task"):
        graph.validate_dag()


def test_dag_context_injection():
    """Verify predecessor task outputs are properly injected into successor context."""
    graph = TaskGraph(mission_id="test-ctx", objective="Context passing")
    node_a = graph.add_node(task_id="A", assigned_agent="research", prompt="Research data")
    node_b = graph.add_node(
        task_id="B",
        assigned_agent="programming",
        prompt="Use research",
        dependencies=["A"],
        context_keys={"A": "research_output"},
    )

    # Simulate completion of task A
    node_a.status = TaskStatus.COMPLETED
    node_a.result_data = {"key_finding": "Use SQLite"}
    node_a.result_summary = "Research completed successfully."

    ctx = graph.build_task_context("B", global_context={"user_id": "test_user"})
    assert ctx["user_id"] == "test_user"
    assert ctx["research_output"] == {"key_finding": "Use SQLite"}
    assert "A" in ctx["predecessor_results"]
    assert ctx["predecessor_results"]["A"]["agent"] == "research"


# 2. Circuit Breaker Tests
def test_circuit_breaker_state_transitions():
    """Verify CLOSED -> OPEN -> HALF_OPEN -> CLOSED transitions."""
    breaker = CircuitBreaker(agent_key="test_agent", failure_threshold=3, recovery_timeout_sec=0.1)

    assert breaker.state == CircuitState.CLOSED
    assert breaker.can_execute() is True

    # Record 2 failures -> remains CLOSED
    breaker.record_failure("error 1")
    breaker.record_failure("error 2")
    assert breaker.state == CircuitState.CLOSED
    assert breaker.consecutive_failures == 2

    # 3rd failure -> trips to OPEN
    breaker.record_failure("error 3")
    assert breaker.state == CircuitState.OPEN
    assert breaker.total_trips == 1
    assert breaker.can_execute() is False

    # Wait for recovery timeout
    time.sleep(0.12)

    # Probing should transition to HALF_OPEN
    assert breaker.can_execute() is True
    assert breaker.state == CircuitState.HALF_OPEN

    # Success in HALF_OPEN resets to CLOSED
    breaker.record_success()
    assert breaker.state == CircuitState.CLOSED
    assert breaker.consecutive_failures == 0


@pytest.mark.asyncio
async def test_circuit_breaker_resilient_retry_and_fallback():
    """Verify exponential retries and fallback agent diversion."""
    registry = CircuitBreakerRegistry()
    breaker = registry.get_breaker("flaky_agent")
    breaker.fallback_agent_key = "backup_agent"

    call_count = 0

    async def flaky_call(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise RuntimeError("Transient network glitch")
        return AgentTaskResult(agent_name="flaky_agent", task_id="t1", success=True, summary="Recovered on retry")

    res = await registry.execute_with_resilience(
        agent_key="flaky_agent",
        execute_fn=flaky_call,
        max_retries=2,
        base_delay_sec=0.01,
    )
    assert res.success is True
    assert call_count == 2
    assert breaker.state == CircuitState.CLOSED


@pytest.mark.asyncio
async def test_circuit_breaker_fallback_diversion():
    """Verify diversion to fallback executor when circuit is OPEN."""
    registry = CircuitBreakerRegistry()
    breaker = registry.get_breaker("failing_agent")
    breaker.fallback_agent_key = "safe_backup"
    breaker.state = CircuitState.OPEN
    breaker.last_failure_time = time.time()  # Not yet expired

    async def fallback_exec(agent_key, *args, **kwargs):
        return AgentTaskResult(agent_name=agent_key, task_id="fb-1", success=True, summary="Executed via fallback agent")

    res = await registry.execute_with_resilience(
        agent_key="failing_agent",
        execute_fn=AsyncMock(),
        fallback_executor=fallback_exec,
    )
    assert res.success is True
    assert res.agent_name == "safe_backup"
    assert "fallback agent" in res.summary


# 3. Consensus Engine Tests
@pytest.mark.asyncio
async def test_consensus_majority_vote():
    """Verify majority vote strategy."""
    engine = ConsensusEngine()

    async def mock_agent_executor(key: str, prompt: str, context=None):
        if key in {"sec_agent_1", "sec_agent_2"}:
            return AgentTaskResult(agent_name=key, task_id="v1", success=True, summary="I approve this security fix.")
        return AgentTaskResult(agent_name=key, task_id="v2", success=False, summary="I reject this proposal due to a flaw.")

    outcome = await engine.evaluate_proposal(
        proposal="Apply parameterized queries to sanitize input",
        verifying_agents=["sec_agent_1", "sec_agent_2", "sec_agent_3"],
        agent_executor=mock_agent_executor,
        strategy=ConsensusStrategy.MAJORITY_VOTE,
    )

    assert outcome.agreed is True
    assert outcome.consensus_score > 0.6
    assert len(outcome.votes) == 3
    assert len(outcome.dissenting_opinions) == 1


@pytest.mark.asyncio
async def test_consensus_unanimous_rejection():
    """Verify unanimous strategy rejects if even a single agent dissents."""
    engine = ConsensusEngine()

    async def mock_agent_executor(key: str, prompt: str, context=None):
        if key == "agent_dissent":
            return AgentTaskResult(agent_name=key, task_id="v1", success=False, summary="Critical security flaw detected, reject.")
        return AgentTaskResult(agent_name=key, task_id="v2", success=True, summary="Approve.")

    outcome = await engine.evaluate_proposal(
        proposal="Disable certificate checking in production",
        verifying_agents=["agent_1", "agent_2", "agent_dissent"],
        agent_executor=mock_agent_executor,
        strategy=ConsensusStrategy.UNANIMOUS,
    )

    assert outcome.agreed is False
    assert len(outcome.dissenting_opinions) == 1
    assert "Critical security flaw" in outcome.dissenting_opinions[0]


# 4. Supervisor & Hierarchical Orchestrator Tests
def test_supervisor_decomposition():
    """Verify supervisor decomposes security objective into proper DAG."""
    supervisor = SupervisorAgent()
    graph = supervisor.decompose_objective("Audit API for CWE vulnerabilities and implement defensive code patch", "mis-001")

    assert len(graph.nodes) == 3
    node_ids = set(graph.nodes.keys())
    assert {"sec_analysis", "code_patch", "sec_verify"}.issubset(node_ids)

    # Verify dependency chain: sec_analysis -> code_patch -> sec_verify
    assert graph.nodes["code_patch"].dependencies == {"sec_analysis"}
    assert graph.nodes["sec_verify"].dependencies == {"code_patch"}


@pytest.mark.asyncio
async def test_hierarchical_orchestrator_end_to_end():
    """Verify complete hierarchical orchestration mission execution."""
    orchestrator = HierarchicalOrchestrator()

    # Create a 2-stage graph: Stage 1: [task_planning], Stage 2: [problem_solving]
    graph = TaskGraph(mission_id="hier-test", objective="Plan and solve architecture optimization")
    graph.add_node(task_id="t_plan", assigned_agent="task_planning", prompt="Decompose architecture targets")
    graph.add_node(
        task_id="t_solve",
        assigned_agent="problem_solving",
        prompt="Solve optimization strategy",
        dependencies=["t_plan"],
        context_keys={"t_plan": "plan_summary"},
    )

    mission = await orchestrator.execute_hierarchical_mission(
        objective="Plan and solve architecture optimization",
        custom_graph=graph,
        consensus_strategy=ConsensusStrategy.MAJORITY_VOTE,
        verifying_agents=["problem_solving"],
    )

    assert isinstance(mission, OrchestrationMission)
    assert mission.status == "completed"
    assert mission.all_succeeded is True
    assert len(mission.results) == 2
    assert "JARVIS Multi-Agent Mission Synthesis" in mission.synthesized_summary
    assert len(mission.timeline) >= 3


# 5. Intent Router /orchestrate Test
def test_router_orchestrate_intent():
    """Verify IntentRouter routes /orchestrate and multi-step tasks."""
    router = IntentRouter()

    dec1 = router.route("/orchestrate Audit cybersecurity, generate patches, and verify defenses")
    assert dec1.intent == IntentType.MULTI_STEP_TASK
    assert dec1.agent_name == "HierarchicalOrchestrator"
    assert dec1.needs_agent is True
    assert "Audit cybersecurity" in dec1.extracted_query

    dec2 = router.route("please coordinate agents to analyze data and draft an executive briefing")
    assert dec2.intent == IntentType.MULTI_STEP_TASK
    assert dec2.agent_name == "HierarchicalOrchestrator"
