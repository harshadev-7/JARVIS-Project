"""JARVIS Multi-Agent Orchestrator — Hierarchical Orchestrator Core
===================================================================
Master coordinator managing hierarchical goal decomposition, Directed Acyclic
Graph (DAG) execution, circuit breaker fault tolerance, and multi-agent consensus.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from core.logger import log
from core.orchestrator.agent_registry import agent_registry
from core.orchestrator.base_agent import AgentTaskResult
from core.orchestrator.circuit_breaker import circuit_breaker_registry
from core.orchestrator.consensus_engine import ConsensusEngine, ConsensusStrategy, consensus_engine
from core.orchestrator.orchestrator import OrchestrationMission
from core.orchestrator.supervisor_agent import SupervisorAgent, supervisor_agent
from core.orchestrator.task_graph import CyclicDependencyError, TaskGraph, TaskNode, TaskStatus

logger = logging.getLogger(__name__)


class HierarchicalOrchestrator:
    """Master orchestrator unifying DAG task scheduling, circuit breakers, and consensus."""

    def __init__(
        self,
        supervisor: Optional[SupervisorAgent] = None,
        consensus: Optional[ConsensusEngine] = None,
    ):
        self.supervisor = supervisor or supervisor_agent
        self.consensus = consensus or consensus_engine
        self.mission_history: List[OrchestrationMission] = []

    async def execute_hierarchical_mission(
        self,
        objective: str,
        custom_graph: Optional[TaskGraph] = None,
        context: Optional[Dict[str, Any]] = None,
        consensus_strategy: Optional[ConsensusStrategy] = None,
        verifying_agents: Optional[List[str]] = None,
        autonomy_level: int = 4,
    ) -> OrchestrationMission:
        """
        Coordinates full multi-agent mission from goal decomposition to final consensus.
        """
        mission_id = f"mis-hier-{abs(hash(objective)) % 1000000}"
        start_time = datetime.now(timezone.utc)

        # 1. Decompose objective into DAG TaskGraph
        graph = custom_graph or self.supervisor.decompose_objective(objective, mission_id)
        graph.validate_dag()

        stages = graph.get_execution_stages()
        participating_agents = list({node.assigned_agent for node in graph.nodes.values()})

        mission = OrchestrationMission(
            mission_id=mission_id,
            objective=objective,
            participating_agents=participating_agents,
            results=[],
            synthesized_summary="",
            all_succeeded=True,
            status="running",
            agent_status=[{"agent": a, "status": "queued", "summary": "Awaiting stage dispatch."} for a in participating_agents],
            timeline=[
                {
                    "phase": "decomposition",
                    "agent": "SupervisorAgent",
                    "summary": f"Decomposed into {len(graph.nodes)} tasks across {len(stages)} stages.",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            ],
        )

        logger.info("Hierarchical mission %s started with %d stages.", mission_id, len(stages))

        # Helper executor for resilient execution and fallback
        async def run_agent(agent_key: str, task_prompt: str, task_context: Dict[str, Any]) -> AgentTaskResult:
            agent = agent_registry.get_agent(agent_key)
            if not agent:
                raise ValueError(f"Agent '{agent_key}' not found in registry.")
            return await agent.execute(task_prompt, context=task_context)

        # 2. Stage-by-Stage Concurrent Execution
        for stage_idx, stage_nodes in enumerate(stages, start=1):
            logger.info("Dispatching Stage %d (%d parallel tasks).", stage_idx, len(stage_nodes))

            async def execute_task_node(node: TaskNode) -> None:
                # Check if prerequisites succeeded
                for dep_id in node.dependencies:
                    dep = graph.nodes.get(dep_id)
                    if not dep or dep.status != TaskStatus.COMPLETED:
                        logger.warning("Task [%s] skipped because dependency [%s] failed or was skipped.", node.task_id, dep_id)
                        node.status = TaskStatus.SKIPPED
                        node.error = f"Prerequisite task '{dep_id}' was not completed."
                        return

                node.status = TaskStatus.RUNNING
                t0 = time.time()
                task_ctx = graph.build_task_context(node.task_id, global_context=context)

                try:
                    res: AgentTaskResult = await circuit_breaker_registry.execute_with_resilience(
                        node.assigned_agent,
                        run_agent,
                        node.assigned_agent,
                        node.prompt,
                        task_ctx,
                        max_retries=node.retry_policy.max_retries,
                        base_delay_sec=node.retry_policy.base_delay_sec,
                        fallback_executor=run_agent,
                    )
                    node.duration_sec = time.time() - t0
                    node.result_summary = res.summary
                    node.result_data = res.data

                    if res.success:
                        node.status = TaskStatus.COMPLETED
                        mission.timeline.append({
                            "phase": "task_completed",
                            "agent": node.assigned_agent,
                            "task_id": node.task_id,
                            "summary": res.summary[:120],
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        })
                    else:
                        node.status = TaskStatus.FAILED
                        node.error = res.error or res.summary
                        mission.all_succeeded = False
                        mission.timeline.append({
                            "phase": "task_failed",
                            "agent": node.assigned_agent,
                            "task_id": node.task_id,
                            "summary": res.summary[:120],
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        })

                except Exception as exc:
                    node.duration_sec = time.time() - t0
                    node.status = TaskStatus.FAILED
                    node.error = str(exc)
                    node.result_summary = f"Execution error: {exc}"
                    mission.all_succeeded = False
                    mission.timeline.append({
                        "phase": "task_exception",
                        "agent": node.assigned_agent,
                        "task_id": node.task_id,
                        "summary": str(exc),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    })

            # Run all tasks in current stage in parallel
            await asyncio.gather(*(execute_task_node(n) for n in stage_nodes))

        # 3. Optional Consensus Verification on Final Output
        consensus_text = None
        if consensus_strategy and verifying_agents and graph.is_all_successful():
            # Get the output from the final stage
            final_node = stages[-1][-1]
            proposal_to_verify = final_node.result_summary

            logger.info("Executing consensus verification on final task output via %s.", consensus_strategy.value)
            consensus_outcome = await self.consensus.evaluate_proposal(
                proposal=proposal_to_verify,
                verifying_agents=verifying_agents,
                agent_executor=run_agent,
                strategy=consensus_strategy,
                context=context,
            )
            consensus_text = (
                f"- **Strategy:** `{consensus_outcome.strategy_used.value}`\n"
                f"- **Agreement:** {'✅ VERIFIED' if consensus_outcome.agreed else '❌ REJECTED'}\n"
                f"- **Consensus Score:** {consensus_outcome.consensus_score * 100:.1f}%\n"
                f"- **Summary:** {consensus_outcome.summary}"
            )
            mission.timeline.append({
                "phase": "consensus_verification",
                "agent": "ConsensusEngine",
                "summary": consensus_outcome.summary,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

        # 4. Synthesize Final Brief
        final_synthesis = self.supervisor.synthesize_mission(
            objective=objective,
            graph=graph,
            consensus_summary=consensus_text,
        )

        # Collect results for mission ledger
        results_list: List[AgentTaskResult] = []
        for n in graph.nodes.values():
            results_list.append(
                AgentTaskResult(
                    agent_name=n.assigned_agent,
                    task_id=n.task_id,
                    success=(n.status == TaskStatus.COMPLETED),
                    data=n.result_data,
                    summary=n.result_summary,
                    error=n.error,
                    duration_sec=n.duration_sec,
                )
            )

        mission.results = results_list
        mission.synthesized_summary = final_synthesis
        mission.status = "completed" if graph.is_all_successful() else "partial"
        mission.completed_at = datetime.now(timezone.utc)

        self.mission_history.insert(0, mission)
        if len(self.mission_history) > 25:
            self.mission_history = self.mission_history[:25]

        return mission


# Global singleton hierarchical orchestrator
hierarchical_orchestrator = HierarchicalOrchestrator()
