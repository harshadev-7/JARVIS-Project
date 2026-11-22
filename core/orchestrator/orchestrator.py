"""
JARVIS AI — Multi-Agent Orchestrator
====================================
Coordinates workflows across the 15 specialized agents.
Dispatches tasks, facilitates inter-agent communication,
aggregates parallel/sequential findings, and produces unified mission briefs.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from pydantic import BaseModel, Field

from core.orchestrator.base_agent import AgentTaskResult, AgentMessage
from core.orchestrator.agent_registry import agent_registry
from core.logger import log
from config.settings import settings


class OrchestrationMission(BaseModel):
    """The outcome of multi-agent coordination."""
    mission_id: str
    objective: str
    participating_agents: List[str]
    results: List[AgentTaskResult]
    synthesized_summary: str
    all_succeeded: bool
    status: str = "running"
    agent_status: List[Dict[str, str]] = Field(default_factory=list)
    timeline: List[Dict[str, Any]] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None


class MultiAgentOrchestrator:
    """Central commander orchestrating agent cooperation."""

    def __init__(self):
        self.mission_history: List[OrchestrationMission] = []
        self.active_missions: Dict[str, OrchestrationMission] = {}

    def list_recent_missions(self, limit: int = 10) -> List[OrchestrationMission]:
        """Return the most recent mission records for control surfaces and the HUD."""
        return self.mission_history[:limit]

    def list_active_missions(self, limit: int = 10) -> List[OrchestrationMission]:
        """Return mission snapshots currently in the active ledger, ordered newest first."""
        missions = [m for m in self.mission_history if m.status in {"running", "completed", "partial"}]
        return missions[:limit]

    def get_mission(self, mission_id: str) -> Optional[OrchestrationMission]:
        """Lookup a mission by ID from the active and recent ledger."""
        if mission_id in self.active_missions:
            return self.active_missions[mission_id]

        for mission in self.mission_history:
            if mission.mission_id == mission_id:
                return mission
        return None

    async def execute_mission(
        self,
        objective: str,
        required_agents: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
        autonomy_level: Optional[int] = None,
    ) -> OrchestrationMission:
        """
        Coordinate multiple specialized agents to solve a complex objective.
        Example: "Develop and deploy a website" -> [programming, research, cybersecurity, task_planning]
        """
        # Determine agent roster
        agent_keys = required_agents or self._determine_agent_roster(objective)

        active_level = autonomy_level if autonomy_level is not None else settings.DEFAULT_AUTONOMY_LEVEL
        blocked = []
        for key in agent_keys:
            agent = agent_registry.get_agent(key)
            if agent and agent.required_autonomy_level > active_level:
                blocked.append(f"{agent.name} requires autonomy level {agent.required_autonomy_level} (current level {active_level})")

        if blocked:
            raise PermissionError(
                f"Autonomy policy exceeded for this mission. {', '.join(blocked)}. "
                "Increase the active autonomy level or reduce the agent roster."
            )

        mission_id = f"mis-{abs(hash(objective)) % 1000000}"
        now = datetime.now(timezone.utc)
        mission = OrchestrationMission(
            mission_id=mission_id,
            objective=objective,
            participating_agents=agent_keys,
            results=[],
            synthesized_summary=f"**Multi-Agent Mission Synthesis:** {objective}\n",
            all_succeeded=True,
            status="running",
            agent_status=[{"agent": key, "status": "queued", "summary": "Awaiting execution."} for key in agent_keys],
            timeline=[
                {"phase": "dispatch", "agent": "system", "summary": f"Mission {mission_id} dispatched for execution.", "timestamp": now.isoformat()},
                {"phase": "planning", "agent": "system", "summary": f"Preparing agent roster: {', '.join(agent_keys)}.", "timestamp": now.isoformat()},
            ],
        )
        self.active_missions[mission.mission_id] = mission
        self.mission_history.insert(0, mission)
        if len(self.mission_history) > 25:
            self.mission_history = self.mission_history[:25]

        log.info(f"Orchestrating mission: '{objective}' across agents: {agent_keys}")

        results: List[AgentTaskResult] = []
        all_succeeded = True

        for key in agent_keys:
            agent = agent_registry.get_agent(key)
            if not agent:
                log.warning(f"Agent '{key}' not found in registry. Skipping.")
                continue

            for item in mission.agent_status:
                if item["agent"] == key:
                    item["status"] = "running"
                    item["summary"] = f"Executing mission objective: {objective}"
                    break

            mission.timeline.append({
                "phase": "agent_running",
                "agent": key,
                "summary": f"{agent.name} began execution for: {objective}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

            try:
                res = await agent.execute(objective, context=context)
                results.append(res)
                if not res.success:
                    all_succeeded = False
                for item in mission.agent_status:
                    if item["agent"] == key:
                        item["status"] = "completed" if res.success else "failed"
                        item["summary"] = res.summary
                        break
                mission.timeline.append({
                    "phase": "agent_result",
                    "agent": key,
                    "summary": res.summary,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
            except Exception as e:
                log.error(f"Agent [{key}] failed: {e}")
                results.append(
                    AgentTaskResult(
                        agent_name=agent.name,
                        task_id="err-01",
                        success=False,
                        error=str(e),
                        summary=f"Failed execution: {e}",
                    )
                )
                all_succeeded = False
                for item in mission.agent_status:
                    if item["agent"] == key:
                        item["status"] = "failed"
                        item["summary"] = str(e)
                        break
                mission.timeline.append({
                    "phase": "agent_failure",
                    "agent": key,
                    "summary": f"Execution failed: {e}",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })

        # Synthesize collective intelligence
        summary_lines = [f"**Multi-Agent Mission Synthesis:** {objective}\n"]
        for r in results:
            status_icon = "✅" if r.success else "❌"
            summary_lines.append(f"{status_icon} **{r.agent_name}:** {r.summary}")

        synthesized = "\n".join(summary_lines)
        mission.results = results
        mission.synthesized_summary = synthesized
        mission.all_succeeded = all_succeeded
        mission.status = "completed" if all_succeeded else "partial"
        for item in mission.agent_status:
            if item["status"] == "running":
                item["status"] = "completed" if all_succeeded else "partial"
        mission.timeline.append({
            "phase": "mission_complete",
            "agent": "system",
            "summary": f"Mission completed with {len(results)} agent result(s).",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        mission.completed_at = datetime.now(timezone.utc)
        self.active_missions[mission.mission_id] = mission
        return mission

    def _determine_agent_roster(self, objective: str) -> List[str]:
        """Heuristically assign the most suitable agents based on objective keywords."""
        lower = objective.lower()
        roster = set()

        if any(w in lower for w in ["search", "find", "news", "trend", "lookup", "research"]):
            roster.add("research")
        if any(w in lower for w in ["code", "program", "build", "api", "website", "script", "debug"]):
            roster.add("programming")
            roster.add("task_planning")
        if any(w in lower for w in ["calculate", "math", "equation", "solve"]):
            roster.add("mathematical")
        if any(w in lower for w in ["data", "average", "metrics", "analytics"]):
            roster.add("data_analysis")
        if any(w in lower for w in ["device", "phone", "laptop", "pair"]):
            roster.add("device_management")
        if any(w in lower for w in ["social", "youtube", "post", "instagram"]):
            roster.add("social_media")
            roster.add("content_creation")
        if any(w in lower for w in ["video", "edit", "render", "short"]):
            roster.add("video_editing")
        if any(w in lower for w in ["security", "audit", "permission", "safe"]):
            roster.add("cybersecurity")

        if not roster:
            # Default to problem solving + task planning
            roster.add("problem_solving")
            roster.add("task_planning")

        return list(roster)


orchestrator = MultiAgentOrchestrator()
