"""J.A.R.V.I.S. Independent Multi-Agent System (Phase 9).

Contains the 8 specialized agents required by Section 10 of the Master Blueprint:
1. ResearchAgent (agents.research)
2. PlanningAgent (agents.planning)
3. CodingAgent (agents.coding)
4. KnowledgeAgent (agents.knowledge)
5. MemoryAgent (agents.memory_agent)
6. ComputerControlAgent (agents.computer_control)
7. VoiceAgent (agents.voice)
8. TaskExecutionAgent (agents.task_execution)
"""

from typing import Any, Dict, List, Optional

from core.logger import log
from core.orchestrator.base_agent import AgentMessage, AgentTaskResult, BaseAgent

from .coding import CodingAgent
from .computer_control import ComputerControlAgent
from .knowledge import KnowledgeAgent
from .memory_agent import MemoryAgent
from .planning import PlanningAgent
from .research import ResearchAgent
from .task_execution import TaskExecutionAgent
from .voice import VoiceAgent


class IndependentAgentSystem:
    """Master registry and inter-agent communication hub for J.A.R.V.I.S."""

    def __init__(self):
        self._agents: Dict[str, BaseAgent] = {}
        self._register_default_agents()

    def _register_default_agents(self) -> None:
        """Register the 8 specialized domain agents."""
        agents = [
            ResearchAgent(),
            PlanningAgent(),
            CodingAgent(),
            KnowledgeAgent(),
            MemoryAgent(),
            ComputerControlAgent(),
            VoiceAgent(),
            TaskExecutionAgent(),
        ]
        for a in agents:
            # Register by class name, lowercase, no "Agent", and snake_case
            self._agents[a.name] = a
            self._agents[a.name.lower()] = a
            key = a.name.replace("Agent", "").lower()
            self._agents[key] = a

            # Snake case conversions
            import re
            snake = re.sub(r"(?<!^)(?=[A-Z])", "_", a.name).lower()
            self._agents[snake] = a
            snake_no_agent = snake.replace("_agent", "")
            self._agents[snake_no_agent] = a

    def get_agent(self, name: str) -> Optional[BaseAgent]:
        """Retrieve agent by name (e.g. 'coding', 'CodingAgent', 'research')."""
        return self._agents.get(name) or self._agents.get(name.lower())

    def list_agents(self) -> List[Dict[str, Any]]:
        """Return roster of registered agents and their capabilities."""
        unique_agents = {a.name: a for a in self._agents.values()}.values()
        return [
            {
                "name": a.name,
                "role": a.role,
                "capabilities": a.capabilities,
                "autonomy_level": a.required_autonomy_level,
            }
            for a in unique_agents
        ]

    def send_message(
        self,
        sender: str,
        recipient: str,
        message_type: str,
        content: Any,
    ) -> bool:
        """Deliver an inter-agent message to a recipient's inbox."""
        target = self.get_agent(recipient)
        if not target:
            log.warning(f"Message delivery failed: Recipient agent '{recipient}' not found.")
            return False

        msg = AgentMessage(
            sender=sender,
            recipient=target.name,
            message_type=message_type,
            content=content,
        )
        target.receive_message(msg)
        return True

    async def execute_task(
        self,
        agent_name: str,
        task: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> AgentTaskResult:
        """Route and execute a task on a specialized agent."""
        agent = self.get_agent(agent_name)
        if not agent:
            raise ValueError(f"Unknown agent '{agent_name}'. Available: {[a['name'] for a in self.list_agents()]}")

        # Check autonomy level
        ctx = context or {}
        active_level = ctx.get("autonomy_level", 2)
        if agent.required_autonomy_level > active_level:
            return AgentTaskResult(
                agent_name=agent.name,
                task_id="auth_blocked",
                success=False,
                error="AutonomyPermissionDenied",
                summary=f"Agent '{agent.name}' requires Autonomy Level {agent.required_autonomy_level} (Current: {active_level}).",
            )

        return await agent.execute(task, context=ctx)


# Singleton agent system instance
agent_system = IndependentAgentSystem()

__all__ = [
    "CodingAgent",
    "ComputerControlAgent",
    "IndependentAgentSystem",
    "KnowledgeAgent",
    "MemoryAgent",
    "PlanningAgent",
    "ResearchAgent",
    "TaskExecutionAgent",
    "VoiceAgent",
    "agent_system",
]
