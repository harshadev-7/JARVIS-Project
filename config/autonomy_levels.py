"""
JARVIS AI — Autonomy Level Configuration
=========================================
Defines what JARVIS is allowed to do at each autonomy level.
This is a core security/safety mechanism — no action bypasses this.

Levels:
  1 = Advisor       → Answer and inform only
  2 = Researcher    → Research, analyze, recommend
  3 = Planner       → Prepare actions, require approval
  4 = Executor      → Execute approved tasks
  5 = Autonomous    → Run pre-authorized workflows independently
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class AutonomyConfig:
    """Defines permissions for a given autonomy level."""
    level: int
    name: str
    description: str
    can_answer: bool
    can_research: bool
    can_plan: bool
    can_execute: bool
    can_run_autonomous: bool
    requires_confirmation_for: List[str]


AUTONOMY_LEVELS: dict[int, AutonomyConfig] = {
    1: AutonomyConfig(
        level=1,
        name="Advisor",
        description="Answer questions and provide information only. No actions taken.",
        can_answer=True,
        can_research=False,
        can_plan=False,
        can_execute=False,
        can_run_autonomous=False,
        requires_confirmation_for=["research", "planning", "execution", "automation"],
    ),
    2: AutonomyConfig(
        level=2,
        name="Researcher",
        description="Research, analyze data, and recommend solutions. No execution.",
        can_answer=True,
        can_research=True,
        can_plan=False,
        can_execute=False,
        can_run_autonomous=False,
        requires_confirmation_for=["planning", "execution", "automation"],
    ),
    3: AutonomyConfig(
        level=3,
        name="Planner",
        description="Prepare complete action plans and present them for approval before execution.",
        can_answer=True,
        can_research=True,
        can_plan=True,
        can_execute=False,
        can_run_autonomous=False,
        requires_confirmation_for=["execution", "automation", "file_operations", "system_commands"],
    ),
    4: AutonomyConfig(
        level=4,
        name="Executor",
        description="Execute approved tasks automatically. Sensitive actions still require confirmation.",
        can_answer=True,
        can_research=True,
        can_plan=True,
        can_execute=True,
        can_run_autonomous=False,
        requires_confirmation_for=["delete_files", "financial", "send_email", "social_post", "system_changes"],
    ),
    5: AutonomyConfig(
        level=5,
        name="Autonomous",
        description="Execute predefined recurring workflows independently within strict pre-authorized permissions.",
        can_answer=True,
        can_research=True,
        can_plan=True,
        can_execute=True,
        can_run_autonomous=True,
        requires_confirmation_for=["financial", "permanent_delete", "external_communication_to_new_contacts"],
    ),
}


def get_autonomy_config(level: int) -> AutonomyConfig:
    """Retrieve autonomy config for a given level. Clamps to valid range."""
    level = max(1, min(5, level))
    return AUTONOMY_LEVELS[level]


def check_permission(level: int, action: str) -> tuple[bool, str]:
    """
    Check if a given action is permitted at the specified autonomy level.

    Returns:
        (allowed: bool, reason: str)
    """
    config = get_autonomy_config(level)

    action_map = {
        "answer": config.can_answer,
        "research": config.can_research,
        "plan": config.can_plan,
        "execute": config.can_execute,
        "autonomous": config.can_run_autonomous,
    }

    if action in action_map:
        allowed = action_map[action]
        if allowed:
            return True, f"Action '{action}' permitted at autonomy level {level} ({config.name})."
        else:
            return False, (
                f"Action '{action}' is not permitted at autonomy level {level} ({config.name}). "
                f"Increase autonomy level to enable this."
            )

    # Check if action requires confirmation
    needs_confirm = action in config.requires_confirmation_for
    if needs_confirm:
        return False, f"Action '{action}' requires explicit user confirmation at level {level}."

    return True, f"Action '{action}' permitted."
