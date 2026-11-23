"""JARVIS Multi-Agent Orchestrator — Supervisor Agent
===================================================
High-level supervisory intelligence that decomposes complex, multi-faceted
objectives into Directed Acyclic Graphs (DAGs) of interdependent sub-tasks,
assigns worker agents, and synthesizes unified mission briefings.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from core.orchestrator.task_graph import TaskGraph, RetryPolicy

logger = logging.getLogger(__name__)


class SupervisorAgent:
    """Supervisory agent responsible for goal decomposition and mission synthesis."""

    def __init__(self, name: str = "SupervisorAgent"):
        self.name = name

    def decompose_objective(self, objective: str, mission_id: str) -> TaskGraph:
        """
        Decomposes a user objective into an interdependent TaskGraph.
        Analyzes keywords to assign specialized agents with proper staging.
        """
        graph = TaskGraph(mission_id=mission_id, objective=objective)
        lower = objective.lower()

        # Pattern 1: Security Audit & Patch Pipeline
        if any(w in lower for w in ["audit", "vulnerability", "cwe", "cve", "harden", "secure"]) and any(w in lower for w in ["fix", "patch", "code", "implement", "deploy"]):
            graph.add_node(
                task_id="sec_analysis",
                assigned_agent="cybersecurity",
                prompt=f"Perform defensive vulnerability analysis and threat modeling for: {objective}",
                dependencies=[],
            )
            graph.add_node(
                task_id="code_patch",
                assigned_agent="programming",
                prompt=f"Implement robust code remediation and defensive patches based on identified vulnerabilities.",
                dependencies=["sec_analysis"],
                context_keys={"sec_analysis": "vulnerability_report"},
            )
            graph.add_node(
                task_id="sec_verify",
                assigned_agent="cybersecurity",
                prompt="Verify that implemented code patches fully mitigate the identified risks with zero regressions.",
                dependencies=["code_patch"],
                context_keys={"code_patch": "patch_code"},
            )
            return graph

        # Pattern 2: Research, Code Implementation & Verification
        if any(w in lower for w in ["research", "investigate", "explore", "learn"]) and any(w in lower for w in ["build", "code", "create", "implement", "script"]):
            graph.add_node(
                task_id="research_stage",
                assigned_agent="research",
                prompt=f"Research best practices, modern architecture patterns, and technical requirements for: {objective}",
                dependencies=[],
            )
            graph.add_node(
                task_id="plan_stage",
                assigned_agent="task_planning",
                prompt="Decompose technical research findings into an actionable phased implementation plan.",
                dependencies=["research_stage"],
                context_keys={"research_stage": "research_findings"},
            )
            graph.add_node(
                task_id="coding_stage",
                assigned_agent="programming",
                prompt="Write production-grade, well-tested code following the architectural plan.",
                dependencies=["plan_stage"],
                context_keys={"plan_stage": "implementation_plan"},
            )
            return graph

        # Pattern 3: Data Analysis & Executive Reporting
        if any(w in lower for w in ["analyze", "data", "metrics", "stats"]) and any(w in lower for w in ["report", "summary", "brief", "present"]):
            graph.add_node(
                task_id="data_analysis",
                assigned_agent="data_analysis",
                prompt=f"Process data, compute descriptive statistics, and identify anomalies for: {objective}",
                dependencies=[],
            )
            graph.add_node(
                task_id="executive_brief",
                assigned_agent="content_creation",
                prompt="Draft a polished executive briefing and visualization plan from the statistical analysis.",
                dependencies=["data_analysis"],
                context_keys={"data_analysis": "statistical_summary"},
            )
            return graph

        # Pattern 4: Video Content Creation & Viral Production
        if any(w in lower for w in ["video", "reel", "short", "youtube", "tiktok", "autoclip"]):
            graph.add_node(
                task_id="topic_research",
                assigned_agent="research",
                prompt=f"Identify viral angles, trending hooks, and audience interest for: {objective}",
                dependencies=[],
            )
            graph.add_node(
                task_id="script_creation",
                assigned_agent="content_creation",
                prompt="Synthesize a high-retention script with curiosity hooks and structured timing.",
                dependencies=["topic_research"],
                context_keys={"topic_research": "viral_trends"},
            )
            graph.add_node(
                task_id="video_production",
                assigned_agent="video_editing",
                prompt="Assemble timeline, apply visual grading, overlay captions, and export video.",
                dependencies=["script_creation"],
                context_keys={"script_creation": "script_content"},
            )
            return graph

        # General Multi-Step Fallback: Planning -> Deduction -> Synthesis
        graph.add_node(
            task_id="plan_step",
            assigned_agent="task_planning",
            prompt=f"Break down the following complex objective into structured milestones: {objective}",
            dependencies=[],
        )
        graph.add_node(
            task_id="reasoning_step",
            assigned_agent="problem_solving",
            prompt=f"Execute rigorous 10-step deductive problem solving on the planned milestones.",
            dependencies=["plan_step"],
            context_keys={"plan_step": "milestones"},
        )
        return graph

    def synthesize_mission(
        self,
        objective: str,
        graph: TaskGraph,
        consensus_summary: Optional[str] = None,
    ) -> str:
        """Synthesizes all executed task outcomes into a comprehensive mission brief."""
        lines = [
            f"# 🎯 JARVIS Multi-Agent Mission Synthesis",
            f"**Mission Objective:** {objective}",
            f"**Execution Status:** {'✅ ALL TASKS COMPLETED' if graph.is_all_successful() else '⚠️ PARTIAL COMPLETION'}",
            "",
            "## 📋 Task Execution Stages & Outputs:",
        ]

        stages = graph.get_execution_stages()
        for idx, stage in enumerate(stages, start=1):
            lines.append(f"### Stage {idx} (Parallel Execution):")
            for node in stage:
                icon = "✅" if node.status.value == "completed" else "❌"
                lines.append(
                    f"- {icon} **[{node.assigned_agent}] `{node.task_id}`** ({node.duration_sec:.2f}s):\n"
                    f"  {node.result_summary}"
                )
            lines.append("")

        if consensus_summary:
            lines.append("## 🤝 Multi-Agent Consensus Verification:")
            lines.append(consensus_summary)
            lines.append("")

        return "\n".join(lines)


# Global singleton supervisor
supervisor_agent = SupervisorAgent()
