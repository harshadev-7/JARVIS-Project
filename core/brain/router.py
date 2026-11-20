"""J.A.R.V.I.S. Intent Classifier & Task Router.

Analyzes user input, classifies intent, and determines whether
a specialized autonomous agent or local Transformer model should handle the request.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
from typing import Any, Dict, List, Optional


class IntentType(str, Enum):
    GREETING = "greeting"
    STATUS_DIAGNOSTICS = "status_diagnostics"
    REASONING = "reasoning"
    RESEARCH = "research"
    PROGRAMMING = "programming"
    MATH = "math"
    DEVICE_CONTROL = "device_control"
    COMPUTER_AUTOMATION = "computer_automation"
    IDENTITY = "identity"
    GENERAL_CONVERSATION = "general_conversation"


@dataclass
class RoutingDecision:
    """The decision result produced by the Intent Router."""
    intent: IntentType
    confidence: float
    needs_agent: bool
    agent_name: Optional[str] = None
    extracted_query: str = ""
    metadata: Dict[str, Any] = None


class IntentRouter:
    """Deterministic, high-speed offline intent classifier and agent selector."""

    # Keywords and regex patterns for rapid offline routing
    GREETING_PATTERNS = [
        r"\b(hello|hi|hey|good morning|good evening|good afternoon|greetings)\b"
    ]
    STATUS_PATTERNS = [
        r"\b(status|diagnostics|system check|health|cpu usage|ram usage|how are you)\b"
    ]
    MATH_PATTERNS = [
        r"\b(calculate|compute|math|solve|multiply|divide|add|subtract)\b",
        r"\d+\s*[\+\-\*\/\^]\s*\d+",
    ]
    CODE_PATTERNS = [
        r"\b(code|python|syntax|debug|function|class|script|def |import )\b",
        r"[{};\[\]]",
    ]
    DEVICE_PATTERNS = [
        r"\b(device|phone|tablet|pair|connect to device|devices near|nearby device)\b"
    ]
    AUTOMATION_PATTERNS = [
        r"\b(open|launch|close|terminate|kill process|start app|notepad|calc|explorer)\b"
    ]
    RESEARCH_PATTERNS = [
        r"\b(research|search|look up|find online|who is|what is the history of)\b"
    ]
    REASONING_PATTERNS = [
        r"\b(solve problem|reason|plan mission|step by step|how should i approach)\b"
    ]
    IDENTITY_PATTERNS = [
        r"\b(who are you|what are you|your name|what can you do|capabilities|jarvis)\b"
    ]

    def route(self, user_message: str, history: Optional[List[dict]] = None) -> RoutingDecision:
        """Classify input intent and assign the optimal handling engine."""
        text = user_message.strip()
        lower = text.lower()

        # 1. System Diagnostics / Telemetry
        if any(re.search(p, lower) for p in self.STATUS_PATTERNS):
            return RoutingDecision(
                intent=IntentType.STATUS_DIAGNOSTICS,
                confidence=0.95,
                needs_agent=True,
                agent_name="SystemMonitor",
                extracted_query=text,
            )

        # 2. Mathematical Calculations
        if any(re.search(p, lower) for p in self.MATH_PATTERNS) and any(c.isdigit() for c in lower):
            return RoutingDecision(
                intent=IntentType.MATH,
                confidence=0.92,
                needs_agent=True,
                agent_name="MathematicalAgent",
                extracted_query=text,
            )

        # 3. Device Mesh / Phone / Tablet
        if any(re.search(p, lower) for p in self.DEVICE_PATTERNS):
            return RoutingDecision(
                intent=IntentType.DEVICE_CONTROL,
                confidence=0.90,
                needs_agent=True,
                agent_name="DeviceManagementAgent",
                extracted_query=text,
            )

        # 4. Programming & Code Analysis
        if any(re.search(p, lower) for p in self.CODE_PATTERNS) and any(
            kw in lower for kw in ["def ", "class ", "import ", "error", "debug", "syntax"]
        ):
            return RoutingDecision(
                intent=IntentType.PROGRAMMING,
                confidence=0.88,
                needs_agent=True,
                agent_name="ProgrammingAgent",
                extracted_query=text,
            )

        # 5. Safe Computer Automation
        if any(re.search(p, lower) for p in self.AUTOMATION_PATTERNS) and any(
            app in lower for app in ["notepad", "calc", "calculator", "vscode", "code", "browser", "app"]
        ):
            return RoutingDecision(
                intent=IntentType.COMPUTER_AUTOMATION,
                confidence=0.85,
                needs_agent=True,
                agent_name="ComputerAutomationAgent",
                extracted_query=text,
            )

        # 6. Deep Research Query
        if lower.startswith("/research") or (
            any(re.search(p, lower) for p in self.RESEARCH_PATTERNS) and len(text.split()) > 3
        ):
            query = text.replace("/research", "").strip()
            return RoutingDecision(
                intent=IntentType.RESEARCH,
                confidence=0.85,
                needs_agent=True,
                agent_name="ResearchAgent",
                extracted_query=query,
            )

        # 7. Multi-step Reasoning / Problem Solving
        if lower.startswith("/solve") or any(re.search(p, lower) for p in self.REASONING_PATTERNS):
            query = text.replace("/solve", "").strip()
            return RoutingDecision(
                intent=IntentType.REASONING,
                confidence=0.90,
                needs_agent=True,
                agent_name="ProblemSolvingAgent",
                extracted_query=query,
            )

        # 8. Greetings
        if any(re.search(p, lower) for p in self.GREETING_PATTERNS) and len(text.split()) <= 4:
            return RoutingDecision(
                intent=IntentType.GREETING,
                confidence=0.95,
                needs_agent=False,
                agent_name=None,
                extracted_query=text,
            )

        # 9. Identity & Capabilities
        if any(re.search(p, lower) for p in self.IDENTITY_PATTERNS):
            return RoutingDecision(
                intent=IntentType.IDENTITY,
                confidence=0.92,
                needs_agent=False,
                agent_name=None,
                extracted_query=text,
            )

        # 10. General Conversation -> Handled by Custom Local Transformer Neural Model
        return RoutingDecision(
            intent=IntentType.GENERAL_CONVERSATION,
            confidence=0.80,
            needs_agent=False,
            agent_name="CustomTransformer",
            extracted_query=text,
        )


__all__ = ["IntentRouter", "IntentType", "RoutingDecision"]
