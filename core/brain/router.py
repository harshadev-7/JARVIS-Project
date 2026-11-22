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
    MEMORY_STORE = "memory_store"
    MEMORY_QUERY = "memory_query"
    VIDEO_EDITING = "video_editing"
    COMPUTER_VISION = "computer_vision"
    CYBERSECURITY = "cybersecurity"
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
        r"\b(solve problem|reason|plan mission|step by step|how should i approach)\b",
        r"\b(explain the difference|difference between|compare|relationship between)\b",
        r"\b(how many|how much|how long|what is the angle|angle between|clock shows|speed of|distance between)\b",
        r"\b(all but \d+|half my age|simple interest|compound interest|handshakes?|shake hands)\b",
        r"\b(riddle|puzzle|logic problem|word problem|reasoning)\b",
        r"\b(two workers|pipes? fill|tank fills|train passes|trains? crossing)\b",
    ]
    IDENTITY_PATTERNS = [
        r"\b(who are you|what are you|your name|what can you do|capabilities|jarvis)\b"
    ]
    VIDEO_PATTERNS = [
        r"\b(video\s+edit|edit\s+video|trim\s+video|cut\s+video|merge\s+video|concatenate\s+video)\b",
        r"\b(make\s+a\s+reel|create\s+short|vertical\s+video|sync\s+to\s+beat|highlight\s+reel)\b",
        r"\b(burn\s+subtitles|add\s+subtitles|export\s+video|video\s+timeline)\b",
        r"\b(play|open|show|watch|listen\s+to|start)\s+(?:that\s+|the\s+|this\s+|my\s+)?(?:video|reel|audio|media|clip|track|short)\b",
        r"\b(play|open|watch)\s+(?:that|this|it)\b",
        r"^(?:play|open|launch)\s+(?:video|reel|clip)$",
    ]
    VISION_PATTERNS = [
        r"\b(image\s+classif|classif\w*\s+image|inspect\s+image|analyze\s+image|view\s+image)\b",
        r"\b(ocr|extract\s+text|read\s+text\s+from|scan\s+text|detect\s+objects|bounding\s+box)\b",
        r"\b(visual\s+search|reverse\s+image|scene\s+understand|image\s+quality|smart\s+crop|auto\s+frame)\b",
    ]
    CYBERSECURITY_PATTERNS = [
        r"\b(cwe-\d+|cve-\d+|vulnerability|cybersecurity|hardening|owasp|sql injection|xss|csrf|ssrf|zero trust|dnssec|tls 1\.3|syn flood|privilege escalation)\b",
        r"\b(mitigate|remediation|security checklist|firewall rules|linux hardening|windows hardening)\b",
    ]
    MEMORY_QUERY_PATTERNS = [
        r"\b(what do you know about me|who am i|who i am|what is my name|what's my name)\b",
        r"\b(?:now\s+)?say\s+my\s+name\b",
        r"\b(?:tell|recall|repeat)\s+(?:me\s+)?my\s+name\b",
        r"\b(?:tell|what do you know|know)\s+(?:me\s+)?about\s+me\b",
        r"\btell\s+about\s+me\b",
        r"\babout\s+me\b",
        r"\bmy\s+profile\b",
        r"\bmy\s+background\b",
        r"\bwhat\s+about\s+me\b",
        r"\b(?:did|when did)\s+i\s+tell\s+you\b",
        r"\bi\s+did(?:n't)?\s+tell\s+you\b",
        r"\b(?:do you know|remember)\s+my\s+name\b",
        r"\b(what (?:programming )?language|what (?:exam|examination)|what am i preparing for|what are my interests)\b",
        r"\b(do you remember|what do you remember|our previous conversation|tell me what you remember|past discussion)\b",
    ]
    MEMORY_STORE_PATTERNS = [
        r"\bmy name is\s+[A-Za-z]+\b",
        r"\bcall me\s+[A-Za-z]+\b",
        r"\bi am\s+[A-Z][a-z]+\b",
        r"\bi am (?:a|an)\s+[^.]+\b",
        r"\binterested in\s+[^.]+\b",
        r"\bpreparing for\s+[^.]+\b",
        r"\bremember (?:that\s+)?",
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

        # 5b. Video Editing Automation
        if lower.startswith("/video") or lower.startswith("/reel") or lower.startswith("/short") or any(
            re.search(p, lower) for p in self.VIDEO_PATTERNS
        ):
            query = re.sub(r"^/(video|reel|short)\s*", "", text).strip()
            return RoutingDecision(
                intent=IntentType.VIDEO_EDITING,
                confidence=0.92,
                needs_agent=True,
                agent_name="VideoEditingAgent",
                extracted_query=query or text,
            )

        # 5c. Computer Vision & Visual Intelligence
        if lower.startswith("/vision") or lower.startswith("/ocr") or lower.startswith("/image") or any(
            re.search(p, lower) for p in self.VISION_PATTERNS
        ):
            query = re.sub(r"^/(vision|ocr|image)\s*", "", text).strip()
            return RoutingDecision(
                intent=IntentType.COMPUTER_VISION,
                confidence=0.92,
                needs_agent=True,
                agent_name="VisionAgent",
                extracted_query=query or text,
            )

        # 5d. Cybersecurity Intelligence & Defensive Advisory
        if lower.startswith("/secinfo") or lower.startswith("/cwe") or any(
            re.search(p, lower) for p in self.CYBERSECURITY_PATTERNS
        ):
            query = re.sub(r"^/(secinfo|cwe)\s*", "", text).strip()
            return RoutingDecision(
                intent=IntentType.CYBERSECURITY,
                confidence=0.92,
                needs_agent=True,
                agent_name="CybersecurityMonitoringAgent",
                extracted_query=query or text,
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

        # 7b. Direct Match against Offline Learned Pattern Schemas
        try:
            from core.planner.learned_pattern_engine import learned_pattern_engine
            if learned_pattern_engine.match_and_solve(text) is not None:
                return RoutingDecision(
                    intent=IntentType.REASONING,
                    confidence=0.95,
                    needs_agent=True,
                    agent_name="ProblemSolvingAgent",
                    extracted_query=text,
                )
        except Exception:
            pass

        # 8. Memory & Profile Query
        if any(re.search(p, lower) for p in self.MEMORY_QUERY_PATTERNS):
            return RoutingDecision(
                intent=IntentType.MEMORY_QUERY,
                confidence=0.95,
                needs_agent=False,
                agent_name="MemoryAgent",
                extracted_query=text,
            )

        # 9. Memory & User Profile Statement / Update
        if any(re.search(p, lower) for p in self.MEMORY_STORE_PATTERNS):
            return RoutingDecision(
                intent=IntentType.MEMORY_STORE,
                confidence=0.95,
                needs_agent=False,
                agent_name="MemoryAgent",
                extracted_query=text,
            )

        # 10. Greetings
        if any(re.search(p, lower) for p in self.GREETING_PATTERNS) and len(text.split()) <= 4:
            return RoutingDecision(
                intent=IntentType.GREETING,
                confidence=0.90,
                needs_agent=False,
                agent_name=None,
                extracted_query=text,
            )

        # 11. Identity & Capabilities
        if any(re.search(p, lower) for p in self.IDENTITY_PATTERNS):
            return RoutingDecision(
                intent=IntentType.IDENTITY,
                confidence=0.92,
                needs_agent=False,
                agent_name=None,
                extracted_query=text,
            )

        # 12. General Conversation -> Handled by Custom Local Transformer Neural Model
        return RoutingDecision(
            intent=IntentType.GENERAL_CONVERSATION,
            confidence=0.80,
            needs_agent=False,
            agent_name="CustomTransformer",
            extracted_query=text,
        )


__all__ = ["IntentRouter", "IntentType", "RoutingDecision"]
