"""J.A.R.V.I.S. Intent Classifier & Task Router.

Analyzes user input, classifies intent into discrete categories,
and determines whether a specialized autonomous agent, deterministic solver,
creative engine, or local Transformer model should handle the request.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
from typing import Any, Dict, List, Optional


class IntentType(str, Enum):
    # 16 Canonical Intents Required by Directive
    GENERAL_QUESTION = "general_question"
    KNOWLEDGE_QUESTION = "knowledge_question"
    MATHEMATICAL_PROBLEM = "mathematical_problem"
    LOGICAL_REASONING = "logical_reasoning"
    PROBLEM_SOLVING = "problem_solving"
    CREATIVE_GENERATION = "creative_generation"
    IMAGE_PROMPT_GENERATION = "image_prompt_generation"
    VIDEO_PROMPT_GENERATION = "video_prompt_generation"
    CODE_GENERATION = "code_generation"
    INTERNET_RESEARCH = "internet_research"
    FILE_OPERATION = "file_operation"
    DESKTOP_AUTOMATION = "desktop_automation"
    VIDEO_EDITING = "video_editing"
    DEVICE_OPERATION = "device_operation"
    CONVERSATIONAL_INTERACTION = "conversational_interaction"
    MULTI_STEP_TASK = "multi_step_task"

    # Contextual Follow-up & Subsystem-specific aliases
    CONVERSATIONAL_FOLLOWUP = "conversational_followup"
    CODE_EXPLANATION = "code_explanation"
    GREETING = "greeting"
    STATUS_DIAGNOSTICS = "status_diagnostics"
    REASONING = "logical_reasoning"
    RESEARCH = "internet_research"
    PROGRAMMING = "code_generation"
    MATH = "mathematical_problem"
    DEVICE_CONTROL = "device_operation"
    COMPUTER_AUTOMATION = "desktop_automation"
    IDENTITY = "identity"
    MEMORY_STORE = "memory_store"
    MEMORY_QUERY = "memory_query"
    VIDEO_PATTERNS = "video_editing"
    COMPUTER_VISION = "computer_vision"
    CYBERSECURITY = "cybersecurity"
    ADAPTIVE_LEARNING = "adaptive_learning"
    LEARNING = "adaptive_learning"
    SENTINEL_OPERATION = "sentinel_operation"
    VOICE_MODE = "voice_mode"
    GENERAL_CONVERSATION = "conversational_interaction"


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

    # 1. Image Prompt Generation Patterns
    IMAGE_PROMPT_PATTERNS = [
        r"\b(?:generate|create|make|write|give me|design)?\s*(?:an?\s+)?(?:image\s+prompt|prompt\s+for\s+(?:an?\s+)?image|prompt\s+for\s+ai\s+to\s+create\s+(?:an?\s+)?image|prompt\s+to\s+create\s+(?:an?\s+)?image)\b",
        r"\b(?:image\s+prompt\s+for|prompt\s+for\s+(?:an?\s+)?image\s+of)\b",
    ]

    # 2. Video Prompt Generation Patterns
    VIDEO_PROMPT_PATTERNS = [
        r"\b(?:generate|create|make|write|give me)?\s*(?:a\s+)?(?:video\s+prompt|prompt\s+for\s+(?:a\s+)?video|cinematic\s+prompt\s+for\s+video)\b",
    ]

    # 3. Creative Generation Patterns
    CREATIVE_PATTERNS = [
        r"\b(?:create\s+a\s+concept|concept\s+for\s+a\s+sci-fi|sci-fi\s+short\s+film|short\s+film\s+concept)\b",
        r"\b(?:story\s+idea|write\s+a\s+story|write\s+a\s+script|film\s+treatment|movie\s+concept)\b",
        r"\b(?:social\s+media\s+caption|create\s+a\s+caption|logo\s+concept|character\s+description)\b",
    ]

    # 4. Logical Reasoning & Riddles
    LOGICAL_REASONING_PATTERNS = [
        r"\b(?:if\s+\d+\s+machines?\s+produce|machines?\s+produce\s+\d+\s+products?)\b",
        r"\b(?:bat\s+and\s+(?:a\s+)?ball|costs?\s+\$?\d+(?:\.\d+)?\s+more\s+than)\b",
        r"\b(?:riddle|puzzle|logic\s+problem|word\s+problem)\b",
        r"\b(?:two\s+workers|pipes?\s+fill|tank\s+fills|train\s+passes|trains?\s+crossing)\b",
        r"\b(?:all\s+but\s+\d+|half\s+my\s+age|handshakes?|shake\s+hands)\b",
        r"\b(?:clock\s+shows|angle\s+between\s+(?:the\s+)?hands?)\b",
    ]

    # 5. Mathematical Problems
    MATH_PATTERNS = [
        r"\b(?:calculate|compute|math|multiply|divide|add|subtract)\b",
        r"\b\d+(?:\.\d+)?\s*[\+\-\*\/\^]\s*\d+(?:\.\d+)?\b",
        r"\bwhat\s+is\s+\d+(?:\.\d+)?\s*[\+\-\*\/\^]\s*\d+\b",
        r"\b\d+\s+(?:multiplied by|times|plus|minus|divided by)\s+\d+\b",
        r"\b(?:solve\s+for\s+x|algebraic|integral\s+of|derivative\s+of)\b",
    ]

    # 6. Multi-Language Code Generation Patterns
    CODE_PATTERNS = [
        r"\b(?:give\s+me\s+(?:a\s+)?code|write\s+(?:a\s+)?(?:code|program|script|function)|create\s+(?:a\s+)?(?:program|script|code)|show\s+me\s+(?:a\s+)?code)\b",
        r"\b(?:in\s+java|in\s+python|in\s+c\+\+|in\s+c\#|in\s+javascript|in\s+golang|in\s+rust|in\s+c\b|java\s+program|python\s+program|c\+\+\s+program)\b",
        r"\b(?:addition\s+of\s+two\s+numbers|addition\s+two\s+program|add\s+two\s+numbers|reverse\s+a\s+string|check\s+palindrome|palindrome|sorting|fibonacci)\b",
        r"\b(?:code|syntax|debug|function|class|script|def |import |public\s+class)\b",
    ]

    # 6b. Knowledge Question Answering Patterns
    KNOWLEDGE_PATTERNS = [
        r"\b(?:what\s+is\s+(?:an?\s+|the\s+)?(?:ai|artificial\s+intelligence|machine\s+learning|ml|java|python|operating\s+system|os|database|dbms|algorithm|computer\s+network|networking|ram))\b",
        r"\b(?:what\s+is\s+the\s+difference\s+between|difference\s+between|compare\s+\w+\s+and\s+\w+)\b",
        r"\b(?:explain\s+(?:the\s+previous\s+answer\s+in\s+)?(?:simpler|simple)\s+words|explain\s+in\s+simple\s+words)\b",
        r"\b(?:explain\s+(?:java\s+)?inheritance|explain\s+(?:polymorphism|machine\s+learning|artificial\s+intelligence|ai))\b",
    ]

    # 7. Document Summarization
    DOCUMENT_SUMMARY_PATTERNS = [
        r"(?i)^(?:summarize\s+(?:this|the\s+following|the\s+document|the\s+text)|document\s+summary)",
    ]

    # 8. General Questions & Concept Explanations
    GENERAL_QUESTION_PATTERNS = [
        r"\b(?:what\s+is\s+the\s+capital\s+of|who\s+wrote|what\s+is\s+the\s+speed\s+of\s+light)\b",
        r"\b(?:explain\s+what\s+\w+\s+is(?:\s+in\s+simple\s+words)?|what\s+is\s+ram\b|explain\s+(?:java\s+)?inheritance)\b",
    ]

    # 9. Internet Research (Strictly explicit or requiring live 2025/2026 data)
    RESEARCH_PATTERNS = [
        r"^/research\b",
        r"\b(?:search\s+(?:the\s+)?web|search\s+online|look\s+up\s+online|google\s+this|find\s+online)\b",
        r"\b(?:latest\s+developments?\s+in|recent\s+(?:developments?|advancements?|news|releases?)|upcoming\s+releases?)\b",
        r"\b(?:electric\s+vehicle\s+releases|latest\s+quantum\s+computing|latest\s+nvidia\s+gpu)\b",
    ]

    # 10. Follow-up and Context Continuity Words
    FOLLOWUP_PATTERNS = [
        r"^(?:proceed|continue|yes|do it|go ahead|sure|ok|okay|the first one|the second one)\b",
        r"^(?:scanner|predefined|predefined values|console input)\b",
        r"\b(?:make it darker|more cinematic|make it better|make the previous prompt)\b",
        r"\b(?:what are its types|what are the types|its types|inheritance types)\b",
        r"\b(?:give an example|show an example|code example|provide an example)\b",
        r"\b(?:explain the (?:first|second|third|fourth|fifth|2nd|3rd|1st) type)\b",
        r"\b(?:explain it|write the code for that|make it faster)\b",
    ]

    # Device & Automation Patterns
    DEVICE_PATTERNS = [
        r"\b(device|phone|tablet|pair|connect to device|devices near|nearby device)\b"
    ]
    AUTOMATION_PATTERNS = [
        r"\b(open|launch|close|terminate|kill process|start app|notepad|calc|explorer)\b"
    ]
    STATUS_PATTERNS = [
        r"\b(status|diagnostics|system check|health|cpu usage|ram usage|how are you)\b"
    ]
    GREETING_PATTERNS = [
        r"\b(hello|hi|hey|good morning|good evening|good afternoon|greetings)\b"
    ]
    IDENTITY_PATTERNS = [
        r"\b(who are you|what are you|your name|what can you do|capabilities|jarvis)\b"
    ]
    VIDEO_PATTERNS = [
        r"\b(autoclip|auto\s+clip|extract\s+clips|extract\s+highlights|slice\s+highlights|podcast\s+clips|viral\s+clips)\b",
        r"\b(video\s+edit|edit\s+video|trim\s+video|cut\s+video|merge\s+video|concatenate\s+video)\b",
        r"\b(make\s+a\s+reel|create\s+short|vertical\s+video|sync\s+to\s+beat|highlight\s+reel)\b",
        r"\b(burn\s+subtitles|add\s+subtitles|export\s+video|video\s+timeline)\b",
        r"\b(?:play|open|watch)\s+(?:that|the)\s+video(?:\s+(?:then|than))?\b",
        r"\b(?:play\s+the\s+reel|watch\s+that\s+video)\b",
    ]
    VISION_PATTERNS = [
        r"^/(?:vision|ocr)\b",
        r"\b(image\s+classif|classif\w*\s+image|inspect\s+image|analyze\s+(?:this\s+)?image|view\s+image)\b",
        r"\b(ocr|extract\s+text|read\s+text\s+from|scan\s+text|detect\s+objects|bounding\s+box|visual\s+search)\b",
    ]
    CYBERSECURITY_PATTERNS = [
        r"\b(cwe-\d+|cve-\d+|vulnerability|cybersecurity|hardening|owasp|sql injection|xss|csrf|ssrf|zero trust|dnssec|tls 1\.3|syn flood|privilege escalation)\b",
    ]
    LEARNING_PATTERNS = [
        r"^/(?:feedback|learn|learning-log|finetune-staged|activate-model|rollback-model)\b",
        r"^(?:that(?:\s+is|\'s)\s+(?:incorrect|wrong|not\s+true|false|inaccurate))\b",
        r"^(?:no|nope|wrong|incorrect)[,:]?\s+(?:the\s+actual\s+answer\s+is|it(?:\s+is|\'s)|the\s+right\s+answer\s+is|actually|the\s+correct\s+answer\s+is)\b",
        r"^actually[,:]?\s+",
        r"\b(?:you\s+made\s+a\s+mistake|that\s+was\s+wrong|the\s+correct\s+answer\s+is)\b",
    ]
    MEMORY_QUERY_PATTERNS = [
        r"\b(?:what(?:\s+do\s+you)?\s+know\s+about\s+me|who\s+am\s+i|who\s+i\s+am|what\s+is\s+my\s+name|what\'s\s+my\s+name)\b",
        r"\b(?:tell|recall|repeat)\s+(?:me\s+)?my\s+name\b",
        r"\b(?:what\s+am\s+i\s+interested\s+in|what\s+are\s+my\s+interests|what\s+is\s+my\s+interest)\b",
        r"\b(?:what\s+am\s+i\s+studying|what\s+is\s+my\s+(?:field|major|branch|education|degree))\b",
        r"\b(?:tell\s+me\s+about\s+myself|tell\s+about\s+me|about\s+myself|about\s+me|my\s+profile)\b",
        r"\b(?:what\s+was\s+our\s+previous\s+conversation\s+about|what\s+did\s+we\s+discuss\s+(?:previously|earlier)|what\s+was\s+our\s+last\s+topic|what\s+did\s+we\s+talk\s+about)\b",
        r"\b(?:what\s+were\s+we\s+building\s+in\s+(?:our\s+)?jarvis\s+project|what\s+are\s+we\s+building\s+in\s+jarvis|what\s+was\s+the\s+last\s+error\s+we\s+fixed)\b",
        r"\b(?:continue\s+our\s+previous\s+(?:discussion|conversation|topic))\b",
    ]
    MEMORY_STORE_PATTERNS = [
        r"\bmy\s+name\s+is\s+[A-Za-z]+\b",
        r"\bcall\s+me\s+[A-Za-z]+\b",
        r"\bi\s+am\s+[A-Z][a-z]+\b",
        r"\bi\s+am\s+(?:a|an)?\s*(?:[A-Za-z\s]+)?student\b",
        r"\bi\s+am\s+(?:a|an)?\s*(?:cse|ece|it|computer\s+science)\s+student\b",
        r"\bi\s+am\s+studying\s+[^.!?]+",
        r"\b(?:i\s+am\s+)?interested\s+in\s+[^.!?]+",
        r"\bmy\s+interest\s+is\s+[^.!?]+",
        r"\bi\s+want\s+to\s+become\s+(?:a|an)?\s*[^.!?]+",
        r"\bmy\s+career\s+goal\s+is\s+[^.!?]+",
    ]

    def route(self, user_message: str, history: Optional[List[dict]] = None) -> RoutingDecision:
        """Classify input intent and assign the optimal handling engine."""
        text = user_message.strip()
        lower = text.lower()

        # Handle explicit /solve prefix: dispatch by inner subject
        if lower.startswith("/solve"):
            inner_query = re.sub(r"^/solve\s*", "", text).strip()
            inner_lower = inner_query.lower()

            # If inner is code:
            if any(lang in inner_lower for lang in ["java", "python", "c++", "program", "code"]) or "addition" in inner_lower:
                return RoutingDecision(
                    intent=IntentType.CODE_GENERATION,
                    confidence=0.98,
                    needs_agent=False,
                    agent_name="DeductiveSolver",
                    extracted_query=inner_query,
                )
            # If inner is math:
            if any(c.isdigit() for c in inner_lower) and any(w in inner_lower for w in ["multiplied by", "times", "*", "+", "-", "/", "divide", "add"]):
                return RoutingDecision(
                    intent=IntentType.MATHEMATICAL_PROBLEM,
                    confidence=0.98,
                    needs_agent=False,
                    agent_name="DeductiveSolver",
                    extracted_query=inner_query,
                )
            # If inner is riddle / rate:
            if any(w in inner_lower for w in ["machine", "bat", "ball", "produce", "riddle", "puzzle"]):
                return RoutingDecision(
                    intent=IntentType.LOGICAL_REASONING,
                    confidence=0.98,
                    needs_agent=False,
                    agent_name="DeductiveSolver",
                    extracted_query=inner_query,
                )
            return RoutingDecision(
                intent=IntentType.PROBLEM_SOLVING,
                confidence=0.95,
                needs_agent=False,
                agent_name="DeductiveSolver",
                extracted_query=inner_query,
            )

        # Handle explicit /research prefix: dispatch directly to internet research
        if lower.startswith("/research"):
            inner_query = re.sub(r"^/research\s*", "", text).strip()
            return RoutingDecision(
                intent=IntentType.INTERNET_RESEARCH,
                confidence=0.99,
                needs_agent=True,
                agent_name="ResearchAgent",
                extracted_query=inner_query or text,
            )

        # Handle explicit /vision or /ocr prefix
        if lower.startswith("/vision") or lower.startswith("/ocr"):
            inner_query = re.sub(r"^/(?:vision|ocr)\s*", "", text).strip()
            return RoutingDecision(
                intent=IntentType.COMPUTER_VISION,
                confidence=0.98,
                needs_agent=True,
                agent_name="VisionAgent",
                extracted_query=inner_query or text,
            )

        # Handle explicit /feedback, /learn, or adaptive learning commands & corrections
        if any(lower.startswith(p) for p in ["/feedback", "/learn", "/learning-log", "/finetune-staged", "/activate-model", "/rollback-model"]) or any(
            re.search(p, lower) for p in self.LEARNING_PATTERNS
        ):
            return RoutingDecision(
                intent=IntentType.ADAPTIVE_LEARNING,
                confidence=0.98,
                needs_agent=True,
                agent_name="AdaptiveLearningEngine",
                extracted_query=text,
            )

        # Multi-turn Context Evaluation:
        # Check if previous turn was asking for input preference or creative subject
        if history and len(history) >= 2:
            prev_turn = history[-1]
            prev_content = prev_turn.get("content", "").lower()
            if "what would you like the ai to create" in prev_content or "what would you like to create" in prev_content:
                return RoutingDecision(
                    intent=IntentType.IMAGE_PROMPT_GENERATION,
                    confidence=0.95,
                    needs_agent=False,
                    agent_name="CreativeEngine",
                    extracted_query=text,
                )

        # Priority 1: Persistent Personal Memory Queries
        if any(re.search(p, lower) for p in self.MEMORY_QUERY_PATTERNS):
            return RoutingDecision(
                intent=IntentType.MEMORY_QUERY,
                confidence=0.98,
                needs_agent=False,
                agent_name="MemoryAgent",
                extracted_query=text,
            )

        # Priority 2: Persistent Personal Memory Storage (Must precede code/programming recognition)
        if any(re.search(p, lower) for p in self.MEMORY_STORE_PATTERNS) and not any(
            w in lower for w in ["write a program", "give me a code", "write a code", "solve", "how to write"]
        ):
            return RoutingDecision(
                intent=IntentType.MEMORY_STORE,
                confidence=0.98,
                needs_agent=False,
                agent_name="MemoryAgent",
                extracted_query=text,
            )

        # 0. Follow-up Continuations (Proceed, Continue, Scanner, Make it darker, etc.)
        if any(re.search(p, lower) for p in self.FOLLOWUP_PATTERNS):
            return RoutingDecision(
                intent=IntentType.CONVERSATIONAL_FOLLOWUP,
                confidence=0.95,
                needs_agent=False,
                agent_name="ContextResolver",
                extracted_query=text,
                metadata={"is_followup": True},
            )

        # 1. Document Summarization
        if any(re.search(p, lower) for p in self.DOCUMENT_SUMMARY_PATTERNS) or (
            lower.startswith("summarize") and len(text.split()) > 10
        ):
            return RoutingDecision(
                intent=IntentType.PROBLEM_SOLVING,
                confidence=0.92,
                needs_agent=False,
                agent_name="DeductiveSolver",
                extracted_query=text,
                metadata={"action": "summarize"},
            )

        # 2. Image Prompt Generation
        if any(re.search(p, lower) for p in self.IMAGE_PROMPT_PATTERNS) or (
            "prompt" in lower and any(w in lower for w in ["image", "picture", "photo", "drawing", "illustration"])
        ):
            return RoutingDecision(
                intent=IntentType.IMAGE_PROMPT_GENERATION,
                confidence=0.95,
                needs_agent=False,
                agent_name="CreativeEngine",
                extracted_query=text,
            )

        # 3. Video Prompt Generation
        if any(re.search(p, lower) for p in self.VIDEO_PROMPT_PATTERNS):
            return RoutingDecision(
                intent=IntentType.VIDEO_PROMPT_GENERATION,
                confidence=0.95,
                needs_agent=False,
                agent_name="CreativeEngine",
                extracted_query=text,
            )

        # 4. Creative Generation (Stories, Concepts, Scripts, Captions)
        if any(re.search(p, lower) for p in self.CREATIVE_PATTERNS):
            return RoutingDecision(
                intent=IntentType.CREATIVE_GENERATION,
                confidence=0.92,
                needs_agent=False,
                agent_name="CreativeEngine",
                extracted_query=text,
            )

        # 4b. Knowledge Questions & Conceptual Explanations
        is_knowledge_question = (
            any(re.search(p, lower) for p in self.KNOWLEDGE_PATTERNS)
            or bool(re.search(r"\bwhat\s+is\s+(?:an?\s+|the\s+)?(ai|machine learning|java|python|operating system|database|algorithm|computer network)\b", lower))
            or ("difference between" in lower and any(w in lower for w in ["ai", "machine learning", "java", "python", "ram"]))
            or any(w in lower for w in ["simpler words", "simple words", "explain inheritance", "what is an ai", "what is ai"])
        )
        if is_knowledge_question and not any(w in lower for w in ["give me a code", "write a code", "write a program"]):
            return RoutingDecision(
                intent=IntentType.KNOWLEDGE_QUESTION,
                confidence=0.96,
                needs_agent=False,
                agent_name="KnowledgeEngine",
                extracted_query=text,
            )

        # 5. Multi-Language Code Generation & Programming
        is_code_request = (
            any(re.search(p, lower) for p in self.CODE_PATTERNS)
            or (any(lang in lower for lang in ["java", "python", "c++", "c#", "javascript", "golang", "rust", "sql", "bash"])
                and any(w in lower for w in ["code", "program", "script", "function", "syntax", "implement", "class", "addition", "reverse", "loop", "inheritance", "method"]))
            or any(w in lower for w in ["give me a code", "write a code", "write a program", "code to", "program to", "script to", "create a python program", "create a java program"])
        )
        if is_code_request:
            return RoutingDecision(
                intent=IntentType.CODE_GENERATION,
                confidence=0.96,
                needs_agent=False,
                agent_name="DeductiveSolver",
                extracted_query=text,
            )

        # 6. Logical Reasoning & Riddles (e.g. 5 machines, bat and ball)
        if any(re.search(p, lower) for p in self.LOGICAL_REASONING_PATTERNS):
            return RoutingDecision(
                intent=IntentType.LOGICAL_REASONING,
                confidence=0.95,
                needs_agent=False,
                agent_name="DeductiveSolver",
                extracted_query=text,
            )

        # 7. Mathematical Problems (Digits + Math operators or Math words)
        is_math = (
            (any(re.search(p, lower) for p in self.MATH_PATTERNS) and any(c.isdigit() for c in lower))
            or bool(re.search(r"\b\d+\s+(?:multiplied by|times|plus|minus|divided by)\s+\d+\b", lower))
        )
        if is_math:
            return RoutingDecision(
                intent=IntentType.MATHEMATICAL_PROBLEM,
                confidence=0.95,
                needs_agent=False,
                agent_name="DeductiveSolver",
                extracted_query=text,
            )

        # 8. General Factual Questions & Concept Explanations
        if any(re.search(p, lower) for p in self.GENERAL_QUESTION_PATTERNS) or "explain what ram is" in lower or "what is ram" in lower:
            return RoutingDecision(
                intent=IntentType.GENERAL_QUESTION,
                confidence=0.92,
                needs_agent=False,
                agent_name="DeductiveSolver",
                extracted_query=text,
            )

        # 9. Explicit or Live Internet Research
        if any(re.search(p, lower) for p in self.RESEARCH_PATTERNS) or "latest nvidia gpu" in lower:
            query = re.sub(r"^/research\s*", "", text).strip()
            return RoutingDecision(
                intent=IntentType.INTERNET_RESEARCH,
                confidence=0.90,
                needs_agent=True,
                agent_name="ResearchAgent",
                extracted_query=query or text,
            )

        # 10. Full-Duplex Voice Assistant Mode
        if lower.startswith(("/listen", "/voice-mode", "/voice-loop")) or any(
            w in lower for w in ["start listening", "voice mode", "conversational voice", "ambient voice"]
        ):
            return RoutingDecision(
                intent=IntentType.VOICE_MODE,
                confidence=0.98,
                needs_agent=True,
                agent_name="VoiceAssistant",
                extracted_query=text,
            )

        # 10b. Autonomous Sentinel Daemon & Maintenance
        if lower.startswith(("/sentinel", "/maintenance", "/subsystems")) or any(
            w in lower for w in ["sentinel", "maintenance", "subsystems", "proactive advisory"]
        ):
            return RoutingDecision(
                intent=IntentType.SENTINEL_OPERATION,
                confidence=0.98,
                needs_agent=True,
                agent_name="SentinelDaemon",
                extracted_query=text,
            )

        # 10b. System Diagnostics / Telemetry
        if lower.startswith(("/diagnostics", "/system-health", "/self-test")) or any(
            w in lower for w in ["diagnostics", "self-test", "system health", "health check"]
        ):
            return RoutingDecision(
                intent=IntentType.STATUS_DIAGNOSTICS,
                confidence=0.98,
                needs_agent=True,
                agent_name="SystemDiagnostics",
                extracted_query=text,
            )

        if any(re.search(p, lower) for p in self.STATUS_PATTERNS):
            return RoutingDecision(
                intent=IntentType.STATUS_DIAGNOSTICS,
                confidence=0.95,
                needs_agent=True,
                agent_name="SystemDiagnostics",
                extracted_query=text,
            )

        # 11. Device Mesh / Phone / Tablet
        if any(re.search(p, lower) for p in self.DEVICE_PATTERNS):
            return RoutingDecision(
                intent=IntentType.DEVICE_OPERATION,
                confidence=0.90,
                needs_agent=True,
                agent_name="DeviceManagementAgent",
                extracted_query=text,
            )

        # 11. Multi-Step Autonomous Mission Orchestration
        if lower.startswith("/orchestrate") or lower.startswith("/swarm") or (
            any(w in lower for w in ["orchestrate", "multi-step", "coordinate agents", "swarm task"])
            and len(text.split()) >= 3
        ):
            query = re.sub(r"^/(orchestrate|swarm)\s*", "", text).strip()
            return RoutingDecision(
                intent=IntentType.MULTI_STEP_TASK,
                confidence=0.95,
                needs_agent=True,
                agent_name="HierarchicalOrchestrator",
                extracted_query=query or text,
            )

        # 12. Safe Computer Automation
        if any(re.search(p, lower) for p in self.AUTOMATION_PATTERNS) and any(
            app in lower for app in ["notepad", "calc", "calculator", "vscode", "code", "browser", "app"]
        ):
            return RoutingDecision(
                intent=IntentType.DESKTOP_AUTOMATION,
                confidence=0.85,
                needs_agent=True,
                agent_name="ComputerAutomationAgent",
                extracted_query=text,
            )

        # 13. Video Editing Automation
        if lower.startswith("/video") or lower.startswith("/reel") or lower.startswith("/short") or lower.startswith("/autoclip") or any(
            re.search(p, lower) for p in self.VIDEO_PATTERNS
        ):
            query = re.sub(r"^/(video|reel|short|autoclip)\s*", "", text).strip()
            return RoutingDecision(
                intent=IntentType.VIDEO_EDITING,
                confidence=0.92,
                needs_agent=True,
                agent_name="VideoEditingAgent",
                extracted_query=query or text,
            )

        # 14. Computer Vision
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

        # 15. Cybersecurity
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

        # 16. System Health & Diagnostics
        if lower.startswith("/diagnostics") or lower.startswith("/system-health") or lower.startswith("/self-test") or any(
            w in lower for w in ["system health", "diagnostics", "self test", "subsystem health", "run diagnostics"]
        ):
            return RoutingDecision(
                intent=IntentType.STATUS_DIAGNOSTICS,
                confidence=0.95,
                needs_agent=False,
                agent_name="SystemDiagnostics",
                extracted_query=text,
            )

        # 17. Greetings
        if any(re.search(p, lower) for p in self.GREETING_PATTERNS) and len(text.split()) <= 4:
            return RoutingDecision(
                intent=IntentType.GREETING,
                confidence=0.90,
                needs_agent=False,
                agent_name=None,
                extracted_query=text,
            )

        # 19. Identity
        if any(re.search(p, lower) for p in self.IDENTITY_PATTERNS):
            return RoutingDecision(
                intent=IntentType.IDENTITY,
                confidence=0.92,
                needs_agent=False,
                agent_name=None,
                extracted_query=text,
            )

        # 20. General Conversational Interaction
        return RoutingDecision(
            intent=IntentType.CONVERSATIONAL_INTERACTION,
            confidence=0.80,
            needs_agent=False,
            agent_name="ConversationalCortex",
            extracted_query=text,
        )


__all__ = ["IntentRouter", "IntentType", "RoutingDecision"]
