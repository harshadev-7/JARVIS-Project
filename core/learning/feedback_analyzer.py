"""J.A.R.V.I.S. Feedback and Error Analyzer.

Detects user corrections, parses corrective intent, and performs
root-cause diagnosis of erroneous or sub-optimal responses.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class DiagnosticResult:
    """The root-cause diagnosis of an incorrect response."""
    root_cause: str  # "KNOWLEDGE_GAP", "FACTUAL_INACCURACY", "INTENT_MISCLASSIFICATION", "LOGICAL_REASONING_ERROR", "PREFERENCE_UPDATE"
    confidence: float
    explanation: str
    recommended_action: str


class FeedbackAnalyzer:
    """Detects, parses, and diagnoses user corrections and feedback."""

    FEEDBACK_PREFIX_PATTERNS = [
        r"^/feedback\b",
        r"^/learn\b",
        r"^(?:that(?:\s+is|\'s)\s+(?:incorrect|wrong|not\s+true|false|inaccurate))\b",
        r"^(?:no|nope|wrong|incorrect)[,:]?\s+(?:the\s+actual\s+answer\s+is|it(?:\s+is|\'s)|the\s+right\s+answer\s+is|actually|the\s+correct\s+answer\s+is)\b",
        r"^actually[,:]?\s+",
        r"\b(?:you\s+made\s+a\s+mistake|that\s+was\s+wrong|correct\s+answer\s+is|the\s+right\s+answer\s+is)\b",
        r"\b(?:remember\s+that|note\s+that)\s+(?:in\s+fact|actually)\b",
    ]

    def is_feedback(self, text: str) -> bool:
        """Check if message represents user corrective feedback."""
        cleaned = text.strip().lower()
        if cleaned.startswith("/feedback") or cleaned.startswith("/learn"):
            return True
        return any(bool(re.search(p, cleaned, re.IGNORECASE)) for p in self.FEEDBACK_PREFIX_PATTERNS)

    def extract_correction(
        self, text: str, previous_turns: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """Extract original query, original response, and clean user correction."""
        raw = text.strip()
        lower = raw.lower()

        # Handle explicit /feedback or /learn commands
        if lower.startswith("/feedback"):
            clean_correction = re.sub(r"^/feedback\s*", "", raw).strip()
            command_type = "feedback"
        elif lower.startswith("/learn"):
            clean_correction = re.sub(r"^/learn\s*", "", raw).strip()
            command_type = "learn"
        else:
            command_type = "natural_correction"
            # Strip common correction lead-ins
            clean_correction = re.sub(
                r"^(?:that(?:\s+is|\'s)\s+(?:incorrect|wrong|not\s+true|false|inaccurate)[,.:;]?\s*|"
                r"no[,:]?\s*(?:the\s+actual\s+answer\s+is|it(?:\s+is|\'s)|actually|the\s+right\s+answer\s+is)[,.:;]?\s*|"
                r"actually[,:]?\s*|"
                r"you\s+made\s+a\s+mistake[,.:;]?\s*|"
                r"the\s+correct\s+answer\s+is[,.:;]?\s*)",
                "",
                raw,
                flags=re.IGNORECASE,
            ).strip()

        # Extract previous context from dialogue history
        orig_query = ""
        orig_resp = ""
        if previous_turns:
            # Find the most recent assistant response and the user turn before it
            assistant_turns = [t for t in previous_turns if t.get("role") == "assistant"]
            user_turns = [t for t in previous_turns if t.get("role") == "user"]
            if assistant_turns:
                orig_resp = assistant_turns[-1].get("content", "")
            if len(user_turns) >= 2:
                # The user turn prior to current feedback turn
                orig_query = user_turns[-2].get("content", "")
            elif len(user_turns) == 1 and not (lower.startswith("/feedback") or lower.startswith("/learn")):
                orig_query = user_turns[0].get("content", "")

        return {
            "command_type": command_type,
            "user_correction": clean_correction or raw,
            "original_query": orig_query or "Direct user instruction",
            "original_response": orig_resp or "No previous response recorded",
        }

    def diagnose_root_cause(
        self,
        original_query: str,
        original_response: str,
        user_correction: str,
        agent_used: Optional[str] = None,
    ) -> DiagnosticResult:
        """Perform automated root-cause diagnosis of why the response required correction."""
        q_lower = original_query.lower()
        r_lower = original_response.lower()
        c_lower = user_correction.lower()

        # 1. Profile / Preference updates
        if any(w in c_lower for w in ["my name is", "i am a", "i work as", "call me", "i prefer", "my college"]):
            return DiagnosticResult(
                root_cause="PREFERENCE_UPDATE",
                confidence=0.95,
                explanation="User provided updated personal profile or preference details.",
                recommended_action="Update user preference record in persistent SQLite memory.",
            )

        # 2. Mathematical / Arithmetic / Calculation errors
        if any(c.isdigit() for c in q_lower) and any(w in q_lower for w in ["calculate", "+", "-", "*", "/", "multiply", "divide", "add", "sum"]):
            return DiagnosticResult(
                root_cause="LOGICAL_REASONING_ERROR",
                confidence=0.92,
                explanation="Discrepancy detected in calculation or deterministic mathematical step.",
                recommended_action="Update algorithmic reasoning steps and log mistake to prevent repetition.",
            )

        # 3. Intent misclassification (e.g. asked for code but got research, or asked for explanation but got code)
        if ("code" in q_lower or "program" in q_lower) and "research report" in r_lower:
            return DiagnosticResult(
                root_cause="INTENT_MISCLASSIFICATION",
                confidence=0.90,
                explanation="Request was misclassified by intent router into internet research instead of code generation.",
                recommended_action="Adjust router priority pattern and add regression test case.",
            )

        # 4. Outdated / Conflicting fact (factual inaccuracy)
        if any(w in c_lower for w in ["not", "no longer", "replaced by", "changed to", "incorrect year"]):
            return DiagnosticResult(
                root_cause="FACTUAL_INACCURACY",
                confidence=0.88,
                explanation="Response contained outdated, inaccurate, or superseded factual data.",
                recommended_action="Supersede old knowledge item with verified user correction in knowledge base.",
            )

        # 5. Default: Knowledge Gap
        return DiagnosticResult(
            root_cause="KNOWLEDGE_GAP",
            confidence=0.85,
            explanation="The requested information was missing or insufficiently detailed in offline knowledge.",
            recommended_action="Index verified correction into SQLite knowledge base and stage for instruction tuning.",
        )


feedback_analyzer = FeedbackAnalyzer()
__all__ = ["DiagnosticResult", "FeedbackAnalyzer", "feedback_analyzer"]
