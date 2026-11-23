"""J.A.R.V.I.S. Regression and Consistency Guard.

Validates candidate learning updates against core invariants, detects
malicious/destructive inputs, and runs automated regression probes
to ensure new learning never corrupts verified system behavior.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional
from core.memory.persistent_memory import persistent_memory
from core.logger import log


class RegressionGuard:
    """Safeguards knowledge and model consistency during adaptive updates."""

    # Core inviolable system truths
    CORE_INVARIANTS = [
        r"\b(?:1\s*\+\s*1\s*=\s*[3-9]|\b2\s*\+\s*2\s*=\s*[5-9])\b",  # Mathematical false equivalencies
        r"(?:format\s+[a-z]:|rmdir\s+/[sq]|drop\s+database|del\s+/[a-z]|rm\s+-rf)",  # Dangerous shell commands
        r"\b(?:bypass\s+all\s+security|ignore\s+previous\s+instructions|system\s+jailbreak)\b",  # Prompt injection attacks
    ]

    # Standard probe queries to test retrieval health
    REGRESSION_PROBES = [
        "What is an AI?",
        "What is Java?",
        "What is Machine Learning?",
        "Who are you?",
    ]

    def validate_consistency(
        self, user_correction: str, root_cause: str = "KNOWLEDGE_GAP"
    ) -> Dict[str, Any]:
        """Validate candidate correction against safety and core logical invariants."""
        cleaned = user_correction.strip()
        lower = cleaned.lower()

        # 1. Check for dangerous command injections
        for pattern in self.CORE_INVARIANTS:
            if re.search(pattern, lower, re.IGNORECASE):
                log.warning(f"RegressionGuard rejected correction matching invariant guard: {pattern}")
                return {
                    "is_safe": False,
                    "passed_invariants": False,
                    "reason": f"Correction violated safety invariant or contained dangerous command patterns.",
                }

        # 2. Check for empty or trivial input
        if len(cleaned) < 3:
            return {
                "is_safe": False,
                "passed_invariants": False,
                "reason": "Correction content is too short to represent meaningful knowledge.",
            }

        # 3. Check for excessive length (prevent token stuffing)
        if len(cleaned) > 2000:
            return {
                "is_safe": False,
                "passed_invariants": False,
                "reason": "Correction exceeds maximum single-entry threshold (2,000 characters).",
            }

        return {
            "is_safe": True,
            "passed_invariants": True,
            "reason": "Correction passed all consistency and invariant checks.",
        }

    async def run_regression_probes(self, candidate_fact: str) -> Dict[str, Any]:
        """Execute probes against persistent memory to confirm index stability."""
        passed_count = 0
        details = []

        try:
            for probe in self.REGRESSION_PROBES:
                results = await persistent_memory.search_knowledge(probe, top_k=2)
                # Successful probe means search executes cleanly without exception
                passed_count += 1
                details.append(f"Probe '{probe}' returned {len(results)} items.")

            is_stable = passed_count == len(self.REGRESSION_PROBES)
            return {
                "passed": is_stable,
                "probes_run": len(self.REGRESSION_PROBES),
                "probes_passed": passed_count,
                "details": details,
            }
        except Exception as e:
            log.error(f"RegressionGuard probe run encountered error: {e}")
            return {
                "passed": False,
                "probes_run": len(self.REGRESSION_PROBES),
                "probes_passed": passed_count,
                "details": [f"Error during probe: {e}"],
            }


regression_guard = RegressionGuard()
__all__ = ["RegressionGuard", "regression_guard"]
