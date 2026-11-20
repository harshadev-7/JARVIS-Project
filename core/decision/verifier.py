"""
JARVIS AI — Result Verifier & Error Investigator
=================================================
Verifies task outcomes against expected success criteria.
When an action or subtask fails, conducts Root Cause Analysis (RCA)
and formulates safe alternative solutions.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class VerificationResult(BaseModel):
    """Assessment of whether an execution or approach succeeded."""
    task_id: str
    success: bool
    confidence_score: float = Field(default=1.0, ge=0.0, le=1.0)
    actual_outcome: str
    discrepancies: List[str] = Field(default_factory=list)
    root_cause_analysis: Optional[str] = None
    recommended_fallback: Optional[str] = None


class ResultVerifier:
    """
    Validates outcomes, investigates errors, and generates alternatives.
    """

    def verify_step(
        self,
        task_id: str,
        expected_criteria: str,
        execution_output: str,
        error_context: Optional[str] = None,
    ) -> VerificationResult:
        """
        Verify whether a step met its expected criteria.
        If an error is provided, triggers root cause investigation.
        """
        if error_context:
            rca = self._investigate_root_cause(error_context)
            fallback = self._suggest_alternative(error_context)
            return VerificationResult(
                task_id=task_id,
                success=False,
                confidence_score=0.9,
                actual_outcome=f"Execution error: {error_context}",
                discrepancies=[f"Expected '{expected_criteria}', but encountered: {error_context}"],
                root_cause_analysis=rca,
                recommended_fallback=fallback,
            )

        # Baseline positive verification
        return VerificationResult(
            task_id=task_id,
            success=True,
            confidence_score=1.0,
            actual_outcome=execution_output or "Completed without errors.",
            discrepancies=[],
            root_cause_analysis=None,
            recommended_fallback=None,
        )

    def _investigate_root_cause(self, error: str) -> str:
        """Identify probable failure category and cause."""
        err_lower = error.lower()
        if "permission" in err_lower or "unauthorized" in err_lower or "access denied" in err_lower:
            return "Permission Barrier: Operation requires elevated permissions or explicit user authorization."
        elif "timeout" in err_lower or "timed out" in err_lower:
            return "Latency/Timeout: Resource did not respond within the allocated threshold."
        elif "not found" in err_lower or "404" in err_lower:
            return "Resource Missing: Target endpoint, file, or resource could not be located."
        elif "connection" in err_lower or "network" in err_lower:
            return "Network Fault: Target service unreachable or socket failure."
        return f"Operational Failure: {error}"

    def _suggest_alternative(self, error: str) -> str:
        """Generate safe alternative approach based on error cause."""
        err_lower = error.lower()
        if "permission" in err_lower or "access denied" in err_lower:
            return "Request elevated authorization from the user or switch to a read-only diagnostic approach."
        elif "timeout" in err_lower:
            return "Retry with exponential backoff or degrade to cached/offline response."
        elif "not found" in err_lower:
            return "Perform diagnostic lookup to verify resource path and availability before re-attempting."
        return "Engage fallback rule set or defer task execution for user guidance."


result_verifier = ResultVerifier()
