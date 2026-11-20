"""
JARVIS AI — Goal Analyzer
===========================
Analyzes user requests to extract goals, constraints, facts, assumptions,
and uncertain information. Distinguishes what is definitively known from
what is assumed or unknown.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class GoalAnalysis(BaseModel):
    """Structured breakdown of an objective."""
    raw_prompt: str
    primary_goal: str = Field(..., description="The main objective identified")
    sub_goals: List[str] = Field(default_factory=list, description="Secondary or supporting objectives")
    constraints: List[str] = Field(default_factory=list, description="Explicit or implicit limitations")
    facts: List[str] = Field(default_factory=list, description="Verified facts present in the request")
    assumptions: List[str] = Field(default_factory=list, description="Assumptions made to proceed")
    uncertainties: List[str] = Field(default_factory=list, description="Unknowns that need clarification or discovery")
    required_resources: List[str] = Field(default_factory=list, description="Tools, APIs, or data needed")
    missing_information: List[str] = Field(default_factory=list, description="Critical information missing to solve cleanly")
    complexity_score: int = Field(default=1, ge=1, le=10, description="Estimated complexity from 1 (trivial) to 10 (highly complex)")


class GoalAnalyzer:
    """
    Analyzes intent and extracts core goals, constraints, facts, and uncertainties.
    """

    def analyze_heuristic(self, user_input: str) -> GoalAnalysis:
        """
        Fast heuristic analyzer that classifies intent when offline or before LLM synthesis.
        """
        cleaned = user_input.strip()
        words = cleaned.split()

        # Basic complexity heuristic
        complexity = min(10, max(1, len(words) // 8 + (1 if "?" in cleaned else 0)))

        facts = []
        assumptions = []
        uncertainties = []
        constraints = []
        missing = []

        # Identify explicit constraints or parameters
        lower = cleaned.lower()
        if "without" in lower or "do not" in lower or "never" in lower or "only" in lower:
            constraints.append("Negative constraints detected in user instructions.")

        if "?" in cleaned:
            uncertainties.append("Inquiry or ambiguity detected in prompt.")

        facts.append(f"User submitted query with {len(words)} words.")

        return GoalAnalysis(
            raw_prompt=cleaned,
            primary_goal=cleaned if len(cleaned) < 120 else cleaned[:117] + "...",
            sub_goals=[],
            constraints=constraints,
            facts=facts,
            assumptions=assumptions,
            uncertainties=uncertainties,
            required_resources=["LLM Intelligence Engine"],
            missing_information=missing,
            complexity_score=complexity,
        )

    def parse_llm_analysis(self, raw_dict: Dict[str, Any], fallback_prompt: str) -> GoalAnalysis:
        """
        Validate and construct a GoalAnalysis object from an LLM response dictionary.
        """
        return GoalAnalysis(
            raw_prompt=raw_dict.get("raw_prompt", fallback_prompt),
            primary_goal=raw_dict.get("primary_goal", fallback_prompt),
            sub_goals=raw_dict.get("sub_goals", []),
            constraints=raw_dict.get("constraints", []),
            facts=raw_dict.get("facts", []),
            assumptions=raw_dict.get("assumptions", []),
            uncertainties=raw_dict.get("uncertainties", []),
            required_resources=raw_dict.get("required_resources", []),
            missing_information=raw_dict.get("missing_information", []),
            complexity_score=raw_dict.get("complexity_score", 1),
        )


goal_analyzer = GoalAnalyzer()
