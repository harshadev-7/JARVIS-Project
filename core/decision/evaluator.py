"""
JARVIS AI — Decision Evaluator
===============================
Develops multiple candidate solutions, evaluates them across safety,
feasibility, cost, and efficiency, and selects the optimal path.
"""

from typing import List, Optional
from pydantic import BaseModel, Field
from config.autonomy_levels import check_permission, get_autonomy_config


class SolutionApproach(BaseModel):
    """A candidate solution strategy."""
    id: str
    name: str
    description: str
    pros: List[str] = Field(default_factory=list)
    cons: List[str] = Field(default_factory=list)
    safety_score: float = Field(default=1.0, ge=0.0, le=1.0, description="1.0 is safest")
    feasibility_score: float = Field(default=1.0, ge=0.0, le=1.0, description="1.0 is most feasible")
    speed_score: float = Field(default=1.0, ge=0.0, le=1.0, description="1.0 is fastest")
    resource_cost_score: float = Field(default=1.0, ge=0.0, le=1.0, description="1.0 is lowest cost")
    required_autonomy_level: int = Field(default=2, ge=1, le=5)
    composite_score: float = 0.0


class EvaluationResult(BaseModel):
    """The result of evaluating multiple solution approaches."""
    objective: str
    approaches: List[SolutionApproach]
    selected_approach: Optional[SolutionApproach] = None
    selection_rationale: str = ""
    is_authorized: bool = True
    authorization_note: str = ""


class DecisionEvaluator:
    """Evaluates multiple approaches and selects the optimal strategy."""

    def evaluate_approaches(
        self,
        objective: str,
        approaches: List[SolutionApproach],
        current_autonomy_level: int = 2,
    ) -> EvaluationResult:
        """
        Score each candidate approach and select the optimal authorized solution.
        """
        if not approaches:
            return EvaluationResult(
                objective=objective,
                approaches=[],
                selected_approach=None,
                selection_rationale="No approaches provided for evaluation.",
                is_authorized=False,
                authorization_note="No executable strategy available.",
            )

        scored_approaches: List[SolutionApproach] = []
        for app in approaches:
            # Composite scoring formula: Safety (35%), Feasibility (30%), Speed (15%), Resource Cost (20%)
            comp = (
                (app.safety_score * 0.35)
                + (app.feasibility_score * 0.30)
                + (app.speed_score * 0.15)
                + (app.resource_cost_score * 0.20)
            )
            # Penalty if approach exceeds current autonomy level
            if app.required_autonomy_level > current_autonomy_level:
                comp *= 0.5  # Significant penalty for unauthorized approach

            app.composite_score = round(comp, 4)
            scored_approaches.append(app)

        # Sort descending by composite score
        scored_approaches.sort(key=lambda x: x.composite_score, reverse=True)
        top = scored_approaches[0]

        # Check authorization
        authorized = top.required_autonomy_level <= current_autonomy_level
        if authorized:
            auth_note = f"Approach '{top.name}' is within current Autonomy Level {current_autonomy_level}."
        else:
            config = get_autonomy_config(current_autonomy_level)
            auth_note = (
                f"Top approach '{top.name}' requires Autonomy Level {top.required_autonomy_level}, "
                f"which exceeds current level {current_autonomy_level} ({config.name}). Requires user authorization."
            )

        rationale = (
            f"Selected '{top.name}' with composite score {top.composite_score:.2f} "
            f"(Safety: {top.safety_score*100:.0f}%, Feasibility: {top.feasibility_score*100:.0f}%). "
            f"{auth_note}"
        )

        return EvaluationResult(
            objective=objective,
            approaches=scored_approaches,
            selected_approach=top,
            selection_rationale=rationale,
            is_authorized=authorized,
            authorization_note=auth_note,
        )


decision_evaluator = DecisionEvaluator()
