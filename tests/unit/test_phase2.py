"""
JARVIS AI — Phase 2 Test Suite
================================
Tests for the Advanced Reasoning & Problem-Solving Engine.
Covers:
  - Goal Analysis (Facts vs Assumptions vs Uncertainties)
  - Task Decomposition (Subtasks, Dependencies, Execution Plans)
  - Decision Evaluation (Candidate scoring & Autonomy level gating)
  - Result Verification (Outcome validation, RCA, Fallback generation)
  - 10-Step Reasoning Engine Cycle
  - FastAPI Reasoning Endpoints
"""

import pytest
from fastapi.testclient import TestClient

from core.planner.goal_analyzer import goal_analyzer
from core.planner.task_decomposer import task_decomposer, TaskPriority, TaskStatus
from core.decision.evaluator import decision_evaluator, SolutionApproach
from core.decision.verifier import result_verifier
from core.planner.reasoning_engine import reasoning_engine
from api.app import app


# ── 1. Goal Analyzer Tests ──────────────────────────────────────────────────

def test_goal_analyzer_heuristic():
    """Goal analyzer should identify constraints and complexity."""
    prompt = "Create a database backup without stopping the active server process."
    analysis = goal_analyzer.analyze_heuristic(prompt)

    assert analysis.primary_goal is not None
    assert len(analysis.constraints) > 0  # detected "without"
    assert len(analysis.facts) > 0
    assert analysis.complexity_score >= 1


def test_goal_analyzer_uncertainties():
    """Questions should trigger uncertainty detection."""
    prompt = "What is the optimal caching layer for our API?"
    analysis = goal_analyzer.analyze_heuristic(prompt)

    assert any("Inquiry" in u for u in analysis.uncertainties)


# ── 2. Task Decomposer Tests ───────────────────────────────────────────────

def test_task_decomposer_linear_plan():
    """Decomposer should create ordered subtasks with dependency tracking."""
    steps = [
        {"id": "step-1", "title": "Check system load", "description": "Verify CPU/RAM"},
        {"id": "step-2", "title": "Execute indexing", "description": "Rebuild indices"},
        {"id": "step-3", "title": "Audit log", "description": "Write completion status"},
    ]
    plan = task_decomposer.create_linear_plan(goal="Database optimization", steps=steps)

    assert plan.total_steps == 3
    assert plan.subtasks[0].id == "step-1"
    assert plan.subtasks[1].dependencies == ["step-1"]
    assert plan.subtasks[2].dependencies == ["step-2"]
    assert plan.estimated_total_duration_sec > 0


# ── 3. Decision Evaluator Tests ────────────────────────────────────────────

def test_decision_evaluator_scoring():
    """Evaluator should rank candidate approaches based on composite score."""
    app_safe = SolutionApproach(
        id="app-1",
        name="Conservative",
        description="Safe path",
        safety_score=1.0,
        feasibility_score=0.9,
        speed_score=0.6,
        resource_cost_score=0.9,
        required_autonomy_level=2,
    )
    app_risky = SolutionApproach(
        id="app-2",
        name="Aggressive",
        description="High risk path",
        safety_score=0.2,
        feasibility_score=0.5,
        speed_score=1.0,
        resource_cost_score=0.3,
        required_autonomy_level=2,
    )

    result = decision_evaluator.evaluate_approaches(
        objective="Deploy update",
        approaches=[app_safe, app_risky],
        current_autonomy_level=2,
    )

    assert result.selected_approach is not None
    assert result.selected_approach.name == "Conservative"
    assert result.is_authorized is True


def test_decision_evaluator_autonomy_penalty():
    """Approaches exceeding current autonomy level should receive penalty."""
    app_autonomous = SolutionApproach(
        id="app-auto",
        name="Fully Autonomous Overhaul",
        description="Auto modify production",
        safety_score=0.9,
        feasibility_score=0.9,
        speed_score=0.9,
        resource_cost_score=0.9,
        required_autonomy_level=5,
    )

    result = decision_evaluator.evaluate_approaches(
        objective="System overhaul",
        approaches=[app_autonomous],
        current_autonomy_level=2,  # Level 2 cannot run Level 5 without authorization
    )

    assert result.is_authorized is False
    assert "exceeds" in result.authorization_note


# ── 4. Result Verifier Tests ────────────────────────────────────────────────

def test_result_verifier_success():
    """Clean execution should verify successfully."""
    v = result_verifier.verify_step(
        task_id="step-01",
        expected_criteria="Process returned exit code 0",
        execution_output="Success: Database backed up to s3://backup",
    )
    assert v.success is True
    assert v.root_cause_analysis is None


def test_result_verifier_rca_and_fallback():
    """Errors should trigger root cause analysis and safe fallbacks."""
    v = result_verifier.verify_step(
        task_id="step-02",
        expected_criteria="Write to protected system directory",
        execution_output="",
        error_context="Permission denied: Access to C:\\Windows\\System32 is restricted",
    )
    assert v.success is False
    assert "Permission" in v.root_cause_analysis
    assert v.recommended_fallback is not None


# ── 5. Reasoning Engine 10-Step Cycle Tests ─────────────────────────────────

def test_reasoning_engine_10_step_cycle():
    """Reasoning engine must execute all 10 steps and return comprehensive result."""
    result = reasoning_engine.solve(
        problem_description="Diagnose high server memory utilization and optimize Redis cache",
        autonomy_level=3,
    )

    assert result.step_1_objective_understanding is not None
    assert result.step_2_situation_assessment != ""
    assert "facts" in result.step_3_collected_information
    assert len(result.step_4_available_resources) > 0
    assert len(result.step_5_possible_approaches) >= 2
    assert result.step_6_evaluation.selected_approach is not None
    assert result.step_7_selected_solution.id == result.step_6_evaluation.selected_approach.id
    assert len(result.step_8_execution_plan.subtasks) > 0
    assert result.step_9_verification.success is True
    assert "JARVIS Problem-Solving Assessment" in result.step_10_final_report


# ── 6. FastAPI Reasoning Routes Tests ───────────────────────────────────────

@pytest.fixture
def client():
    return TestClient(app)


def test_api_reasoning_solve(client):
    """POST /reasoning/solve should return complete 10-step reasoning object."""
    payload = {
        "problem": "Investigate database query bottlenecks",
        "autonomy_level": 2,
    }
    response = client.post("/reasoning/solve", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "step_1_objective_understanding" in data
    assert "step_8_execution_plan" in data
    assert "step_10_final_report" in data


def test_api_reasoning_analyze(client):
    """POST /reasoning/analyze should classify facts and constraints."""
    payload = {"problem": "Secure API endpoints without breaking backwards compatibility"}
    response = client.post("/reasoning/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "primary_goal" in data
    assert "constraints" in data


def test_api_reasoning_plan(client):
    """POST /reasoning/plan should return decomposed task plan."""
    payload = {"goal": "Migrate database schema"}
    response = client.post("/reasoning/plan", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["goal"] == "Migrate database schema"
    assert len(data["subtasks"]) >= 3
