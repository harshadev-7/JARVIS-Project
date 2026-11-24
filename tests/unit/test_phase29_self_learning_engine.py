"""Unit tests for J.A.R.V.I.S. Phase 29 — Self-Learning Engine.
=============================================================
Tests the controlled self-learning capabilities:
1. Learning from approved user feedback
2. Diagnostic root-cause analysis and user approval gating
3. Task result post-mortem analysis and automatic lesson extraction
4. Mistake and correction recording with proactive mistake prevention
5. Local offline knowledge database updates (without modifying neural weights)
6. Isolated candidate training example generation in staging buffers
7. Quantitative evaluation of learning progress and velocity
"""

import asyncio
import json
import pytest
from pathlib import Path

from core.self_evolution.self_learning_engine import (
    SelfLearningEngine,
    MistakeCorrectionRecord,
    TaskAnalysisResult,
    CandidateTrainingExample,
    LearningProgressReport,
)
from core.knowledge.knowledge_engine import knowledge_engine
from core.memory.persistent_memory import persistent_memory


@pytest.fixture
def temp_engine(tmp_path):
    """Provides an isolated SelfLearningEngine instance with dedicated temp paths."""
    evo_dir = tmp_path / "evolution"
    stage_dir = tmp_path / "staging"
    engine = SelfLearningEngine(evolution_dir=evo_dir, staging_dir=stage_dir)
    return engine


@pytest.mark.asyncio
async def test_self_learning_initialization(temp_engine):
    """Verifies that SelfLearningEngine creates directories and initializes persistence."""
    await temp_engine.initialize()
    assert temp_engine.evolution_dir.exists()
    assert temp_engine.staging_dir.exists()
    assert temp_engine._initialized is True


@pytest.mark.asyncio
async def test_process_user_feedback_pending_and_approval(temp_engine):
    """
    Verifies that natural user feedback is parsed, diagnosed, held in pending_approval
    until authorized, and then committed to knowledge and candidate training pairs.
    """
    await temp_engine.initialize()

    feedback_text = "Actually, the capital of Australia is Canberra, not Sydney."
    res = await temp_engine.process_user_feedback(
        user_message=feedback_text,
        session_id="test_session_1",
        auto_approve=False,
    )

    assert res["success"] is True
    assert res["status"] == "pending_approval"
    assert res["needs_approval"] is True
    assert "prompt_text" in res
    assert "Canberra" in res["user_correction"]
    record_id = res["record_id"]

    # Now simulate user granting explicit approval
    commit_res = await temp_engine.apply_approved_feedback(record_id)
    assert commit_res["success"] is True
    assert commit_res["status"] == "committed"
    assert "knowledge_id" in commit_res
    assert "mistake_record_id" in commit_res
    assert "candidate_example_id" in commit_res

    # Check that candidate training pair exists in staging file
    assert temp_engine.candidate_file.exists()
    staged_count = temp_engine.count_candidate_examples()
    assert staged_count >= 1


@pytest.mark.asyncio
async def test_process_user_feedback_auto_approve(temp_engine):
    """Verifies instant application of feedback when auto_approve=True."""
    await temp_engine.initialize()

    res = await temp_engine.process_user_feedback(
        user_message="/learn Python was created by Guido van Rossum in 1991",
        session_id="test_session_auto",
        auto_approve=True,
    )

    assert res["success"] is True
    assert res["status"] == "committed"
    assert res["auto_approved"] is True
    assert res["record_id"] > 0

    # Verify local knowledge search recalls it immediately without cloud lookups
    results = await persistent_memory.search_knowledge("Guido van Rossum")
    assert len(results) > 0
    assert any("Guido van Rossum" in r["text"] for r in results)


@pytest.mark.asyncio
async def test_task_result_analysis_and_mistake_recording(temp_engine):
    """Verifies post-mortem failure analysis extracts lessons and logs mistakes."""
    await temp_engine.initialize()

    # 1. Analyze a failed task execution
    task_id = "task_video_export_99"
    analysis = await temp_engine.analyze_task_result(
        task_id=task_id,
        goal="Export vertical 9:16 reel to C:/output/reel.mp4",
        status="failed",
        duration_sec=3.4,
        error_summary="FileNotFoundError: Missing background audio track in assets",
        steps_count=4,
    )

    assert isinstance(analysis, TaskAnalysisResult)
    assert analysis.error_detected is True
    assert analysis.root_cause == "RESOURCE_NOT_FOUND"
    assert analysis.should_record_mistake is True
    assert "background audio" in analysis.lesson_learned

    # Verify mistake is written to mistakes_log.jsonl
    assert temp_engine.mistakes_file.exists()
    with open(temp_engine.mistakes_file, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    assert len(records) > 0
    assert any(r.get("task_id") == task_id for r in records)


@pytest.mark.asyncio
async def test_proactive_mistake_prevention_check(temp_engine):
    """Verifies that proactive checking warns against repeating past recorded mistakes."""
    await temp_engine.initialize()

    # Record a mistake
    record = await temp_engine.record_mistake_and_correction(
        mistake_description="Database migration crashed when executing raw DROP TABLE without backup",
        verified_correction="Always create a verified SQLite backup snapshot before executing schema modifications",
        root_cause="EXECUTION_FAILURE",
        context="migration_unit_test",
    )
    assert record.record_id in temp_engine._in_memory_mistakes

    # Check proactive warning against a similar planned operation
    warnings = temp_engine.check_mistake_prevention("Execute database migration DROP TABLE on schema")
    assert len(warnings) > 0
    assert any("DROP TABLE" in w["text"] for w in warnings)

    # Avoidance count should have incremented
    assert temp_engine._in_memory_mistakes[record.record_id].avoidance_count >= 1


@pytest.mark.asyncio
async def test_candidate_training_example_generation(temp_engine):
    """
    Verifies that instruction-tuning pairs are formatted and appended to candidate
    staging buffers without touching or training active neural network weights.
    """
    await temp_engine.initialize()

    example = temp_engine.generate_candidate_training_example(
        prompt="User: What is the primary difference between synchronous and asynchronous I/O?\nJARVIS:",
        completion=" Synchronous I/O blocks the calling thread until completion, whereas asynchronous I/O delegates execution and notifies via callbacks or event loops.",
        category="reasoning",
        source="unit_test",
    )

    assert isinstance(example, CandidateTrainingExample)
    assert example.is_staged is True
    assert example.example_id.startswith("trn_")

    # Verify file content
    with open(temp_engine.candidate_file, "r", encoding="utf-8") as f:
        lines = [json.loads(l) for l in f if l.strip()]
    assert len(lines) >= 1
    last_entry = lines[-1]
    assert "Synchronous I/O" in last_entry["completion"]
    assert last_entry["category"] == "reasoning"


@pytest.mark.asyncio
async def test_evaluate_learning_progress_metrics(temp_engine):
    """Verifies that evaluate_learning_progress computes comprehensive quantitative scores."""
    await temp_engine.initialize()

    # Add a feedback item and a mistake
    await temp_engine.process_user_feedback(
        user_message="Actually, HTTP status 418 means I'm a teapot.",
        session_id="metrics_session",
        auto_approve=True,
    )
    await temp_engine.record_mistake_and_correction(
        mistake_description="Assumed port 8000 was free without probing",
        verified_correction="Probe port availability before attempting socket bind",
        root_cause="EXECUTION_FAILURE",
    )

    report = await temp_engine.evaluate_learning_progress()

    assert isinstance(report, LearningProgressReport)
    assert report.total_feedback_processed >= 1
    assert report.approved_feedback_count >= 1
    assert report.mistakes_recorded >= 1
    assert 0.0 <= report.mistake_repeat_rate <= 1.0
    assert 0.0 <= report.overall_learning_score <= 100.0
    assert "JARVIS Self-Learning Score:" in report.summary

    # Verify markdown generation
    md = temp_engine.get_summary_report_markdown(report)
    assert "J.A.R.V.I.S. Self-Learning Engine Status (Phase 29)" in md
    assert "Overall Learning Score" in md
