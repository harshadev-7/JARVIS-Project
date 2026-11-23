"""Unit and Integration Test Suite for JARVIS Phase 26 — Adaptive Learning Engine.

Covers:
1. Feedback detection & parsing (explicit commands + natural language corrections).
2. Automated root-cause diagnosis (Knowledge Gap, Factual Inaccuracy, Intent Misclassification, Logic Error, Preference Update).
3. Regression guard consistency & invariant validation (blocking malicious injections, probe execution).
4. Learning registry audited SQLite persistence (pending, approved, committed, rejected).
5. Dataset curation & JSONL staging buffer.
6. Fine-tuning staging manager (CPU execution, versioned checkpointing, activation & backup rollback).
7. End-to-end multi-turn affirmative approval dialogue workflow.
8. Direct learning command (/learn) and instant retrieval.
"""

import pytest
from pathlib import Path
from core.learning import (
    feedback_analyzer,
    regression_guard,
    learning_registry,
    dataset_curator,
    fine_tuning_staging,
    adaptive_learning_engine,
)
from core.brain.jarvis_brain import JARVISBrain
from core.brain.router import IntentRouter, IntentType


# ==============================================================================
# 1. Feedback Detection and Parsing
# ==============================================================================

def test_feedback_detection():
    """Verify detection of explicit and natural feedback expressions."""
    assert feedback_analyzer.is_feedback("/feedback Canberra is the capital of Australia")
    assert feedback_analyzer.is_feedback("/learn Java uses JVM bytecode execution")
    assert feedback_analyzer.is_feedback("That is incorrect, the answer is 42")
    assert feedback_analyzer.is_feedback("No, the actual answer is Canberra")
    assert feedback_analyzer.is_feedback("Actually, it runs on Linux")
    assert feedback_analyzer.is_feedback("You made a mistake, that was wrong")
    assert not feedback_analyzer.is_feedback("What is the capital of France?")
    assert not feedback_analyzer.is_feedback("Write a Java program to add two numbers")


def test_correction_extraction():
    """Verify clean extraction of correction and historical turns."""
    history = [
        {"role": "user", "content": "What is the capital of Australia?"},
        {"role": "assistant", "content": "The capital of Australia is Sydney."},
    ]
    parsed = feedback_analyzer.extract_correction(
        "That is incorrect, the actual answer is Canberra.",
        previous_turns=history,
    )
    assert "Canberra" in parsed["user_correction"]
    assert "Australia" in parsed["original_query"]
    assert "Sydney" in parsed["original_response"]


# ==============================================================================
# 2. Root-Cause Diagnosis
# ==============================================================================

def test_root_cause_diagnosis_categories():
    """Verify categorization across the 5 diagnostic failure modes."""
    # 1. Preference Update
    d_pref = feedback_analyzer.diagnose_root_cause(
        original_query="Who am I?",
        original_response="I don't have your name yet.",
        user_correction="My name is Harsha and I work as a backend developer.",
    )
    assert d_pref.root_cause == "PREFERENCE_UPDATE"

    # 2. Logical / Math Error
    d_math = feedback_analyzer.diagnose_root_cause(
        original_query="What is 15 * 4?",
        original_response="Calculation result: 50",
        user_correction="No, 15 * 4 is 60.",
    )
    assert d_math.root_cause == "LOGICAL_REASONING_ERROR"

    # 3. Intent Misclassification
    d_intent = feedback_analyzer.diagnose_root_cause(
        original_query="Write a python program to reverse a string",
        original_response="**Universal Research Report:** Live query for python string reverse...",
        user_correction="I asked for code, not internet research.",
    )
    assert d_intent.root_cause == "INTENT_MISCLASSIFICATION"

    # 4. Factual Inaccuracy
    d_fact = feedback_analyzer.diagnose_root_cause(
        original_query="What is the default port for HTTP?",
        original_response="HTTP defaults to port 8080.",
        user_correction="No, HTTP defaults to port 80, not 8080.",
    )
    assert d_fact.root_cause == "FACTUAL_INACCURACY"

    # 5. Knowledge Gap
    d_gap = feedback_analyzer.diagnose_root_cause(
        original_query="What is the capital of Tuvalu?",
        original_response="I do not have detailed information on that topic.",
        user_correction="The capital of Tuvalu is Funafuti.",
    )
    assert d_gap.root_cause == "KNOWLEDGE_GAP"


# ==============================================================================
# 3. Regression Guard & Consistency Validation
# ==============================================================================

def test_regression_guard_invariants():
    """Verify rejection of dangerous shell injections and false axioms."""
    # Dangerous shell command
    res_shell = regression_guard.validate_consistency("format c: /fs:NTFS")
    assert res_shell["is_safe"] is False

    # Mathematical absurdity
    res_math = regression_guard.validate_consistency("2 + 2 = 5")
    assert res_math["is_safe"] is False

    # Valid factual update
    res_valid = regression_guard.validate_consistency("Canberra is the capital of Australia.")
    assert res_valid["is_safe"] is True


@pytest.mark.asyncio
async def test_regression_guard_probes():
    """Verify regression probes execute successfully on persistent knowledge."""
    probe_res = await regression_guard.run_regression_probes("test probe fact")
    assert probe_res["passed"] is True
    assert probe_res["probes_run"] > 0
    assert probe_res["probes_passed"] == probe_res["probes_run"]


# ==============================================================================
# 4. Learning Registry Persistence
# ==============================================================================

@pytest.mark.asyncio
async def test_learning_registry_lifecycle():
    """Verify creating, approving, listing, and committing learning records in SQLite."""
    session_id = "test_reg_session"
    rec_id = await learning_registry.record_feedback(
        session_id=session_id,
        original_query="What is SQLite?",
        original_response="A cloud database.",
        user_correction="SQLite is a serverless, embedded SQL database engine.",
        root_cause="FACTUAL_INACCURACY",
        status="pending_approval",
    )
    assert rec_id > 0

    record = await learning_registry.get_record(rec_id)
    assert record is not None
    assert record["status"] == "pending_approval"
    assert record["root_cause"] == "FACTUAL_INACCURACY"

    # Update to approved
    await learning_registry.update_status(rec_id, "approved")
    updated = await learning_registry.get_record(rec_id)
    assert updated["status"] == "approved"
    assert updated["approved_at"] is not None

    # List records
    all_recs = await learning_registry.list_records(limit=5)
    assert any(r["id"] == rec_id for r in all_recs)


# ==============================================================================
# 5. Dataset Curation & Staging
# ==============================================================================

def test_dataset_curation_and_staging(tmp_path):
    """Verify training pair synthesis and local JSONL staging buffer."""
    dataset_curator.staged_file = tmp_path / "staged_learning.jsonl"

    pair = dataset_curator.synthesize_pair(
        original_query="What is the capital of Australia?",
        user_correction="Canberra is the capital of Australia.",
        root_cause="KNOWLEDGE_GAP",
    )
    assert "User: What is the capital of Australia?" in pair["prompt"]
    assert "Canberra" in pair["completion"]

    count = dataset_curator.append_to_staged_file(
        prompt=pair["prompt"],
        completion=pair["completion"],
        category=pair["category"],
    )
    assert count == 1
    assert dataset_curator.count_staged_lines() == 1

    entries = dataset_curator.load_staged_dataset()
    assert len(entries) == 1
    assert "Canberra" in entries[0]["completion"]

    dataset_curator.clear_staged_file()
    assert dataset_curator.count_staged_lines() == 0


# ==============================================================================
# 6. Fine-Tuning Staging & Rollback Manager
# ==============================================================================

def test_fine_tuning_staging_and_rollback(tmp_path):
    """Verify CPU candidate model training, activation, and rollback."""
    staging_file = tmp_path / "data" / "staged_learning.jsonl"
    staging_file.parent.mkdir(parents=True, exist_ok=True)
    staging_file.write_text('{"prompt": "Hello", "completion": " World", "category": "general"}\n', encoding="utf-8")

    dataset_curator.staged_file = staging_file
    fine_tuning_staging.checkpoints_dir = tmp_path / "checkpoints"
    fine_tuning_staging.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    fine_tuning_staging.active_checkpoint = tmp_path / "checkpoints" / "jarvis_micro_v1.pt"
    fine_tuning_staging.backup_checkpoint = tmp_path / "checkpoints" / "jarvis_micro_v1.backup.pt"

    # Train candidate model (1 epoch for speed)
    train_res = fine_tuning_staging.train_candidate(epochs=1, lr=1e-3, batch_size=1)
    assert train_res["success"] is True
    assert Path(train_res["candidate_path"]).exists()

    # Active checkpoint does not exist yet until activated
    assert not fine_tuning_staging.active_checkpoint.exists()

    # Activate candidate model
    act_res = fine_tuning_staging.activate_candidate()
    assert act_res["success"] is True
    assert fine_tuning_staging.active_checkpoint.exists()

    # Train a second candidate and activate to test backup
    staging_file.write_text('{"prompt": "Hello 2", "completion": " World 2", "category": "general"}\n', encoding="utf-8")
    train_res2 = fine_tuning_staging.train_candidate(epochs=1, lr=1e-3, batch_size=1)
    assert train_res2["success"] is True
    fine_tuning_staging.activate_candidate()
    assert fine_tuning_staging.backup_checkpoint.exists()

    # Test rollback
    roll_res = fine_tuning_staging.rollback_to_backup()
    assert roll_res["success"] is True
    assert fine_tuning_staging.active_checkpoint.exists()


# ==============================================================================
# 7. Router Intent Classification
# ==============================================================================

def test_router_adaptive_learning_intent():
    """Verify intent routing for adaptive learning commands and corrections."""
    router = IntentRouter()

    dec1 = router.route("/feedback The capital of Australia is Canberra")
    assert dec1.intent == IntentType.ADAPTIVE_LEARNING

    dec2 = router.route("/learn Python dictionaries preserve insertion order")
    assert dec2.intent == IntentType.ADAPTIVE_LEARNING

    dec3 = router.route("/learning-log")
    assert dec3.intent == IntentType.ADAPTIVE_LEARNING

    dec4 = router.route("/finetune-staged")
    assert dec4.intent == IntentType.ADAPTIVE_LEARNING

    dec5 = router.route("That is incorrect, the answer is 42")
    assert dec5.intent == IntentType.ADAPTIVE_LEARNING

    dec6 = router.route("Actually, it runs on Linux")
    assert dec6.intent == IntentType.ADAPTIVE_LEARNING


# ==============================================================================
# 8. End-to-End Affirmative Approval & Direct Learning Workflow
# ==============================================================================

@pytest.mark.asyncio
async def test_e2e_feedback_affirmative_approval_workflow():
    """Test full feedback lifecycle: proposal -> confirmation (yes) -> commitment."""
    brain = JARVISBrain()
    session_id = "test_e2e_learning"

    # Step 1: User provides conversational correction
    resp_prop = await brain.think(
        "That is incorrect, the capital of Australia is Canberra.",
        session_id=session_id,
    )
    assert "Feedback Diagnostic Analysis" in resp_prop.content
    assert "KNOWLEDGE_GAP" in resp_prop.content
    assert "Would you like me to commit" in resp_prop.content

    # Step 2: User confirms with "yes"
    resp_confirm = await brain.think("yes", session_id=session_id)
    assert "Adaptive Learning Confirmed & Committed" in resp_confirm.content
    assert "Committed to permanent SQLite memory" in resp_confirm.content


@pytest.mark.asyncio
async def test_e2e_direct_learn_command():
    """Test /learn command directly learning and indexing verified knowledge."""
    brain = JARVISBrain()
    session_id = "test_direct_learn"

    resp = await brain.think(
        "/learn Python dictionaries preserve insertion order since version 3.7.",
        session_id=session_id,
    )
    assert "Direct Fact Learned & Indexed" in resp.content
    assert "Python dictionaries preserve insertion order" in resp.content


@pytest.mark.asyncio
async def test_e2e_feedback_rejection_workflow():
    """Test full feedback lifecycle: proposal -> rejection (no) -> aborted."""
    brain = JARVISBrain()
    session_id = "test_e2e_rejection"

    # Step 1: User provides correction
    resp_prop = await brain.think(
        "That is incorrect, test rejection fact.",
        session_id=session_id,
    )
    assert "Would you like me to commit" in resp_prop.content

    # Step 2: User rejects with "no"
    resp_reject = await brain.think("no", session_id=session_id)
    assert "rejected and will not be saved" in resp_reject.content
