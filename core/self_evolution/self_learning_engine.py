"""J.A.R.V.I.S. Self-Evolution System — Phase 29: Self-Learning Engine
=====================================================================
The Self-Learning Engine is the foundational intelligence component of the
J.A.R.V.I.S. Self-Evolution Architecture.

Core Capabilities:
1. Learn from Approved User Feedback:
   - Detects corrections, parses user intent, diagnoses root causes, and gates
     permanent changes behind explicit user approval.
2. Analyze Previous Task Results:
   - Evaluates workflow execution history, detects failures/exceptions, and
     extracts actionable lessons learned.
3. Record Mistakes and Corrections:
   - Maintains an audited mistake-and-correction ledger (`mistakes_log.jsonl`)
     and indexes lessons in vector memory for proactive avoidance during future planning.
4. Update Local Knowledge Database:
   - Integrates verified facts directly into SQLite `KnowledgeItem` and in-memory
     knowledge engines, ensuring immediate factual availability without cloud lookups.
5. Generate Candidate Training Examples:
   - Curates clean (prompt, completion) instruction pairs into staging buffers
     (`.jarvis/staging/candidate_training_pairs.jsonl`), keeping candidate data isolated
     and never silently training or altering active neural weights.
6. Evaluate Learning Progress:
   - Measures learning velocity, mistake repetition rate, verified knowledge additions,
     and produces an overall quantitative learning progress score (0.0 to 100.0).

Strict Safety Principles:
- Never silently modify the active custom neural model or training datasets.
- Strictly decouple local knowledge updates (instant SQLite/vector recall) from
  neural-network parameter fine-tuning.
- Keep the user in control through explicit approval verification.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from config.settings import BASE_DIR
from core.logger import log
from core.learning.feedback_analyzer import feedback_analyzer, DiagnosticResult
from core.learning.learning_registry import learning_registry
from core.learning.regression_guard import regression_guard
from core.memory.persistent_memory import persistent_memory, semantic_vector_store
from core.knowledge.knowledge_engine import knowledge_engine


@dataclass
class MistakeCorrectionRecord:
    """An audited record of a detected mistake, root cause, and verified correction."""
    record_id: str
    task_id: Optional[str]
    context: str
    mistake_description: str
    root_cause: str  # e.g. "KNOWLEDGE_GAP", "FACTUAL_INACCURACY", "LOGICAL_REASONING_ERROR", "EXECUTION_FAILURE", "INTENT_MISCLASSIFICATION"
    verified_correction: str
    timestamp: float = field(default_factory=time.time)
    avoidance_count: int = 0
    status: str = "pending_approval"  # "pending_approval", "approved", "rejected", "committed"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TaskAnalysisResult:
    """Analysis of a completed or failed task execution."""
    task_id: str
    goal: str
    status: str  # "completed", "failed", "degraded", "aborted"
    error_detected: bool
    root_cause: Optional[str]
    lesson_learned: Optional[str]
    should_record_mistake: bool
    recommendation: str
    duration_sec: float = 0.0
    steps_count: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CandidateTrainingExample:
    """An instruction-tuning prompt-completion pair staged for future offline fine-tuning."""
    example_id: str
    prompt: str
    completion: str
    category: str  # "knowledge", "reasoning", "orchestration", "automation"
    source: str    # "user_feedback", "task_lesson", "knowledge_update"
    created_at: float = field(default_factory=time.time)
    is_staged: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LearningProgressReport:
    """Quantitative evaluation of J.A.R.V.I.S. learning metrics over time."""
    total_feedback_processed: int
    approved_feedback_count: int
    mistakes_recorded: int
    mistakes_avoided: int
    knowledge_items_updated: int
    candidate_training_examples_count: int
    mistake_repeat_rate: float       # Percentage of repeat errors (0.0 to 1.0)
    learning_velocity: float          # Items acquired per session/period
    overall_learning_score: float     # 0.0 to 100.0 score
    summary: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SelfLearningEngine:
    """Master controller for the Phase 29 Self-Learning Engine."""

    def __init__(
        self,
        evolution_dir: Optional[Path] = None,
        staging_dir: Optional[Path] = None,
    ):
        self.evolution_dir = evolution_dir or (BASE_DIR / ".jarvis" / "evolution")
        self.staging_dir = staging_dir or (BASE_DIR / ".jarvis" / "staging")
        self.mistakes_file = self.evolution_dir / "mistakes_log.jsonl"
        self.metrics_file = self.evolution_dir / "learning_metrics.json"
        self.candidate_file = self.staging_dir / "candidate_training_pairs.jsonl"
        self.legacy_staged_file = BASE_DIR / "custom_ai" / "data" / "staged_learning.jsonl"

        self._in_memory_mistakes: Dict[str, MistakeCorrectionRecord] = {}
        self._initialized = False

    async def initialize(self) -> None:
        """Sets up directories, loads existing logs, and connects to persistent stores."""
        if self._initialized:
            return

        self.evolution_dir.mkdir(parents=True, exist_ok=True)
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        self.legacy_staged_file.parent.mkdir(parents=True, exist_ok=True)

        # Initialize persistent memory and learning registry
        await persistent_memory.initialize()
        await learning_registry.initialize()

        # Load persisted mistakes into memory cache
        if self.mistakes_file.exists():
            try:
                with open(self.mistakes_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            data = json.loads(line)
                            record = MistakeCorrectionRecord(**data)
                            self._in_memory_mistakes[record.record_id] = record
            except Exception as exc:
                log.warning(f"SelfLearningEngine failed to load existing mistakes log: {exc}")

        self._initialized = True
        log.info("Phase 29 SelfLearningEngine initialized successfully.")

    # ──────────────────────────────────────────────────────────────────────────
    # 1. LEARN FROM APPROVED USER FEEDBACK
    # ──────────────────────────────────────────────────────────────────────────

    async def process_user_feedback(
        self,
        user_message: str,
        session_id: str = "default_session",
        previous_turns: Optional[List[Dict[str, str]]] = None,
        auto_approve: bool = False,
    ) -> Dict[str, Any]:
        """
        Parses user feedback, detects errors, diagnoses root causes, and either
        prompts the user for approval or applies the update immediately.
        """
        await self.initialize()

        # 1. Parse correction details
        parsed = feedback_analyzer.extract_correction(user_message, previous_turns=previous_turns)
        user_correction = parsed["user_correction"]
        original_query = parsed["original_query"]
        original_response = parsed["original_response"]

        # 2. Diagnose root-cause
        diagnosis: DiagnosticResult = feedback_analyzer.diagnose_root_cause(
            original_query=original_query,
            original_response=original_response,
            user_correction=user_correction,
        )

        # 3. Safety and regression check
        val_result = regression_guard.validate_consistency(
            user_correction=user_correction,
            root_cause=diagnosis.root_cause,
        )
        if not val_result["is_safe"]:
            log.warning(f"Feedback rejected by regression guard: {val_result['reason']}")
            return {
                "success": False,
                "error": val_result["reason"],
                "diagnosis": asdict(diagnosis),
                "status": "rejected",
            }

        # 4. Record in persistent registry
        initial_status = "approved" if auto_approve else "pending_approval"
        record_id = await learning_registry.record_feedback(
            session_id=session_id,
            original_query=original_query,
            original_response=original_response,
            user_correction=user_correction,
            root_cause=diagnosis.root_cause,
            status=initial_status,
        )

        # 5. Handle approval or pending state
        if auto_approve:
            commit_res = await self.apply_approved_feedback(record_id)
            return {
                "success": True,
                "record_id": record_id,
                "status": "committed",
                "diagnosis": asdict(diagnosis),
                "auto_approved": True,
                "commit_details": commit_res,
                "summary": f"Verified correction applied: '{user_correction[:60]}...'",
            }

        # User approval prompt
        approval_prompt = (
            f"**Self-Learning Diagnostic Analysis:**\n"
            f"- **Detected Issue:** `{diagnosis.root_cause}`\n"
            f"- **Explanation:** {diagnosis.explanation}\n"
            f"- **Proposed Knowledge Update:** *\"{user_correction}\"*\n\n"
            f"Would you like me to commit this update to permanent verified memory? (yes/no)"
        )

        return {
            "success": True,
            "record_id": record_id,
            "status": "pending_approval",
            "needs_approval": True,
            "diagnosis": asdict(diagnosis),
            "user_correction": user_correction,
            "prompt_text": approval_prompt,
        }

    async def apply_approved_feedback(self, record_id: int) -> Dict[str, Any]:
        """
        Applies a verified, approved feedback record:
        - Updates local knowledge database (instant recall, zero cloud).
        - Logs mistake and correction for future avoidance.
        - Stages candidate training pair (never silently fine-tuning neural weights).
        """
        await self.initialize()
        record_dict = await learning_registry.get_record(record_id)
        if not record_dict:
            return {"success": False, "error": f"Record #{record_id} not found."}

        original_query = record_dict.get("original_query", "")
        user_correction = record_dict.get("user_correction", "")
        root_cause = record_dict.get("root_cause", "KNOWLEDGE_GAP")
        original_response = record_dict.get("original_response", "")

        # 1. Update status to approved in registry
        await learning_registry.update_status(record_id, "approved")

        # 2. Update local knowledge database
        topic = original_query if original_query and original_query != "Direct user instruction" else "Learned Correction"
        knowledge_id = await self.update_local_knowledge(
            title=topic[:100],
            content=user_correction,
            category="user_learning",
            source="approved_feedback",
        )

        # 3. Record structured mistake & correction
        mistake_rec = await self.record_mistake_and_correction(
            mistake_description=f"Query: '{original_query}' led to inaccurate response: '{original_response[:100]}...'",
            verified_correction=user_correction,
            root_cause=root_cause,
            context=f"session_record_{record_id}",
        )

        # 4. Synthesize candidate training example into isolated staging buffer
        example = self.generate_candidate_training_example(
            prompt=f"User: {original_query if original_query else 'Explain the concept of ' + user_correction[:30]}\nJARVIS:",
            completion=f" {user_correction}",
            category="knowledge",
            source="user_feedback",
        )

        # 5. Mark record committed
        await learning_registry.update_status(record_id, "committed")

        return {
            "success": True,
            "record_id": record_id,
            "knowledge_id": knowledge_id,
            "mistake_record_id": mistake_rec.record_id,
            "candidate_example_id": example.example_id,
            "status": "committed",
        }

    # ──────────────────────────────────────────────────────────────────────────
    # 2. ANALYZE PREVIOUS TASK RESULTS
    # ──────────────────────────────────────────────────────────────────────────

    async def analyze_task_result(
        self,
        task_id: str,
        goal: str,
        status: str,
        duration_sec: float = 0.0,
        error_summary: Optional[str] = None,
        lessons_learned: Optional[str] = None,
        steps_count: int = 1,
    ) -> TaskAnalysisResult:
        """
        Post-mortem analysis of an executed autonomous task.
        Detects failure, extracts lessons, records mistakes, and indexes them
        in vector memory so future plans automatically avoid repeating them.
        """
        await self.initialize()

        status_lower = status.lower().strip()
        error_detected = status_lower in ["failed", "aborted", "degraded", "error"] or bool(error_summary)

        root_cause: Optional[str] = None
        extracted_lesson = lessons_learned
        recommendation: str = ""

        if error_detected:
            # Diagnose task error root cause
            err_text = (error_summary or "").lower()
            if "timeout" in err_text or "time out" in err_text:
                root_cause = "EXECUTION_TIMEOUT"
                recommendation = "Increase step timeout or decompose into smaller atomic tasks."
            elif "not found" in err_text or "filenotfound" in err_text or "missing" in err_text:
                root_cause = "RESOURCE_NOT_FOUND"
                recommendation = "Verify file/resource paths before dispatching execution."
            elif "permission" in err_text or "denied" in err_text or "access" in err_text:
                root_cause = "PERMISSION_DENIED"
                recommendation = "Check user permissions and security sandboxing boundaries."
            elif "syntax" in err_text or "parse" in err_text:
                root_cause = "SYNTAX_PARSING_ERROR"
                recommendation = "Run static AST syntax validation prior to execution."
            else:
                root_cause = "GENERAL_EXECUTION_FAILURE"
                recommendation = "Review execution plan and ensure retry circuits are armed."

            if not extracted_lesson:
                extracted_lesson = f"Task '{goal}' failed due to {root_cause}. Recommendation: {recommendation}. Error: {error_summary or 'None'}"

            # Record mistake in ledger
            await self.record_mistake_and_correction(
                mistake_description=f"Task '{goal}' failed: {error_summary or 'Unspecified error'}",
                verified_correction=extracted_lesson,
                root_cause=root_cause,
                context=f"task_{task_id}",
                task_id=task_id,
            )

        else:
            recommendation = "Task succeeded without anomalies."

        # Save to persistent execution history
        try:
            await persistent_memory.log_task_execution(
                task_id=task_id,
                goal=goal,
                status=status,
                steps_count=steps_count,
                duration_sec=duration_sec,
                error_summary=error_summary,
                lessons_learned=extracted_lesson,
            )
        except Exception as exc:
            log.warning(f"Failed to persist task execution in database: {exc}")

        result = TaskAnalysisResult(
            task_id=task_id,
            goal=goal,
            status=status,
            error_detected=error_detected,
            root_cause=root_cause,
            lesson_learned=extracted_lesson,
            should_record_mistake=error_detected,
            recommendation=recommendation,
            duration_sec=duration_sec,
            steps_count=steps_count,
        )

        log.info(f"Task #{task_id} analysis completed. Error detected: {error_detected} [{root_cause or 'OK'}]")
        return result

    # ──────────────────────────────────────────────────────────────────────────
    # 3. RECORD MISTAKES AND CORRECTIONS
    # ──────────────────────────────────────────────────────────────────────────

    async def record_mistake_and_correction(
        self,
        mistake_description: str,
        verified_correction: str,
        root_cause: str = "EXECUTION_FAILURE",
        context: str = "",
        task_id: Optional[str] = None,
    ) -> MistakeCorrectionRecord:
        """
        Appends mistake to persistent audit ledger (`mistakes_log.jsonl`)
        and indexes it in the semantic vector store for proactive retrieval.
        """
        await self.initialize()

        record_id = f"mstk_{int(time.time() * 1000)}"
        record = MistakeCorrectionRecord(
            record_id=record_id,
            task_id=task_id,
            context=context,
            mistake_description=mistake_description,
            root_cause=root_cause,
            verified_correction=verified_correction,
            timestamp=time.time(),
            avoidance_count=0,
            status="committed",
        )

        self._in_memory_mistakes[record_id] = record

        # Append to JSONL file
        try:
            with open(self.mistakes_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
        except Exception as exc:
            log.error(f"Failed to write mistake record to {self.mistakes_file}: {exc}")

        # Index in vector memory for mistake avoidance search
        try:
            semantic_vector_store.add_document(
                doc_id=f"mistake_{record_id}",
                text=f"Mistake: {mistake_description}. Root Cause: {root_cause}. Correction/Lesson: {verified_correction}",
                metadata={
                    "category": "mistake_lesson",
                    "record_id": record_id,
                    "root_cause": root_cause,
                    "task_id": task_id or "",
                },
            )
        except Exception as exc:
            log.warning(f"Failed to index mistake in vector memory: {exc}")

        log.info(f"Recorded mistake #{record_id} [{root_cause}]: {mistake_description[:80]}")
        return record

    def check_mistake_prevention(self, planned_action: str) -> List[Dict[str, Any]]:
        """
        Searches recorded mistakes for past failures related to a planned action
        or goal, returning proactive warnings and lessons learned.
        """
        warnings = []
        try:
            matches = semantic_vector_store.search(planned_action, top_k=3)
            for doc, score in matches:
                if doc.metadata.get("category") == "mistake_lesson" or "Mistake:" in doc.text:
                    rec_id = doc.metadata.get("record_id", "")
                    if rec_id in self._in_memory_mistakes:
                        self._in_memory_mistakes[rec_id].avoidance_count += 1

                    warnings.append({
                        "doc_id": doc.doc_id,
                        "text": doc.text,
                        "relevance_score": round(score, 3),
                        "metadata": doc.metadata,
                    })
        except Exception as exc:
            log.warning(f"Error querying mistake prevention: {exc}")

        return warnings

    # ──────────────────────────────────────────────────────────────────────────
    # 4. UPDATE LOCAL KNOWLEDGE DATABASE
    # ──────────────────────────────────────────────────────────────────────────

    async def update_local_knowledge(
        self,
        title: str,
        content: str,
        category: str = "general",
        source: str = "user_feedback",
    ) -> int:
        """
        Updates the local offline knowledge store (SQLite + Vector DB + In-Memory QA).
        Strict safety: This updates the factual retrieval layer and does NOT modify
        active neural network weights.
        """
        await self.initialize()

        # 1. Store in SQLite knowledge table and index in TF-IDF vector store
        knowledge_id = await persistent_memory.store_knowledge(
            title=title,
            content=content,
            category=category,
            source=source,
        )

        # 2. Update KnowledgeEngine in-memory QA map if appropriate
        slug = re.sub(r"[^a-z0-9_]+", "_", title.lower().strip()).strip("_")
        if slug and slug not in knowledge_engine.KNOWLEDGE_BASE:
            knowledge_engine.KNOWLEDGE_BASE[slug] = {
                "title": title,
                "aliases": [slug, title.lower()],
                "overview": content,
                "how_it_works": f"Verified factual knowledge stored from source '{source}'.",
                "examples": [content],
                "simplified": content,
            }

        log.info(f"Local knowledge updated with ID #{knowledge_id}: '{title}'")
        return knowledge_id

    # ──────────────────────────────────────────────────────────────────────────
    # 5. GENERATE CANDIDATE TRAINING EXAMPLES
    # ──────────────────────────────────────────────────────────────────────────

    def generate_candidate_training_example(
        self,
        prompt: str,
        completion: str,
        category: str = "knowledge",
        source: str = "user_feedback",
    ) -> CandidateTrainingExample:
        """
        Creates a clean (prompt, completion) instruction-tuning pair and appends
        it to the staging file. Candidate pairs are isolated from active weights.
        """
        example_id = f"trn_{int(time.time() * 1000)}"

        # Clean prompt and completion formatting
        cleaned_prompt = prompt.strip()
        cleaned_completion = completion if completion.startswith(" ") else f" {completion.strip()}"

        example = CandidateTrainingExample(
            example_id=example_id,
            prompt=cleaned_prompt,
            completion=cleaned_completion,
            category=category,
            source=source,
            created_at=time.time(),
            is_staged=True,
        )

        entry_json = json.dumps(example.to_dict(), ensure_ascii=False) + "\n"

        # 1. Write to .jarvis/staging/candidate_training_pairs.jsonl
        try:
            with open(self.candidate_file, "a", encoding="utf-8") as f:
                f.write(entry_json)
        except Exception as exc:
            log.error(f"Failed to write candidate training pair to {self.candidate_file}: {exc}")

        # 2. Also append to custom_ai/data/staged_learning.jsonl for compatibility
        try:
            with open(self.legacy_staged_file, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "prompt": cleaned_prompt,
                    "completion": cleaned_completion,
                    "category": category,
                }, ensure_ascii=False) + "\n")
        except Exception as exc:
            log.warning(f"Failed to append to legacy staged file: {exc}")

        log.info(f"Synthesized candidate training pair #{example_id} [category={category}]")
        return example

    def count_candidate_examples(self) -> int:
        """Counts total staged candidate training examples."""
        count = 0
        if self.candidate_file.exists():
            try:
                with open(self.candidate_file, "r", encoding="utf-8") as f:
                    count = sum(1 for line in f if line.strip())
            except Exception:
                pass
        return count

    # ──────────────────────────────────────────────────────────────────────────
    # 6. EVALUATE LEARNING PROGRESS
    # ──────────────────────────────────────────────────────────────────────────

    async def evaluate_learning_progress(self) -> LearningProgressReport:
        """
        Computes measurable metrics for J.A.R.V.I.S. self-learning:
        - Feedback volume and approval counts
        - Mistake count and proactive avoidance frequency
        - Knowledge database expansion
        - Staged candidate training set size
        - Overall learning score (0.0 to 100.0)
        """
        await self.initialize()

        # Feedback stats from persistent memory
        records = await learning_registry.list_records(limit=200)
        total_feedback = len(records)
        approved_count = sum(1 for r in records if r.get("status") in ["approved", "committed"])

        # Mistake stats
        mistakes_count = len(self._in_memory_mistakes)
        total_avoidances = sum(m.avoidance_count for m in self._in_memory_mistakes.values())

        # Knowledge stats
        knowledge_count = len(knowledge_engine.KNOWLEDGE_BASE)

        # Candidate examples
        candidate_count = self.count_candidate_examples()

        # Mistake repeat rate calculation
        # If total mistakes > 0, repeat rate is ratio of repeat mistakes vs total
        repeat_count = 0
        seen_roots: Dict[str, int] = {}
        for m in self._in_memory_mistakes.values():
            key = f"{m.root_cause}:{m.mistake_description[:30]}"
            seen_roots[key] = seen_roots.get(key, 0) + 1
            if seen_roots[key] > 1:
                repeat_count += 1

        mistake_repeat_rate = round(repeat_count / max(mistakes_count, 1), 3)

        # Compute composite learning score (0.0 to 100.0)
        # Factor 1: Feedback conversion rate (approved / total) -> up to 30 pts
        conv_rate = (approved_count / max(total_feedback, 1)) if total_feedback > 0 else 1.0
        score_feedback = min(conv_rate * 30.0, 30.0)

        # Factor 2: Mistake avoidance score -> up to 30 pts (higher avoidances, lower repeat rate)
        avoidance_factor = max(0.0, 1.0 - mistake_repeat_rate)
        score_avoidance = avoidance_factor * 30.0

        # Factor 3: Knowledge base growth -> up to 25 pts
        score_knowledge = min(knowledge_count * 2.5, 25.0)

        # Factor 4: Staged training curation -> up to 15 pts
        score_staging = min(candidate_count * 3.0, 15.0)

        overall_score = round(score_feedback + score_avoidance + score_knowledge + score_staging, 2)
        overall_score = min(max(overall_score, 0.0), 100.0)

        summary = (
            f"JARVIS Self-Learning Score: {overall_score}/100. "
            f"Processed {total_feedback} feedback items ({approved_count} approved). "
            f"Logged {mistakes_count} unique mistakes with {total_avoidances} proactive avoidances. "
            f"Knowledge store holds {knowledge_count} topics; {candidate_count} candidate training pairs staged."
        )

        report = LearningProgressReport(
            total_feedback_processed=total_feedback,
            approved_feedback_count=approved_count,
            mistakes_recorded=mistakes_count,
            mistakes_avoided=total_avoidances,
            knowledge_items_updated=knowledge_count,
            candidate_training_examples_count=candidate_count,
            mistake_repeat_rate=mistake_repeat_rate,
            learning_velocity=round(approved_count / max(total_feedback, 1), 2),
            overall_learning_score=overall_score,
            summary=summary,
        )

        # Save metrics to disk
        try:
            with open(self.metrics_file, "w", encoding="utf-8") as f:
                json.dump(report.to_dict(), f, indent=2)
        except Exception as exc:
            log.warning(f"Failed to persist learning metrics: {exc}")

        return report

    def get_summary_report_markdown(self, progress: LearningProgressReport) -> str:
        """Generates a clean human-readable markdown status report."""
        return (
            f"### 🧠 J.A.R.V.I.S. Self-Learning Engine Status (Phase 29)\n\n"
            f"- **Overall Learning Score**: `{progress.overall_learning_score} / 100.0`\n"
            f"- **Total Feedback Processed**: `{progress.total_feedback_processed}` (Approved: `{progress.approved_feedback_count}`)\n"
            f"- **Mistakes Logged**: `{progress.mistakes_recorded}`\n"
            f"- **Proactive Mistake Avoidances**: `{progress.mistakes_avoided}`\n"
            f"- **Mistake Repeat Rate**: `{progress.mistake_repeat_rate * 100:.1f}%`\n"
            f"- **Knowledge Items Active**: `{progress.knowledge_items_updated}`\n"
            f"- **Staged Training Pairs**: `{progress.candidate_training_examples_count}`\n\n"
            f"> [!NOTE]\n"
            f"> Knowledge updates are isolated in SQLite and vector memory. Active neural network weights "
            f"remain untouched until candidate datasets pass regression testing and receive explicit user authorization."
        )


# Global singleton instance for system-wide access
self_learning_engine = SelfLearningEngine()
