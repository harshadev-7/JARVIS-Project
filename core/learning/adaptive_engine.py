"""J.A.R.V.I.S. Adaptive Learning Engine Orchestrator.

Unifies feedback analysis, root-cause diagnosis, persistent audits,
regression checking, knowledge indexing, and fine-tuning staging.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from core.learning.feedback_analyzer import feedback_analyzer
from core.learning.learning_registry import learning_registry
from core.learning.regression_guard import regression_guard
from core.learning.dataset_curator import dataset_curator
from core.learning.fine_tuning_staging import fine_tuning_staging
from core.memory.persistent_memory import persistent_memory
from core.logger import log


class AdaptiveLearningEngine:
    """Master orchestrator for the JARVIS controlled adaptive learning pipeline."""

    async def process_feedback(
        self,
        user_message: str,
        session_id: str,
        previous_turns: Optional[List[Dict[str, str]]] = None,
        auto_commit: bool = False,
    ) -> Dict[str, Any]:
        """
        Ingest user feedback/correction, analyze root cause, validate invariants,
        and log to the learning registry.
        """
        parsed = feedback_analyzer.extract_correction(user_message, previous_turns=previous_turns)
        user_correction = parsed["user_correction"]
        original_query = parsed["original_query"]
        original_response = parsed["original_response"]

        # 1. Diagnose root-cause
        diagnosis = feedback_analyzer.diagnose_root_cause(
            original_query=original_query,
            original_response=original_response,
            user_correction=user_correction,
        )

        # 2. Safety and consistency validation
        val_result = regression_guard.validate_consistency(
            user_correction=user_correction,
            root_cause=diagnosis.root_cause,
        )
        if not val_result["is_safe"]:
            return {
                "success": False,
                "error": val_result["reason"],
                "diagnosis": diagnosis.__dict__,
            }

        # 3. Record in persistent registry
        initial_status = "approved" if auto_commit else "pending_approval"
        record_id = await learning_registry.record_feedback(
            session_id=session_id,
            original_query=original_query,
            original_response=original_response,
            user_correction=user_correction,
            root_cause=diagnosis.root_cause,
            status=initial_status,
        )

        if auto_commit:
            commit_res = await self.commit_record(record_id)
            return {
                "success": True,
                "record_id": record_id,
                "diagnosis": diagnosis.__dict__,
                "auto_committed": True,
                "commit_details": commit_res,
                "summary": (
                    f"Verified and committed to permanent memory [{diagnosis.root_cause}]: "
                    f"\"{user_correction[:60]}...\""
                ),
            }

        prompt_text = (
            f"**Feedback Diagnostic Analysis:**\n"
            f"- **Diagnosis:** `{diagnosis.root_cause}` ({diagnosis.explanation})\n"
            f"- **Recommended Action:** {diagnosis.recommended_action}\n"
            f"- **Proposed Update:** *\"{user_correction}\"*\n\n"
            f"Would you like me to commit this correction to permanent verified memory? (yes/no)"
        )

        return {
            "success": True,
            "record_id": record_id,
            "diagnosis": diagnosis.__dict__,
            "user_correction": user_correction,
            "needs_approval": True,
            "prompt_text": prompt_text,
        }

    async def commit_record(self, record_id: int) -> Dict[str, Any]:
        """
        Approve, index into SQL knowledge base, synthesize training pair,
        and stage for optional fine-tuning.
        """
        record = await learning_registry.get_record(record_id)
        if not record:
            return {"success": False, "error": f"Record #{record_id} not found."}

        correction = record["user_correction"]
        query = record["original_query"]
        root_cause = record["root_cause"]

        # 1. Run regression probe test
        probe_res = await regression_guard.run_regression_probes(correction)

        # 2. Index into persistent SQL knowledge base & vector store
        title = f"Learned [{root_cause}]: {correction[:30]}"
        item_id = await persistent_memory.store_knowledge(
            title=title,
            content=correction,
            category=root_cause.lower(),
            source="user_feedback",
        )

        # 3. Synthesize training pair for fine-tuning
        pair = dataset_curator.synthesize_pair(
            original_query=query,
            user_correction=correction,
            root_cause=root_cause,
        )

        # 4. Save to SQL examples table and append to staged JSONL buffer
        ex_id = await learning_registry.add_training_example(
            prompt=pair["prompt"],
            completion=pair["completion"],
            category=pair["category"],
            record_id=record_id,
            is_staged=True,
        )
        dataset_curator.append_to_staged_file(
            prompt=pair["prompt"],
            completion=pair["completion"],
            category=pair["category"],
        )

        # 5. Mark record committed
        await learning_registry.update_status(record_id, "committed")

        log.info(f"AdaptiveLearningEngine successfully committed record #{record_id} (Knowledge #{item_id}, Example #{ex_id})")

        return {
            "success": True,
            "record_id": record_id,
            "knowledge_id": item_id,
            "training_example_id": ex_id,
            "probes_passed": probe_res.get("passed", True),
            "message": "Update successfully committed to knowledge repository and staged for neural tuning.",
        }

    async def reject_record(self, record_id: int) -> bool:
        """Mark proposed correction as rejected."""
        return await learning_registry.update_status(record_id, "rejected")

    async def learn_direct_fact(self, fact_text: str, session_id: str = "direct_learn") -> Dict[str, Any]:
        """Directly learn a verified fact (/learn <fact>)."""
        return await self.process_feedback(
            user_message=f"/learn {fact_text}",
            session_id=session_id,
            auto_commit=True,
        )

    async def get_learning_log(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieve recent learning audit records."""
        return await learning_registry.list_records(limit=limit)

    def get_staging_status(self) -> Dict[str, Any]:
        """Inspect fine-tuning staging buffer status."""
        return fine_tuning_staging.get_staging_status()

    def train_staged_model(self, epochs: int = 3) -> Dict[str, Any]:
        """Trigger CPU fine-tuning on staged candidate dataset."""
        return fine_tuning_staging.train_candidate(epochs=epochs)

    def activate_staged_model(self) -> Dict[str, Any]:
        """Activate last trained candidate model."""
        return fine_tuning_staging.activate_candidate()

    def rollback_model(self) -> Dict[str, Any]:
        """Roll back to previous model checkpoint."""
        return fine_tuning_staging.rollback_to_backup()


adaptive_learning_engine = AdaptiveLearningEngine()
__all__ = ["AdaptiveLearningEngine", "adaptive_learning_engine"]
