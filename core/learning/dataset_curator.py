"""J.A.R.V.I.S. Training Dataset Curator.

Converts approved user feedback, verified corrections, and newly acquired
knowledge into clean, balanced instruction-tuning pairs for optional
neural fine-tuning.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from core.logger import log


class DatasetCurator:
    """Synthesizes high-quality training pairs from validated corrections."""

    def __init__(self, data_dir: Optional[Path] = None):
        from config.settings import BASE_DIR
        self.data_dir = data_dir or (BASE_DIR / "custom_ai" / "data")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.staged_file = self.data_dir / "staged_learning.jsonl"

    def synthesize_pair(
        self,
        original_query: str,
        user_correction: str,
        root_cause: str = "KNOWLEDGE_GAP",
    ) -> Dict[str, str]:
        """Convert query and correction into clean prompt-completion pair."""
        cleaned_query = original_query.strip()
        cleaned_correction = user_correction.strip()

        # If original query was generic instruction or missing, infer from correction
        if not cleaned_query or cleaned_query == "Direct user instruction":
            prompt = f"User: Can you explain {cleaned_correction[:40]}?\nJARVIS:"
        else:
            prompt = f"User: {cleaned_query}\nJARVIS:"

        completion = f" {cleaned_correction}"

        category_map = {
            "KNOWLEDGE_GAP": "knowledge",
            "FACTUAL_INACCURACY": "knowledge",
            "LOGICAL_REASONING_ERROR": "logic",
            "INTENT_MISCLASSIFICATION": "orchestration",
            "PREFERENCE_UPDATE": "profile",
        }
        category = category_map.get(root_cause, "general")

        return {
            "prompt": prompt,
            "completion": completion,
            "category": category,
        }

    def append_to_staged_file(self, prompt: str, completion: str, category: str = "knowledge") -> int:
        """Append training pair to local JSONL staging buffer."""
        entry = {
            "prompt": prompt,
            "completion": completion,
            "category": category,
        }
        with open(self.staged_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        log.info(f"DatasetCurator staged training pair to {self.staged_file}")
        return self.count_staged_lines()

    def count_staged_lines(self) -> int:
        """Count total lines in staged JSONL file."""
        if not self.staged_file.exists():
            return 0
        with open(self.staged_file, "r", encoding="utf-8") as f:
            return sum(1 for _ in f if _.strip())

    def load_staged_dataset(self) -> List[Dict[str, str]]:
        """Load all entries from JSONL staging buffer."""
        if not self.staged_file.exists():
            return []
        entries = []
        with open(self.staged_file, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if line_str:
                    try:
                        entries.append(json.loads(line_str))
                    except json.JSONDecodeError:
                        pass
        return entries

    def clear_staged_file(self) -> bool:
        """Clear the staged JSONL buffer after successful fine-tuning."""
        if self.staged_file.exists():
            self.staged_file.unlink()
            return True
        return False


dataset_curator = DatasetCurator()
__all__ = ["DatasetCurator", "dataset_curator"]
