"""J.A.R.V.I.S. Neural Experiment Tracker (Phase 25).

Logs and persists neural network research experiments:
- Architectural sweeps (e.g. MHA vs GQA, RMSNorm vs LayerNorm, SwiGLU vs GELU)
- Hyperparameter records (learning rate, batch size, sequence length)
- Training trajectories (loss curves, perplexity)
- Checkpoint associations and metadata
- Persists to JSON (experiments.json) and Markdown (EXPERIMENT_LOG.md)
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class ExperimentTracker:
    """Manages experiment persistence, telemetry, and reporting."""

    def __init__(self, experiment_dir: Optional[str | Path] = None):
        if experiment_dir is None:
            self.experiment_dir = Path(__file__).resolve().parents[1] / "experiments"
        else:
            self.experiment_dir = Path(experiment_dir)

        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.json_file = self.experiment_dir / "experiments.json"
        self.md_file = self.experiment_dir / "EXPERIMENT_LOG.md"

        self._ensure_storage()

    def _ensure_storage(self) -> None:
        if not self.json_file.exists():
            with open(self.json_file, "w", encoding="utf-8") as f:
                json.dump([], f, indent=2)

    def log_experiment(
        self,
        experiment_id: str,
        architecture_type: str,
        config: Dict[str, Any],
        metrics: Dict[str, Any],
        notes: str = "",
        checkpoint_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Record an experiment run to persistent storage."""
        timestamp = datetime.now(timezone.utc).isoformat()
        entry = {
            "experiment_id": experiment_id,
            "architecture_type": architecture_type,
            "timestamp": timestamp,
            "config": config,
            "metrics": metrics,
            "notes": notes,
            "checkpoint_path": checkpoint_path,
        }

        # Update JSON database
        experiments = self.list_experiments()
        # Replace if existing or append
        existing_idx = next((i for i, e in enumerate(experiments) if e["experiment_id"] == experiment_id), None)
        if existing_idx is not None:
            experiments[existing_idx] = entry
        else:
            experiments.append(entry)

        with open(self.json_file, "w", encoding="utf-8") as f:
            json.dump(experiments, f, indent=2)

        # Update Markdown log
        self._update_markdown_log(experiments)
        return entry

    def list_experiments(self) -> List[Dict[str, Any]]:
        """Retrieve all recorded experiment runs."""
        try:
            with open(self.json_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []

    def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve details of a specific experiment."""
        for exp in self.list_experiments():
            if exp["experiment_id"] == experiment_id:
                return exp
        return None

    def _update_markdown_log(self, experiments: List[Dict[str, Any]]) -> None:
        """Regenerate Markdown log summarizing all experiments."""
        lines = [
            "# J.A.R.V.I.S. Neural Architecture Experiment Log",
            "Automated tracking of neural network research, ablation studies, and benchmarks.",
            "",
            "| Experiment ID | Architecture | Loss | PPL | Latency (Mean) | Timestamp | Notes |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ]

        for exp in sorted(experiments, key=lambda x: x.get("timestamp", ""), reverse=True):
            exp_id = exp.get("experiment_id", "")
            arch = exp.get("architecture_type", "")
            metrics = exp.get("metrics", {})
            loss = metrics.get("loss", "N/A")
            ppl = metrics.get("perplexity", "N/A")
            lat = metrics.get("mean_latency_ms", metrics.get("mean_ms", "N/A"))
            ts = exp.get("timestamp", "")[:19].replace("T", " ")
            notes = exp.get("notes", "").replace("|", "/")
            lines.append(f"| `{exp_id}` | {arch} | {loss} | {ppl} | {lat} ms | {ts} | {notes} |")

        with open(self.md_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")


experiment_tracker = ExperimentTracker()
