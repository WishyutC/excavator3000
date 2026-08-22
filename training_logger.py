"""CSV episode logging for long-running DQN experiments."""

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import platform

from config import CONFIG


class TrainingLogger:
    FIELDNAMES = (
        "episode",
        "stage_episode",
        "curriculum_stage",
        "curriculum_target_checkpoint",
        "map_name",
        "steps",
        "decisions",
        "total_reward",
        "mean_reward_per_step",
        "termination_reason",
        "success",
        "epsilon",
        "buffer_size",
        "training_steps",
        "loss",
        "min_goal_distance_m",
        "final_goal_distance_m",
        "track_progress",
        "checkpoints_reached",
        "checkpoint_count",
        "action_forward_pct",
        "action_left_pct",
        "action_right_pct"
    )

    def __init__(self, path=None, enabled=None):
        logging_config = CONFIG["logging"]
        self.enabled = (
            logging_config["enabled"] if enabled is None else bool(enabled)
        )
        self.path = Path(path or logging_config["directory"]) / "training.csv"
        self.manifest_path = self.path.parent / "run_manifest.json"

        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._write_manifest(logging_config)
            if not self.path.exists():
                with self.path.open("w", newline="", encoding="utf-8") as file:
                    csv.DictWriter(file, fieldnames=self.FIELDNAMES).writeheader()

    def _write_manifest(self, logging_config):
        """Capture the effective run configuration once for reproducibility."""
        if self.manifest_path.exists():
            return

        manifest = {
            "schema_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "run_name": logging_config.get("run_name", self.path.parent.name),
            "description": logging_config.get("description", ""),
            "baseline_reference": logging_config.get(
                "baseline_reference"
            ),
            "python_version": platform.python_version(),
            "csv_file": self.path.name,
            "csv_fields": list(self.FIELDNAMES),
            "configuration": CONFIG
        }
        self.manifest_path.write_text(
            json.dumps(manifest, indent=2, allow_nan=False),
            encoding="utf-8"
        )

    def log_episode(self, values):
        if not self.enabled:
            return

        row = {name: values.get(name) for name in self.FIELDNAMES}
        with self.path.open("a", newline="", encoding="utf-8") as file:
            csv.DictWriter(file, fieldnames=self.FIELDNAMES).writerow(row)
