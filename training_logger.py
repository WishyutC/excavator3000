"""CSV episode logging for long-running DQN experiments."""

import csv
from pathlib import Path

from config import CONFIG


class TrainingLogger:
    FIELDNAMES = (
        "episode",
        "stage_episode",
        "curriculum_stage",
        "curriculum_target_checkpoint",
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

        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if not self.path.exists():
                with self.path.open("w", newline="", encoding="utf-8") as file:
                    csv.DictWriter(file, fieldnames=self.FIELDNAMES).writeheader()

    def log_episode(self, values):
        if not self.enabled:
            return

        row = {name: values.get(name) for name in self.FIELDNAMES}
        with self.path.open("a", newline="", encoding="utf-8") as file:
            csv.DictWriter(file, fieldnames=self.FIELDNAMES).writerow(row)
