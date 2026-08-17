import csv
from pathlib import Path
import tempfile
import unittest

from dashboard_config import public_fields, read_config, update_config
from dashboard_data import TrainingDataReader


SAMPLE_CONFIG = '''CONFIG = {
    "program": {"mode": "train"},
    "environment": {
        # Keep this comment and formatting.
        "max_steps": 3_000,
        "collision_threshold": 3900,
        "reward": {
            "collision": -100.0,
            "timeout": -50.0,
            "goal_base": 100.0,
            "goal_time_bonus": 50.0,
            "safe_motion_scale": 0.03,
            "danger_penalty_scale": 0.20,
            "time_penalty_start": 0.005,
            "time_penalty_growth": 0.020
        },
        "progress": {"checkpoint_reward": 5.0, "distance_reward_scale": 2.0},
        "stuck_detection": {"no_progress_steps": 400}
    },
    "robot": {"drive": {"speed_scale": 0.95}},
    "training": {
        "episodes": 10000,
        "gamma": 0.99,
        "learning_rate": 0.001,
        "action_repeat": 4,
        "double_dqn": True,
        "reward_scale": 0.01,
        "epsilon": {"start": 1.0, "end": 0.05, "decay_steps": 50000},
        "target_update_steps": 1000,
        "save_every_episodes": 50,
        "resume": True,
        "replay_buffer": {
            "capacity": 50000,
            "batch_size": 64,
            "learning_starts": 2000
        }
    }
}
'''


class DashboardConfigTests(unittest.TestCase):
    def test_selected_values_are_updated_without_reformatting_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.py"
            path.write_text(SAMPLE_CONFIG, encoding="utf-8")

            update_config(path, {
                "environment.max_steps": 2500,
                "training.learning_rate": 0.0003,
                "training.resume": False
            })

            source = path.read_text(encoding="utf-8")
            config = read_config(path)
            self.assertIn("# Keep this comment and formatting.", source)
            self.assertEqual(config["environment"]["max_steps"], 2500)
            self.assertEqual(config["training"]["learning_rate"], 0.0003)
            self.assertFalse(config["training"]["resume"])

    def test_invalid_replay_relationship_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.py"
            path.write_text(SAMPLE_CONFIG, encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "warmup"):
                update_config(path, {
                    "training.replay_buffer.batch_size": 3000
                })

    def test_public_fields_excludes_unapproved_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.py"
            path.write_text(SAMPLE_CONFIG, encoding="utf-8")

            fields = public_fields(path)

        paths = {field["path"] for field in fields}
        self.assertIn("environment.max_steps", paths)
        self.assertNotIn("robot.drive.action_ratios", paths)


class DashboardDataTests(unittest.TestCase):
    def test_snapshot_reports_windows_trend_and_history(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training.csv"
            fieldnames = (
                "episode", "steps", "total_reward", "termination_reason",
                "success", "epsilon", "buffer_size", "training_steps", "loss"
            )
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                for episode in range(1, 41):
                    writer.writerow({
                        "episode": episode,
                        "steps": 100,
                        "total_reward": -100 + episode,
                        "termination_reason": "goal_reached" if episode == 40 else "collision",
                        "success": episode == 40,
                        "epsilon": 0.2,
                        "buffer_size": episode,
                        "training_steps": episode,
                        "loss": 1.0 / episode
                    })

            snapshot = TrainingDataReader(path).snapshot()

        self.assertTrue(snapshot["has_data"])
        self.assertEqual(snapshot["latest"]["episode"], 40)
        self.assertGreater(snapshot["reward_trend_20"], 0)
        self.assertEqual(snapshot["windows"]["100"]["success_rate"], 2.5)
        self.assertEqual(len(snapshot["recent_rows"]), 12)


if __name__ == "__main__":
    unittest.main()
