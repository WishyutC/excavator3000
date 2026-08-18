from copy import deepcopy
import unittest

from config import CONFIG
from pathlib import Path
import tempfile

from curriculum_trainer import (
    _last_logged_episode,
    _stage_training_config,
    _success_rate
)


class FakeSummary:
    def __init__(self, success):
        self.success = success


class CurriculumTrainerTests(unittest.TestCase):
    def test_stage_config_uses_fresh_training_state_and_own_directory(self):
        base = deepcopy(CONFIG["training"])
        base["resume"] = True
        stage = base["curriculum"]["stages"][1]

        configured = _stage_training_config(base, stage, 2)

        self.assertFalse(configured["resume"])
        self.assertEqual(configured["episodes"], stage["maximum_episodes"])
        self.assertEqual(
            configured["epsilon"]["start"],
            stage["epsilon_start"]
        )
        self.assertTrue(
            configured["save_directory"].endswith(stage["name"])
        )
        self.assertEqual(configured["seed"], base["seed"] + 2)
        self.assertIsNone(configured["forced_action"])
        self.assertEqual(configured["expert_policy"], "sensor")
        self.assertEqual(configured["expert_imitation_weight"], 1.0)
        self.assertTrue(base["resume"])

    def test_success_rate_counts_curriculum_and_goal_success_flags(self):
        summaries = [FakeSummary(True), FakeSummary(False), FakeSummary(True)]

        self.assertAlmostEqual(_success_rate(summaries), 2 / 3)
        self.assertEqual(_success_rate([]), 0.0)

    def test_last_logged_episode_supports_curriculum_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training.csv"
            path.write_text("episode,success\n10,True\n25,False\n", encoding="utf-8")

            self.assertEqual(_last_logged_episode(path), 25)


if __name__ == "__main__":
    unittest.main()
