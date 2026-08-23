from copy import deepcopy
import unittest

from config import CONFIG
from pathlib import Path
import tempfile

from curriculum_trainer import (
    _evaluate_saved_candidates,
    _last_logged_episode,
    _stage_training_config,
    _success_rate
)


class FakeSummary:
    def __init__(self, success, reward=0.0, progress=0.0):
        self.success = success
        self.total_reward = reward
        self.track_progress = progress


class FakeAgent:
    def __init__(self):
        self.loaded = None

    def load_checkpoint(self, path, load_optimizer=False):
        self.loaded = Path(path)


class FakeTrainer:
    def __init__(self, agent, results):
        self.agent = agent
        self.results = results

    def evaluate(self, episodes, verbose=False):
        return self.results[self.agent.loaded.name]


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

    def test_saved_candidate_gate_selects_reloadable_best_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "latest.pt").touch()
            (root / "best.pt").touch()
            config = {
                "save_directory": str(root),
                "checkpoint_name": "latest.pt",
                "best_checkpoint_name": "best.pt"
            }
            agent = FakeAgent()
            trainer = FakeTrainer(agent, {
                "latest.pt": [FakeSummary(False, -10.0, 0.1)],
                "best.pt": [FakeSummary(True, 50.0, 1.0)]
            })

            selected, summaries, checks = _evaluate_saved_candidates(
                agent, trainer, config, 1
            )

            self.assertEqual(selected.name, "best.pt")
            self.assertTrue(summaries[0].success)
            self.assertEqual(len(checks), 2)
            self.assertEqual(agent.loaded, selected)


if __name__ == "__main__":
    unittest.main()
