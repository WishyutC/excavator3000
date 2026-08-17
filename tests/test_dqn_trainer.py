from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from config import CONFIG
from dqn_agent import DQNAgent
from dqn_trainer import DQNTrainer


class MemoryLogger:
    def __init__(self):
        self.rows = []

    def log_episode(self, values):
        self.rows.append(values)


class FakeEnvironment:
    state_size = 10
    action_size = 3

    def __init__(self, success=True):
        self.steps = 0
        self.hud_updates = []
        self.success = success

    def reset(self):
        self.steps = 0
        return [0.0] * 10

    def step(self, action):
        self.steps += 1
        done = self.steps == 2
        info = {
            "termination_reason": (
                "goal_reached" if done and self.success
                else "collision" if done
                else "running"
            ),
            "is_success": done and self.success,
            "reward_breakdown": {},
            "goal_distance_m": 0.0 if done and self.success else 1.0,
            "episode_steps": self.steps,
            "progress": {
                "progress_fraction": 1.0 if done else 0.5,
                "checkpoints_reached": 2 if done else 1,
                "checkpoint_count": 2
            }
        }
        return [0.1] * 10, 2.0 if done else 0.1, done, info

    def update_hud(self, *values):
        self.hud_updates.append(values)


def trainer_config(directory):
    config = deepcopy(CONFIG["training"])
    config.update({
        "device": "cpu",
        "episodes": 2,
        "hidden_sizes": [8, 8],
        "save_directory": str(directory),
        "save_every_episodes": 1,
        "target_update_steps": 1,
        "train_every_steps": 1
    })
    config["epsilon"] = {"start": 0.0, "end": 0.0, "decay_steps": 1}
    config["replay_buffer"] = {
        "type": "uniform",
        "capacity": 20,
        "batch_size": 4,
        "learning_starts": 4
    }
    return config


class DQNTrainerTests(unittest.TestCase):
    def test_training_runs_episodes_logs_and_saves(self):
        with tempfile.TemporaryDirectory() as directory:
            config = trainer_config(directory)
            environment = FakeEnvironment()
            agent = DQNAgent(10, 3, config)
            logger = MemoryLogger()
            trainer = DQNTrainer(environment, agent, config, logger)

            summaries = trainer.train(episodes=2)

            self.assertEqual(len(summaries), 2)
            self.assertTrue(all(summary.success for summary in summaries))
            self.assertEqual(len(logger.rows), 2)
            self.assertEqual(len(environment.hud_updates), 4)
            self.assertTrue(
                (Path(directory) / config["checkpoint_name"]).exists()
            )
            self.assertTrue(
                (Path(directory) / config["best_checkpoint_name"]).exists()
            )
            self.assertGreater(agent.training_steps, 0)
            self.assertAlmostEqual(
                summaries[-1].action_forward_pct
                + summaries[-1].action_left_pct
                + summaries[-1].action_right_pct,
                100.0
            )
            self.assertEqual(summaries[-1].steps, 2)
            self.assertEqual(summaries[-1].decisions, 2)

    def test_failed_episode_saves_candidate_but_not_best(self):
        with tempfile.TemporaryDirectory() as directory:
            config = trainer_config(directory)
            environment = FakeEnvironment(success=False)
            agent = DQNAgent(10, 3, config)
            trainer = DQNTrainer(environment, agent, config, MemoryLogger())

            summaries = trainer.train(episodes=1)

            self.assertFalse(summaries[0].success)
            self.assertTrue(
                (Path(directory) / config["candidate_checkpoint_name"]).exists()
            )
            self.assertFalse(
                (Path(directory) / config["best_checkpoint_name"]).exists()
            )

    def test_evaluation_does_not_store_or_learn(self):
        with tempfile.TemporaryDirectory() as directory:
            config = trainer_config(directory)
            environment = FakeEnvironment()
            agent = DQNAgent(10, 3, config)
            trainer = DQNTrainer(environment, agent, config, MemoryLogger())

            summaries = trainer.evaluate(episodes=2)

        self.assertEqual(len(summaries), 2)
        self.assertEqual(len(agent.replay_buffer), 0)
        self.assertEqual(agent.training_steps, 0)


if __name__ == "__main__":
    unittest.main()
