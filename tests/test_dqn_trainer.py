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
    action_size = 7

    def __init__(self):
        self.steps = 0
        self.hud_updates = []

    def reset(self):
        self.steps = 0
        return [0.0] * 10

    def step(self, action):
        self.steps += 1
        done = self.steps == 2
        info = {
            "termination_reason": "goal_reached" if done else "running",
            "is_success": done,
            "reward_breakdown": {}
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
            agent = DQNAgent(10, 7, config)
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

    def test_evaluation_does_not_store_or_learn(self):
        with tempfile.TemporaryDirectory() as directory:
            config = trainer_config(directory)
            environment = FakeEnvironment()
            agent = DQNAgent(10, 7, config)
            trainer = DQNTrainer(environment, agent, config, MemoryLogger())

            summaries = trainer.evaluate(episodes=2)

        self.assertEqual(len(summaries), 2)
        self.assertEqual(len(agent.replay_buffer), 0)
        self.assertEqual(agent.training_steps, 0)


if __name__ == "__main__":
    unittest.main()
