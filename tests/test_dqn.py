from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import torch

from config import CONFIG
from dqn_agent import DQNAgent
from dqn_network import DQNNetwork


def small_training_config():
    config = deepcopy(CONFIG["training"])
    config.update({
        "device": "cpu",
        "hidden_sizes": [16, 16],
        "target_update_steps": 1,
        "train_every_steps": 1
    })
    config["epsilon"] = {"start": 1.0, "end": 0.0, "decay_steps": 10}
    config["replay_buffer"] = {
        "type": "uniform",
        "capacity": 20,
        "batch_size": 4,
        "learning_starts": 4
    }
    return config


class DQNTests(unittest.TestCase):
    def test_network_output_matches_action_count(self):
        network = DQNNetwork(10, 7, (16, 16))
        output = network(torch.zeros((3, 10)))

        self.assertEqual(tuple(output.shape), (3, 7))
        self.assertGreater(network.parameter_count, 0)

    def test_epsilon_decays_linearly(self):
        agent = DQNAgent(10, 7, small_training_config())
        self.assertEqual(agent.epsilon, 1.0)

        agent.environment_steps = 5
        self.assertAlmostEqual(agent.epsilon, 0.5)

        agent.environment_steps = 20
        self.assertEqual(agent.epsilon, 0.0)

    def test_greedy_action_uses_highest_q_value(self):
        config = small_training_config()
        config["epsilon"] = {"start": 0.0, "end": 0.0, "decay_steps": 1}
        agent = DQNAgent(10, 7, config)

        with torch.no_grad():
            for parameter in agent.online_network.parameters():
                parameter.zero_()
            agent.online_network.model[-1].bias[3] = 2.0

        self.assertEqual(agent.select_action([0.0] * 10), 3)

    def test_learning_updates_network_and_syncs_target(self):
        agent = DQNAgent(10, 7, small_training_config())

        for index in range(4):
            state = [index / 10.0] * 10
            agent.remember(
                state,
                index % 7,
                1.0,
                [value + 0.1 for value in state],
                index == 3,
                "goal_reached" if index == 3 else "running"
            )

        before = [
            parameter.detach().clone()
            for parameter in agent.online_network.parameters()
        ]
        metrics = agent.learn()

        self.assertIsNotNone(metrics)
        self.assertEqual(metrics.training_steps, 1)
        self.assertTrue(any(
            not torch.equal(old, new)
            for old, new in zip(before, agent.online_network.parameters())
        ))
        self.assertTrue(all(
            torch.equal(online, target)
            for online, target in zip(
                agent.online_network.parameters(),
                agent.target_network.parameters()
            )
        ))

    def test_checkpoint_round_trip_restores_counters_and_weights(self):
        config = small_training_config()
        agent = DQNAgent(10, 7, config)
        agent.environment_steps = 123
        agent.training_steps = 45

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "agent.pt"
            agent.save_checkpoint(checkpoint, episode=9)

            restored = DQNAgent(10, 7, config)
            metadata = restored.load_checkpoint(checkpoint)

        self.assertEqual(metadata["episode"], 9)
        self.assertEqual(restored.environment_steps, 123)
        self.assertEqual(restored.training_steps, 45)
        self.assertTrue(all(
            torch.equal(original, loaded)
            for original, loaded in zip(
                agent.online_network.parameters(),
                restored.online_network.parameters()
            )
        ))


if __name__ == "__main__":
    unittest.main()
