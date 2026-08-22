from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

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

    def test_double_dqn_selects_online_action_and_target_value(self):
        class FixedValues(nn.Module):
            def __init__(self, values):
                super().__init__()
                self.register_buffer("values", torch.tensor(values))

            def forward(self, states):
                return self.values.unsqueeze(0).expand(states.shape[0], -1)

        config = small_training_config()
        config["double_dqn"] = True
        agent = DQNAgent(10, 3, config)
        agent.online_network = FixedValues([0.0, 10.0, 1.0])
        agent.target_network = FixedValues([100.0, 2.0, 50.0])

        values = agent._next_state_values(torch.zeros((4, 10)))

        self.assertTrue(torch.equal(values, torch.full((4,), 2.0)))

    def test_reward_scale_is_applied_to_terminal_targets(self):
        config = small_training_config()
        config["gamma"] = 0.0
        config["reward_scale"] = 0.01
        agent = DQNAgent(10, 7, config)
        for index in range(4):
            agent.remember(
                [0.0] * 10,
                index,
                100.0,
                [0.0] * 10,
                True,
                "goal_reached"
            )

        metrics = agent.learn()

        self.assertAlmostEqual(metrics.mean_target, 1.0)

    def test_checkpoint_round_trip_restores_counters_and_weights(self):
        config = small_training_config()
        agent = DQNAgent(10, 7, config)
        agent.remember(
            [0.0] * 10, 2, 1.0, [0.1] * 10, False, "running"
        )
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
        self.assertEqual(len(restored.replay_buffer), 1)
        self.assertTrue(all(
            torch.equal(original, loaded)
            for original, loaded in zip(
                agent.online_network.parameters(),
                restored.online_network.parameters()
            )
        ))

    def test_policy_transfer_keeps_new_stage_memory_and_counters_fresh(self):
        config = small_training_config()
        source = DQNAgent(10, 7, config)
        source.environment_steps = 123
        source.training_steps = 45
        with torch.no_grad():
            for parameter in source.online_network.parameters():
                parameter.fill_(0.25)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "source.pt"
            source.save_checkpoint(checkpoint, episode=9)
            target = DQNAgent(10, 7, config)
            target.load_policy_weights(checkpoint)

        self.assertEqual(target.environment_steps, 0)
        self.assertEqual(target.training_steps, 0)
        self.assertEqual(len(target.replay_buffer), 0)
        self.assertTrue(all(
            torch.equal(original, loaded)
            for original, loaded in zip(
                source.online_network.parameters(),
                target.online_network.parameters()
            )
        ))

    def test_expert_imitation_teaches_demonstrated_action(self):
        config = small_training_config()
        config["expert_imitation_weight"] = 1.0
        config["learning_rate"] = 0.01
        agent = DQNAgent(10, 3, config)
        state = [0.0] * 10
        for _ in range(20):
            agent.remember(state, 2, 0.0, state, False, "running")
        for _ in range(30):
            agent.learn()

        self.assertEqual(agent.select_action(state, evaluate=True), 2)


if __name__ == "__main__":
    unittest.main()
