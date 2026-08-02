import unittest

from episode_manager import TerminationReason
from replay_buffer import create_replay_buffer, UniformReplayBuffer


def add_transition(buffer, action, done=False, reason="running"):
    buffer.add(
        state=[action, action + 0.5],
        action=action,
        reward=float(action),
        next_state=[action + 1, action + 1.5],
        done=done,
        termination_reason=reason
    )


class UniformReplayBufferTests(unittest.TestCase):
    def create_buffer(self, capacity=5, batch_size=2, learning_starts=3):
        return UniformReplayBuffer(
            capacity=capacity,
            batch_size=batch_size,
            learning_starts=learning_starts,
            seed=42
        )

    def test_warmup_readiness(self):
        buffer = self.create_buffer()

        add_transition(buffer, 0)
        add_transition(buffer, 1)
        self.assertFalse(buffer.is_ready)

        add_transition(buffer, 2)
        self.assertTrue(buffer.is_ready)

    def test_capacity_evicts_oldest_transition(self):
        buffer = self.create_buffer(
            capacity=3,
            batch_size=3,
            learning_starts=3
        )

        for action in range(4):
            add_transition(buffer, action)

        batch = buffer.sample(3)
        self.assertEqual(set(batch.actions), {1, 2, 3})

    def test_sample_returns_all_training_fields(self):
        buffer = self.create_buffer(batch_size=2, learning_starts=2)
        add_transition(buffer, 0)
        add_transition(
            buffer,
            1,
            done=True,
            reason=TerminationReason.GOAL_REACHED
        )

        batch = buffer.sample()

        self.assertEqual(len(batch), 2)
        self.assertIn(True, batch.dones)
        self.assertIn("goal_reached", batch.termination_reasons)

    def test_stored_states_are_copied(self):
        buffer = self.create_buffer(batch_size=1, learning_starts=1)
        state = [1.0, 2.0]
        buffer.add(state, 0, 1.0, [2.0, 3.0], False)

        state[0] = 99.0
        batch = buffer.sample()

        self.assertEqual(batch.states[0], (1.0, 2.0))

    def test_sampling_too_early_raises_clear_error(self):
        buffer = self.create_buffer()
        add_transition(buffer, 0)

        with self.assertRaisesRegex(ValueError, "Cannot sample"):
            buffer.sample()

    def test_configured_factory_creates_uniform_buffer(self):
        buffer = create_replay_buffer()

        self.assertIsInstance(buffer, UniformReplayBuffer)
        self.assertEqual(buffer.capacity, 50_000)
        self.assertEqual(buffer.batch_size, 64)
        self.assertEqual(buffer.learning_starts, 2_000)


if __name__ == "__main__":
    unittest.main()
