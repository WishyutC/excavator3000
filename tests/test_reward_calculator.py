import unittest

from config import CONFIG
from episode_manager import TerminationReason
from reward_calculator import RewardCalculator


class RewardCalculatorTests(unittest.TestCase):
    def setUp(self):
        self.calculator = RewardCalculator()
        self.clear_observation = [0.0] * 10

    def test_goal_reward_decays_from_150_to_100(self):
        maximum_steps = CONFIG["environment"]["max_steps"]
        early = self.calculator.calculate(
            [],
            TerminationReason.GOAL_REACHED,
            0
        )
        halfway = self.calculator.calculate(
            [],
            TerminationReason.GOAL_REACHED,
            maximum_steps // 2
        )
        late = self.calculator.calculate(
            [],
            TerminationReason.GOAL_REACHED,
            maximum_steps
        )

        self.assertEqual(early.total, 150.0)
        self.assertEqual(halfway.total, 125.0)
        self.assertEqual(late.total, 100.0)

    def test_collision_and_timeout_have_fixed_penalties(self):
        collision = self.calculator.calculate(
            [],
            TerminationReason.COLLISION,
            10
        )
        timeout = self.calculator.calculate(
            [],
            TerminationReason.TIMEOUT,
            CONFIG["environment"]["max_steps"]
        )

        self.assertEqual(collision.total, -100.0)
        self.assertEqual(timeout.total, -50.0)

    def test_waiting_is_always_negative(self):
        reward = self.calculator.calculate(
            self.clear_observation,
            TerminationReason.RUNNING,
            0
        )

        self.assertAlmostEqual(reward.time, -0.005)
        self.assertAlmostEqual(reward.stuck, -0.020)
        self.assertAlmostEqual(reward.total, -0.025)

    def test_safe_forward_motion_earns_only_small_reward(self):
        observation = self.clear_observation.copy()
        observation[8] = 1.0

        reward = self.calculator.calculate(
            observation,
            TerminationReason.RUNNING,
            0
        )

        self.assertAlmostEqual(reward.safe_motion, 0.030)
        self.assertAlmostEqual(reward.total, 0.025)
        self.assertLess(reward.total, 0.03)

    def test_fast_motion_near_wall_is_strongly_negative(self):
        observation = self.clear_observation.copy()
        observation[0] = 1.0
        observation[8] = 1.0

        reward = self.calculator.calculate(
            observation,
            TerminationReason.RUNNING,
            0
        )

        self.assertEqual(reward.safe_motion, 0.0)
        self.assertAlmostEqual(reward.danger, -0.4)
        self.assertAlmostEqual(reward.total, -0.405)

    def test_time_penalty_increases_during_episode(self):
        start = self.calculator.calculate(
            self.clear_observation,
            TerminationReason.RUNNING,
            0
        )
        end = self.calculator.calculate(
            self.clear_observation,
            TerminationReason.RUNNING,
            CONFIG["environment"]["max_steps"]
        )

        self.assertAlmostEqual(start.time, -0.005)
        self.assertAlmostEqual(end.time, -0.025)
        self.assertLess(end.total, start.total)

    def test_reward_breakdown_sums_to_total(self):
        observation = self.clear_observation.copy()
        observation[4] = 0.5
        observation[8] = 0.4

        reward = self.calculator.calculate(
            observation,
            "running",
            100
        )
        parts = reward.to_info()
        component_sum = sum(
            parts[name]
            for name in ("terminal", "safe_motion", "danger", "time", "stuck")
        )

        self.assertAlmostEqual(parts["total"], component_sum)

    def test_safe_driving_until_timeout_still_has_negative_return(self):
        observation = self.clear_observation.copy()
        observation[8] = 1.0
        maximum_steps = CONFIG["environment"]["max_steps"]

        episode_return = sum(
            self.calculator.calculate(
                observation,
                TerminationReason.RUNNING,
                step
            ).total
            for step in range(1, maximum_steps)
        )
        episode_return += self.calculator.calculate(
            observation,
            TerminationReason.TIMEOUT,
            maximum_steps
        ).total

        self.assertLess(episode_return, 0.0)
        self.assertLess(episode_return, 100.0)


if __name__ == "__main__":
    unittest.main()
