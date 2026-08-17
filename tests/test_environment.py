from dataclasses import dataclass
import unittest

from environment import RCEnvironment


class FakeRobot:
    action_table = {0: (1, 1), 1: (0.5, 1), 2: (1, 0.5)}

    def __init__(self):
        self.applied = []
        self.physics_steps = 0

    def reset(self):
        self.physics_steps = 0

    def get_raw_sensor_values(self):
        return [0.0] * 8

    def get_observation(self, values):
        return [0.0] * 10

    def get_position(self):
        return (0.0, 0.0, 0.0)

    def apply_action(self, action):
        self.applied.append(action)

    def step(self):
        self.physics_steps += 1
        return 0

    def close(self):
        pass

    def update_hud(self, *args):
        pass


@dataclass
class FakeStatus:
    done: bool

    @property
    def reason(self):
        return "collision" if self.done else "running"

    def to_info(self):
        return {
            "termination_reason": self.reason,
            "is_success": False,
            "goal_distance_m": 1.0,
            "goal_center_distance_m": 1.0
        }


class FakeEpisodeManager:
    def __init__(self, done_at=None):
        self.calls = 0
        self.done_at = done_at

    def evaluate(self, values, step_count, stuck=False):
        self.calls += 1
        return FakeStatus(self.done_at == self.calls)

    @staticmethod
    def simulation_stopped_status():
        return FakeStatus(True)


class FakeReward:
    total = 1.0

    def to_info(self):
        return {"total": 1.0}


class FakeRewardCalculator:
    def calculate(self, *args, **kwargs):
        return FakeReward()


class FakeProgress:
    distance_reward = 0.0
    checkpoint_reward = 0.0

    def to_info(self):
        return {
            "progress_fraction": 0.0,
            "checkpoints_reached": 0,
            "checkpoint_count": 1
        }


class FakeProgressTracker:
    def reset(self, *args):
        return FakeProgress()

    def snapshot(self):
        return FakeProgress()

    def update(self, *args):
        return FakeProgress()

    def is_stuck(self, step_count):
        return False


class EnvironmentTests(unittest.TestCase):
    def create_environment(self, done_at=None):
        robot = FakeRobot()
        manager = FakeEpisodeManager(done_at)
        environment = RCEnvironment(
            robot=robot,
            episode_manager=manager,
            reward_calculator=FakeRewardCalculator(),
            progress_tracker=FakeProgressTracker()
        )
        environment.reset()
        return environment, robot, manager

    def test_action_repeat_accumulates_physics_rewards(self):
        environment, robot, manager = self.create_environment()

        _, reward, done, info = environment.step(1)

        self.assertEqual(robot.applied, [1])
        self.assertEqual(robot.physics_steps, 4)
        self.assertEqual(manager.calls, 4)
        self.assertEqual(reward, 4.0)
        self.assertFalse(done)
        self.assertEqual(info["episode_steps"], 4)
        self.assertEqual(info["decision_steps"], 1)

    def test_action_repeat_stops_at_terminal_physics_step(self):
        environment, robot, manager = self.create_environment(done_at=2)

        _, reward, done, info = environment.step(2)

        self.assertEqual(robot.applied, [2])
        self.assertEqual(robot.physics_steps, 2)
        self.assertEqual(reward, 2.0)
        self.assertTrue(done)
        self.assertEqual(info["action_repeat_executed"], 2)

if __name__ == "__main__":
    unittest.main()
