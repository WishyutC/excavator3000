from dataclasses import dataclass
import unittest
import math

from environment import RCEnvironment


class FakeRobot:
    action_table = {0: (1, 1), 1: (0.5, 1), 2: (1, 0.5)}

    def __init__(self):
        self.applied = []
        self.physics_steps = 0
        self.forward_direction = (1.0, 0.0)
        self.current_action = 0
        self.drive_ratios = []

    def reset(self):
        self.physics_steps = 0

    def get_raw_sensor_values(self):
        return [0.0] * 8

    def get_observation(self, values):
        return [0.0] * 10

    def get_position(self):
        return (0.0, 0.0, 0.0)

    def get_forward_direction(self):
        return self.forward_direction

    def apply_action(self, action):
        self.applied.append(action)
        self.current_action = action

    def apply_drive_ratios(self, left, right):
        self.drive_ratios.append((left, right))
        self.current_action = 0

    def step(self):
        self.physics_steps += 1
        angle = math.atan2(
            self.forward_direction[1], self.forward_direction[0]
        )
        if self.current_action == 1:
            angle += 0.15
        elif self.current_action == 2:
            angle -= 0.15
        self.forward_direction = (math.cos(angle), math.sin(angle))
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

    def evaluate(
        self,
        values,
        step_count,
        stuck=False,
        curriculum_complete=False
    ):
        self.calls += 1
        return FakeStatus(curriculum_complete or self.done_at == self.calls)

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
    checkpoints_reached = 0

    def to_info(self):
        return {
            "progress_fraction": 0.0,
            "checkpoints_reached": 0,
            "checkpoint_count": 1
        }


class FakeProgressTracker:
    current_index = 0
    waypoints = [(1.0, 0.0)]

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

        _, reward, done, info = environment.step(0)

        self.assertEqual(robot.applied, [0])
        self.assertEqual(robot.physics_steps, 4)
        self.assertEqual(manager.calls, 4)
        self.assertEqual(reward, 4.0)
        self.assertFalse(done)
        self.assertEqual(info["episode_steps"], 4)
        self.assertEqual(info["decision_steps"], 1)

    def test_turn_macro_commits_then_clears_corner(self):
        environment, robot, manager = self.create_environment()

        _, reward, done, info = environment.step(1)

        self.assertEqual(robot.applied, [1, 0])
        self.assertEqual(manager.calls, robot.physics_steps)
        self.assertEqual(reward, float(robot.physics_steps))
        self.assertFalse(done)
        self.assertTrue(info["turn_macro_used"])
        self.assertGreaterEqual(info["turn_macro_turn_steps"], 10)
        self.assertEqual(info["turn_macro_exit_steps"], 560)
        self.assertEqual(len(robot.drive_ratios), 560)

    def test_macro_centering_steers_away_from_right_wall(self):
        environment, robot, _ = self.create_environment()
        observation = [0.0] * 10
        observation[3] = 0.8

        robot.forward_direction = (0.0, 1.0)
        integral = environment._apply_macro_centering(
            observation,
            (1.0, 0.0),
            1,
            desired_angle_magnitude=1.57
        )

        left, right = robot.drive_ratios[-1]
        self.assertLess(left, right)
        self.assertGreater(integral, 0.0)

    def test_action_repeat_stops_at_terminal_physics_step(self):
        environment, robot, manager = self.create_environment(done_at=2)

        _, reward, done, info = environment.step(2)

        self.assertEqual(robot.applied, [2])
        self.assertEqual(robot.physics_steps, 2)
        self.assertEqual(reward, 2.0)
        self.assertTrue(done)
        self.assertEqual(info["action_repeat_executed"], 2)

    def test_waypoint_expert_turns_toward_training_target(self):
        environment, _, _ = self.create_environment()

        environment.progress_tracker.waypoints = [(0.0, 1.0)]
        self.assertEqual(environment.waypoint_expert_action(), 1)
        environment.progress_tracker.waypoints = [(0.0, -1.0)]
        self.assertEqual(environment.waypoint_expert_action(), 2)
        environment.progress_tracker.waypoints = [(1.0, 0.0)]
        self.assertEqual(environment.waypoint_expert_action(), 0)

    def test_sensor_expert_drives_clear_and_turns_to_open_side(self):
        environment, robot, _ = self.create_environment()
        clear = [0.0] * 10
        blocked = clear.copy()
        blocked[0] = 0.8

        self.assertEqual(environment.sensor_expert_action(clear), 0)
        environment.progress_tracker.current_index = 1
        self.assertEqual(environment.sensor_expert_action(blocked), 1)
        self.assertIsNone(environment._sensor_expert_turn_action)
        self.assertEqual(environment._sensor_expert_exit_checkpoint, 1)
        environment.reset()
        environment.progress_tracker.current_index = 2
        self.assertEqual(environment.sensor_expert_action(blocked), 0)
        environment.progress_tracker.current_index = 3
        environment.progress_tracker.waypoints = [
            (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, -1.0)
        ]
        self.assertEqual(environment.sensor_expert_action(blocked), 2)
        robot.forward_direction = (0.0, -1.0)
        self.assertEqual(environment.sensor_expert_action(clear), 0)
        self.assertEqual(environment.sensor_expert_action(blocked), 0)

if __name__ == "__main__":
    unittest.main()
