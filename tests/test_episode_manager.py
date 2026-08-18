import unittest

from config import CONFIG
from episode_manager import EpisodeManager, TerminationReason


class FakeNode:
    def __init__(self, position):
        self.position = position

    def getPosition(self):
        return self.position


class FakeSupervisor:
    def __init__(self, goal_node):
        self.goal_node = goal_node

    def getFromDef(self, name):
        return self.goal_node if name == "GOAL" else None


class FakeRobotController:
    def __init__(self, robot_position, goal_position=(0.0, 0.0, 0.0)):
        self.node = FakeNode(robot_position)
        self.robot = FakeSupervisor(FakeNode(goal_position))


class EpisodeManagerTests(unittest.TestCase):
    def create_manager(self, robot_position):
        return EpisodeManager(FakeRobotController(robot_position))

    def test_running_outside_goal(self):
        status = self.create_manager((1.0, 0.0, 0.0)).evaluate(
            [0.0] * 8,
            1
        )

        self.assertEqual(status.reason, TerminationReason.RUNNING)
        self.assertAlmostEqual(status.goal_distance_m, 0.15)

    def test_goal_reached(self):
        status = self.create_manager((0.87, 0.07, 0.0)).evaluate(
            [0.0] * 8,
            1
        )

        self.assertEqual(status.reason, TerminationReason.GOAL_REACHED)
        self.assertTrue(status.is_success)

    def test_collision_has_priority_over_goal(self):
        status = self.create_manager((0.0, 0.0, 0.0)).evaluate(
            [4000.0] + [0.0] * 7,
            1
        )

        self.assertEqual(status.reason, TerminationReason.COLLISION)
        self.assertFalse(status.is_success)

    def test_timeout_occurs_at_configured_step(self):
        status = self.create_manager((1.0, 0.0, 0.0)).evaluate(
            [0.0] * 8,
            CONFIG["environment"]["max_steps"]
        )

        self.assertEqual(status.reason, TerminationReason.TIMEOUT)

    def test_stuck_terminates_before_timeout(self):
        status = self.create_manager((1.0, 0.0, 0.0)).evaluate(
            [0.0] * 8,
            10,
            stuck=True
        )

        self.assertEqual(status.reason, TerminationReason.STUCK)

    def test_curriculum_target_is_a_success(self):
        status = self.create_manager((1.0, 0.0, 0.0)).evaluate(
            [0.0] * 8,
            10,
            curriculum_complete=True
        )

        self.assertEqual(status.reason, TerminationReason.CURRICULUM_COMPLETE)
        self.assertTrue(status.is_success)
        self.assertFalse(status.to_info()["goal_reached"])
        self.assertTrue(status.to_info()["curriculum_complete"])

    def test_collision_has_priority_over_curriculum_target(self):
        status = self.create_manager((1.0, 0.0, 0.0)).evaluate(
            [4000.0] + [0.0] * 7,
            10,
            curriculum_complete=True
        )

        self.assertEqual(status.reason, TerminationReason.COLLISION)

    def test_simulation_stopped_status(self):
        status = EpisodeManager.simulation_stopped_status()

        self.assertEqual(
            status.reason,
            TerminationReason.SIMULATION_STOPPED
        )
        self.assertTrue(status.done)


if __name__ == "__main__":
    unittest.main()
