"""Goal detection and episode termination decisions for the RL environment."""

from dataclasses import dataclass
from enum import Enum
import math
from typing import Optional

from config import CONFIG


class TerminationReason(str, Enum):
    RUNNING = "running"
    COLLISION = "collision"
    GOAL_REACHED = "goal_reached"
    TIMEOUT = "timeout"
    STUCK = "stuck"
    SIMULATION_STOPPED = "simulation_stopped"


@dataclass(frozen=True)
class GoalMeasurement:
    reached: bool
    distance_to_region_m: float
    distance_to_center_m: float


@dataclass(frozen=True)
class EpisodeStatus:
    done: bool
    reason: TerminationReason
    goal_distance_m: Optional[float]
    goal_center_distance_m: Optional[float]

    @property
    def is_success(self):
        return self.reason == TerminationReason.GOAL_REACHED

    def to_info(self):
        return {
            "termination_reason": self.reason.value,
            "is_success": self.is_success,
            "goal_reached": self.is_success,
            "goal_distance_m": self.goal_distance_m,
            "goal_center_distance_m": self.goal_center_distance_m
        }


class GoalDetector:
    """Detect when the robot center enters the configured goal rectangle."""

    def __init__(self, supervisor, robot_node, goal_config):
        self.robot_node = robot_node
        self.goal_node = supervisor.getFromDef(goal_config["def"])

        if self.goal_node is None:
            raise RuntimeError(
                f'Could not find goal node DEF "{goal_config["def"]}".'
            )

        self.half_x = goal_config["size_m"]["x"] / 2.0
        self.half_y = goal_config["size_m"]["y"] / 2.0
        self.tolerance = goal_config["tolerance_m"]

    def measure(self):
        robot_position = self.robot_node.getPosition()
        goal_position = self.goal_node.getPosition()

        dx = abs(robot_position[0] - goal_position[0])
        dy = abs(robot_position[1] - goal_position[1])

        outside_x = max(0.0, dx - self.half_x)
        outside_y = max(0.0, dy - self.half_y)
        distance_to_region = math.hypot(outside_x, outside_y)

        reached = (
            dx <= self.half_x + self.tolerance
            and dy <= self.half_y + self.tolerance
        )

        return GoalMeasurement(
            reached=reached,
            distance_to_region_m=distance_to_region,
            distance_to_center_m=math.hypot(dx, dy)
        )


class EpisodeManager:
    """Evaluate collision, success, and timeout with deterministic priority."""

    def __init__(self, robot_controller):
        environment_config = CONFIG["environment"]
        self.collision_threshold = environment_config["collision_threshold"]
        self.max_steps = environment_config["max_steps"]

        goal_config = environment_config["goal"]
        self.goal_detector = None

        if goal_config["enabled"]:
            self.goal_detector = GoalDetector(
                robot_controller.robot,
                robot_controller.node,
                goal_config
            )

    def evaluate(self, raw_sensor_values, step_count, stuck=False):
        measurement = (
            self.goal_detector.measure()
            if self.goal_detector is not None
            else None
        )

        goal_distance = (
            measurement.distance_to_region_m
            if measurement is not None
            else None
        )
        goal_center_distance = (
            measurement.distance_to_center_m
            if measurement is not None
            else None
        )

        # Collision takes priority, so touching a wall cannot count as success.
        if max(raw_sensor_values) > self.collision_threshold:
            reason = TerminationReason.COLLISION
        elif measurement is not None and measurement.reached:
            reason = TerminationReason.GOAL_REACHED
        elif stuck:
            reason = TerminationReason.STUCK
        elif step_count >= self.max_steps:
            reason = TerminationReason.TIMEOUT
        else:
            reason = TerminationReason.RUNNING

        return EpisodeStatus(
            done=reason != TerminationReason.RUNNING,
            reason=reason,
            goal_distance_m=goal_distance,
            goal_center_distance_m=goal_center_distance
        )

    @staticmethod
    def simulation_stopped_status():
        return EpisodeStatus(
            done=True,
            reason=TerminationReason.SIMULATION_STOPPED,
            goal_distance_m=None,
            goal_center_distance_m=None
        )
