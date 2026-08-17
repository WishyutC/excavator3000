"""Ordered, training-only track progress for dense reward and stuck detection."""

from dataclasses import asdict, dataclass
import math

from config import CONFIG


def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


@dataclass(frozen=True)
class ProgressUpdate:
    reward: float
    distance_reward: float
    checkpoint_reward: float
    checkpoints_reached: int
    checkpoint_count: int
    target_distance_m: float | None
    progress_fraction: float
    made_progress: bool

    def to_info(self):
        return asdict(self)


class ProgressTracker:
    """Reward ordered waypoint progress without exposing map data to the policy."""

    def __init__(self, environment_config=None):
        environment = environment_config or CONFIG["environment"]
        self.config = environment["progress"]
        self.stuck_config = environment["stuck_detection"]
        self.enabled = bool(self.config["enabled"])
        self.waypoints = [tuple(map(float, point)) for point in self.config["waypoints"]]
        if self.enabled and not self.waypoints:
            raise ValueError("Progress tracking requires at least one waypoint.")
        self.current_index = 0
        self.previous_distance = None
        self.best_distance = None
        self.last_progress_step = 0

    @staticmethod
    def _distance(position, target):
        return math.hypot(position[0] - target[0], position[1] - target[1])

    def reset(self, position, step_count=0):
        self.current_index = 0
        self.last_progress_step = int(step_count)
        if self.enabled:
            distance = self._distance(position, self.waypoints[0])
            self.previous_distance = distance
            self.best_distance = distance
        else:
            self.previous_distance = None
            self.best_distance = None
        return self.snapshot()

    def update(self, position, step_count):
        if not self.enabled or self.current_index >= len(self.waypoints):
            return self.snapshot()

        target = self.waypoints[self.current_index]
        distance = self._distance(position, target)
        previous = distance if self.previous_distance is None else self.previous_distance
        delta = clamp(
            previous - distance,
            -float(self.config["distance_delta_clip_m"]),
            float(self.config["distance_delta_clip_m"])
        )
        distance_reward = delta * float(self.config["distance_reward_scale"])
        checkpoint_reward = 0.0
        made_progress = False

        minimum_progress = float(self.stuck_config["minimum_progress_m"])
        if self.best_distance is None or distance <= self.best_distance - minimum_progress:
            self.best_distance = distance
            self.last_progress_step = int(step_count)
            made_progress = True

        if distance <= float(self.config["checkpoint_radius_m"]):
            checkpoint_reward = float(self.config["checkpoint_reward"])
            self.current_index += 1
            self.last_progress_step = int(step_count)
            made_progress = True
            if self.current_index < len(self.waypoints):
                next_distance = self._distance(position, self.waypoints[self.current_index])
                self.previous_distance = next_distance
                self.best_distance = next_distance
            else:
                self.previous_distance = None
                self.best_distance = None
        else:
            self.previous_distance = distance

        return self.snapshot(
            distance_reward=distance_reward,
            checkpoint_reward=checkpoint_reward,
            made_progress=made_progress
        )

    def is_stuck(self, step_count):
        if not self.enabled or not self.stuck_config["enabled"]:
            return False
        return (
            int(step_count) - self.last_progress_step
            >= int(self.stuck_config["no_progress_steps"])
        )

    def snapshot(self, distance_reward=0.0, checkpoint_reward=0.0, made_progress=False):
        target_distance = self.previous_distance
        checkpoint_count = len(self.waypoints)
        fraction = (
            self.current_index / checkpoint_count
            if checkpoint_count
            else 0.0
        )
        return ProgressUpdate(
            reward=float(distance_reward + checkpoint_reward),
            distance_reward=float(distance_reward),
            checkpoint_reward=float(checkpoint_reward),
            checkpoints_reached=self.current_index,
            checkpoint_count=checkpoint_count,
            target_distance_m=target_distance,
            progress_fraction=float(fraction),
            made_progress=bool(made_progress)
        )
