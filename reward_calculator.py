"""Modular reward shaping for safe and efficient track driving."""

from dataclasses import asdict, dataclass

from config import CONFIG
from episode_manager import TerminationReason


def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


@dataclass(frozen=True)
class RewardResult:
    total: float
    terminal: float = 0.0
    safe_motion: float = 0.0
    danger: float = 0.0
    time: float = 0.0
    stuck: float = 0.0
    progress: float = 0.0
    checkpoint: float = 0.0

    def to_info(self):
        return asdict(self)


class RewardCalculator:
    """Calculate bounded shaping rewards without rewarding mere survival."""

    def __init__(self, environment_config=None):
        if environment_config is None:
            environment_config = CONFIG["environment"]

        self.max_steps = environment_config["max_steps"]
        self.config = environment_config["reward"]

        if self.max_steps <= 0:
            raise ValueError("environment.max_steps must be greater than zero.")

    @staticmethod
    def _reason_value(termination_reason):
        if isinstance(termination_reason, TerminationReason):
            return termination_reason.value
        return str(termination_reason)

    def calculate(
        self,
        observation,
        termination_reason,
        step_count,
        progress_reward=0.0,
        checkpoint_reward=0.0
    ):
        progress = clamp(step_count / self.max_steps, 0.0, 1.0)
        reason = self._reason_value(termination_reason)

        if reason == TerminationReason.COLLISION.value:
            return self._terminal_result(self.config["collision"])

        if reason == TerminationReason.TIMEOUT.value:
            return self._terminal_result(self.config["timeout"])

        if reason == TerminationReason.STUCK.value:
            return self._terminal_result(self.config["stuck_terminal"])

        if reason == TerminationReason.GOAL_REACHED.value:
            goal_reward = (
                self.config["goal_base"]
                + self.config["goal_time_bonus"] * (1.0 - progress)
            )
            return self._terminal_result(goal_reward)

        if reason == TerminationReason.SIMULATION_STOPPED.value:
            return self._terminal_result(0.0)

        if len(observation) < 10:
            raise ValueError(
                "Reward calculation requires the 10-value model observation."
            )

        front_danger = max(
            clamp(observation[0], 0.0, 1.0),
            clamp(observation[4], 0.0, 1.0),
            clamp(observation[5], 0.0, 1.0)
        )
        forward_speed = clamp(observation[8], -1.0, 1.0)
        positive_speed = max(0.0, forward_speed)

        safe_motion = (
            self.config["safe_motion_scale"]
            * positive_speed
            * (1.0 - front_danger)
        )
        danger = -(
            self.config["danger_penalty_scale"]
            * front_danger ** 2
            * (1.0 + positive_speed)
        )
        time = -(
            self.config["time_penalty_start"]
            + self.config["time_penalty_growth"] * progress
        )
        stuck = (
            -self.config["stuck_penalty"]
            if abs(forward_speed) < self.config["stuck_speed_threshold"]
            else 0.0
        )
        progress_component = float(progress_reward)
        checkpoint_component = float(checkpoint_reward)
        total = (
            safe_motion
            + danger
            + time
            + stuck
            + progress_component
            + checkpoint_component
        )

        return RewardResult(
            total=total,
            safe_motion=safe_motion,
            danger=danger,
            time=time,
            stuck=stuck,
            progress=progress_component,
            checkpoint=checkpoint_component
        )

    @staticmethod
    def _terminal_result(reward):
        return RewardResult(total=reward, terminal=reward)
