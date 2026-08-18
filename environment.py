import math
import json
from pathlib import Path

from config import CONFIG
from episode_manager import EpisodeManager
from reward_calculator import RewardCalculator
from progress_tracker import ProgressTracker


class RCEnvironment:

    def __init__(
        self,
        robot=None,
        episode_manager=None,
        reward_calculator=None,
        progress_tracker=None
    ):

        if robot is None:
            from robot_controller import RobotController
            robot = RobotController()
        self.robot = robot
        self.episode_manager = episode_manager or EpisodeManager(self.robot)
        self.reward_calculator = reward_calculator or RewardCalculator()
        self.progress_tracker = progress_tracker or ProgressTracker()

        self.action_ids = tuple(CONFIG["training"]["action_ids"])
        if not self.action_ids or len(set(self.action_ids)) != len(self.action_ids):
            raise ValueError("training.action_ids must contain unique actions.")
        unknown_actions = set(self.action_ids) - set(self.robot.action_table)
        if unknown_actions:
            raise ValueError(f"Unknown configured robot actions: {sorted(unknown_actions)}")
        self.action_repeat = max(1, int(CONFIG["training"]["action_repeat"]))
        self.turn_macro = dict(CONFIG["training"].get("turn_macro", {}))
        self.curriculum_stage = "single"
        self.curriculum_target_checkpoint = None
        self._expert_trace_counts = {}
        self._sensor_expert_turn_action = None
        self._sensor_expert_exit_checkpoint = None

        self.step_count = 0
        self.decision_count = 0

        self.previous_state = None
        self.raw_sensor_values = None

    # ----------------------------------------------------
    # Reset Environment
    # ----------------------------------------------------

    def reset(self):

        self.robot.reset()

        self.step_count = 0
        self.decision_count = 0
        self._sensor_expert_turn_action = None
        self._sensor_expert_exit_checkpoint = None

        self.raw_sensor_values = self.robot.get_raw_sensor_values()
        state = self.robot.get_observation(self.raw_sensor_values)

        self.previous_state = state
        self.progress_tracker.reset(self.robot.get_position(), self.step_count)

        return state

    # ----------------------------------------------------
    # Execute Action
    # ----------------------------------------------------

    def step(self, action):

        if not 0 <= int(action) < len(self.action_ids):
            raise ValueError(f"Invalid policy action {action}.")

        robot_action = self.action_ids[int(action)]
        self.robot.apply_action(robot_action)
        macro_enabled = (
            bool(self.turn_macro.get("enabled", False))
            and robot_action in (1, 2)
        )
        macro_phase = "turn" if macro_enabled else "repeat"
        macro_start_heading = self.robot.get_forward_direction()
        macro_target_angle = float(
            self.turn_macro.get("target_angle_rad", math.pi / 2.0)
        )
        macro_max_turn_steps = max(
            1, int(self.turn_macro.get("maximum_turn_steps", 80))
        )
        macro_exit_steps = max(
            0, int(self.turn_macro.get("exit_forward_steps", 0))
        )
        macro_turn_steps = 0
        macro_exit_executed = 0
        macro_side_integral = 0.0
        maximum_steps = (
            macro_max_turn_steps + macro_exit_steps
            if macro_enabled
            else self.action_repeat
        )
        self.decision_count += 1
        total_reward = 0.0
        accumulated_breakdown = {}
        repeated_steps = 0
        state = self.previous_state
        status = None
        progress_update = self.progress_tracker.snapshot()

        for _ in range(maximum_steps):
            if self.robot.step() == -1:
                stopped = self.episode_manager.simulation_stopped_status()
                return None, total_reward, True, stopped.to_info()

            repeated_steps += 1
            self.step_count += 1
            self.raw_sensor_values = self.robot.get_raw_sensor_values()
            state = self.robot.get_observation(self.raw_sensor_values)
            progress_update = self.progress_tracker.update(
                self.robot.get_position(),
                self.step_count
            )
            status = self.episode_manager.evaluate(
                self.raw_sensor_values,
                self.step_count,
                stuck=self.progress_tracker.is_stuck(self.step_count),
                curriculum_complete=self._curriculum_complete(progress_update)
            )
            reward_result = self.reward_calculator.calculate(
                state,
                status.reason,
                self.step_count,
                progress_reward=progress_update.distance_reward,
                checkpoint_reward=progress_update.checkpoint_reward
            )
            total_reward += reward_result.total
            for name, value in reward_result.to_info().items():
                accumulated_breakdown[name] = (
                    accumulated_breakdown.get(name, 0.0) + value
                )
            if status.done:
                if (
                    self.curriculum_stage != "single"
                    and CONFIG["diagnostics"].get(
                        "sensor_expert_trace", False
                    )
                ):
                    self._trace_terminal_state(status, state, robot_action)
                break

            if macro_enabled:
                if macro_phase == "turn":
                    macro_turn_steps += 1
                    signed_angle = self._signed_heading_change(
                        macro_start_heading,
                        self.robot.get_forward_direction()
                    )
                    desired_angle = (
                        signed_angle if robot_action == 1 else -signed_angle
                    )
                    if desired_angle >= macro_target_angle:
                        if macro_exit_steps == 0:
                            break
                        macro_phase = "exit"
                        self.robot.apply_action(0)
                else:
                    macro_exit_executed += 1
                    oversteer_steps = int(
                        self.turn_macro.get("straighten_after_steps", 0)
                    )
                    exit_angle = (
                        macro_target_angle
                        if macro_exit_executed <= oversteer_steps
                        else float(
                            self.turn_macro.get(
                                "exit_heading_angle_rad", macro_target_angle
                            )
                        )
                    )
                    macro_side_integral = self._apply_macro_centering(
                        state,
                        macro_start_heading,
                        robot_action,
                        macro_side_integral,
                        exit_angle
                    )
                    if macro_exit_executed >= macro_exit_steps:
                        break

        self.previous_state = state
        info = status.to_info()
        info.update({
            "reward_breakdown": accumulated_breakdown,
            "progress": progress_update.to_info(),
            "episode_steps": self.step_count,
            "decision_steps": self.decision_count,
            "action_repeat_executed": repeated_steps,
            "policy_action": int(action),
            "robot_action": robot_action,
            "turn_macro_used": macro_enabled,
            "turn_macro_phase": macro_phase,
            "turn_macro_turn_steps": macro_turn_steps,
            "turn_macro_exit_steps": macro_exit_executed,
            "curriculum_stage": self.curriculum_stage,
            "curriculum_target_checkpoint": self.curriculum_target_checkpoint
        })

        return state, total_reward, status.done, info

    @staticmethod
    def _signed_heading_change(start, current):
        """Return the shortest signed planar angle from start to current."""

        cross = start[0] * current[1] - start[1] * current[0]
        dot = start[0] * current[0] + start[1] * current[1]
        return math.atan2(cross, dot)

    def _apply_macro_centering(
        self,
        observation,
        start_heading,
        robot_action,
        side_integral=0.0,
        desired_angle_magnitude=None
    ):
        """Center in the new lane while holding the intended exit heading."""

        left = max(observation[2], observation[4], observation[6])
        right = max(observation[3], observation[5], observation[7])
        side_error = right - left
        side_gain = float(self.turn_macro.get("centering_gain", 0.35))
        integral_limit = float(
            self.turn_macro.get("centering_integral_limit", 35.0)
        )
        integral_decay = float(
            self.turn_macro.get("centering_integral_decay", 1.0)
        )
        side_integral = max(
            -integral_limit,
            min(
                integral_limit,
                side_integral * integral_decay + side_error
            )
        )
        integral_gain = float(
            self.turn_macro.get("centering_integral_gain", 0.0)
        )
        heading_gain = float(
            self.turn_macro.get("heading_hold_gain", 1.0)
        )
        signed_angle = self._signed_heading_change(
            start_heading, self.robot.get_forward_direction()
        )
        if desired_angle_magnitude is None:
            desired_angle_magnitude = float(
                self.turn_macro.get("target_angle_rad", math.pi / 2.0)
            )
        desired_angle = (
            float(desired_angle_magnitude)
            * (1.0 if robot_action == 1 else -1.0)
        )
        heading_error = math.atan2(
            math.sin(desired_angle - signed_angle),
            math.cos(desired_angle - signed_angle)
        )
        minimum = float(
            self.turn_macro.get("minimum_inner_wheel_ratio", 0.70)
        )
        steering = (
            heading_gain * heading_error
            + side_gain * side_error
            + integral_gain * side_integral
        )
        correction = min(1.0 - minimum, abs(steering))
        if steering > 0.0:
            self.robot.apply_drive_ratios(1.0 - correction, 1.0)
        elif steering < 0.0:
            self.robot.apply_drive_ratios(1.0, 1.0 - correction)
        else:
            self.robot.apply_drive_ratios(1.0, 1.0)
        return side_integral

    def set_curriculum_stage(self, name="single", target_checkpoint=None):
        """Configure an ordered-checkpoint terminal target for one stage."""

        if target_checkpoint is not None:
            target_checkpoint = int(target_checkpoint)
            checkpoint_count = len(self.progress_tracker.waypoints)
            if not 1 <= target_checkpoint <= checkpoint_count:
                raise ValueError(
                    "Curriculum target checkpoint must be between 1 and "
                    f"{checkpoint_count}."
                )
        self.curriculum_stage = str(name)
        self.curriculum_target_checkpoint = target_checkpoint

    def _curriculum_complete(self, progress_update):
        target = self.curriculum_target_checkpoint
        return (
            target is not None
            and progress_update.checkpoints_reached >= target
        )

    def waypoint_expert_action(self):
        """Training-only route expert; waypoint data never enters the model."""

        index = self.progress_tracker.current_index
        waypoints = self.progress_tracker.waypoints
        if index >= len(waypoints):
            return self.action_ids.index(0)

        position = self.robot.get_position()
        forward_x, forward_y = self.robot.get_forward_direction()
        target_x, target_y = waypoints[index]
        direction_x = target_x - position[0]
        direction_y = target_y - position[1]
        heading_error = math.atan2(
            forward_x * direction_y - forward_y * direction_x,
            forward_x * direction_x + forward_y * direction_y
        )
        tolerance = float(
            CONFIG["training"]["curriculum"][
                "expert_heading_tolerance_rad"
            ]
        )
        robot_action = (
            0 if abs(heading_error) <= tolerance
            else 1 if heading_error > 0.0
            else 2
        )
        return self.action_ids.index(robot_action)

    def sensor_expert_action(self, observation):
        """Observable corridor policy built only from deployable sensors."""

        front = max(observation[0], observation[4], observation[5])
        left_obstacle = observation[2]
        right_obstacle = observation[3]
        threshold = float(
            CONFIG["training"]["curriculum"][
                "sensor_expert_front_threshold"
            ]
        )
        macro_enabled = bool(self.turn_macro.get("enabled", False))

        trace_corner = False
        if self._sensor_expert_turn_action is not None:
            trace_corner = True
            index = self.progress_tracker.current_index
            current_x, current_y = self.robot.get_forward_direction()
            position = self.robot.get_position()
            target_x, target_y = self.progress_tracker.waypoints[index]
            direction_x = target_x - position[0]
            direction_y = target_y - position[1]
            direction_length = math.hypot(direction_x, direction_y)
            if direction_length > 0.0:
                direction_x /= direction_length
                direction_y /= direction_length
            alignment = max(
                -1.0,
                min(1.0, current_x * direction_x + current_y * direction_y)
            )
            heading_error = math.acos(alignment)
            tolerance = float(
                CONFIG["training"]["curriculum"][
                    "expert_heading_tolerance_rad"
                ]
            )
            if heading_error <= tolerance:
                self._sensor_expert_turn_action = None
                self._sensor_expert_exit_checkpoint = int(
                    min(
                        self.progress_tracker.current_index + 2,
                        len(self.progress_tracker.waypoints)
                    )
                )
                robot_action = 0
            else:
                robot_action = self._sensor_expert_turn_action
        elif self._sensor_expert_exit_checkpoint is not None:
            trace_corner = True
            if (
                self.progress_tracker.current_index
                >= self._sensor_expert_exit_checkpoint
            ):
                self._sensor_expert_exit_checkpoint = None
                robot_action = 0
            else:
                # The macro has already completed the corner. Keep moving out
                # of it without stacking another full turn maneuver.
                robot_action = (
                    0 if macro_enabled
                    else self.action_ids[self.waypoint_expert_action()]
                )
        elif self.progress_tracker.current_index not in set(
            CONFIG["training"]["curriculum"][
                "sensor_expert_turn_checkpoints"
            ]
        ):
            robot_action = 0
        elif front < threshold:
            robot_action = 0
        else:
            trace_corner = True
            # Wait for an observable end wall, then use the known training
            # route to supply successful demonstrations. The greedy gate does
            # not use this expert and must reproduce the turns from sensors.
            robot_action = (
                1 if self.progress_tracker.current_index == 1 else 2
            )
            if macro_enabled:
                self._sensor_expert_exit_checkpoint = int(
                    min(
                        self.progress_tracker.current_index + 2,
                        len(self.progress_tracker.waypoints)
                    )
                )
            else:
                self._sensor_expert_turn_action = robot_action

        if (
            trace_corner
            and CONFIG["diagnostics"].get("sensor_expert_trace", False)
        ):
            self._trace_sensor_expert(
                observation,
                left_obstacle,
                right_obstacle,
                robot_action
            )
        return self.action_ids.index(robot_action)

    def _trace_terminal_state(self, status, observation, robot_action):
        trace_path = Path("runs") / "episode_terminal_trace.jsonl"
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "stage": self.curriculum_stage,
            "reason": status.reason,
            "checkpoint_index": int(self.progress_tracker.current_index),
            "step": int(self.step_count),
            "position": list(map(float, self.robot.get_position())),
            "forward_direction": list(map(float, self.robot.get_forward_direction())),
            "raw_sensors": list(map(float, self.raw_sensor_values)),
            "observation": list(map(float, observation)),
            "robot_action": int(robot_action)
        }
        with trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    def _trace_sensor_expert(
        self,
        observation,
        left_obstacle,
        right_obstacle,
        robot_action
    ):
        checkpoint = int(self.progress_tracker.current_index)
        count = self._expert_trace_counts.get(checkpoint, 0)
        if count >= 50:
            return
        self._expert_trace_counts[checkpoint] = count + 1
        trace_path = Path("runs") / "sensor_expert_trace.jsonl"
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "stage": self.curriculum_stage,
            "checkpoint_index": checkpoint,
            "position": list(map(float, self.robot.get_position())),
            "forward_direction": list(map(float, self.robot.get_forward_direction())),
            "observation": list(map(float, observation)),
            "left_obstacle": float(left_obstacle),
            "right_obstacle": float(right_obstacle),
            "robot_action": int(robot_action)
        }
        with trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    # ----------------------------------------------------
    # Close Environment
    # ----------------------------------------------------

    def close(self):

        self.robot.close()

    def update_hud(
        self,
        episode,
        step,
        action,
        reward,
        total_reward,
        state,
        info
    ):

        robot_action = self.action_ids[int(action)]
        self.robot.update_hud(
            episode,
            step,
            robot_action,
            reward,
            total_reward,
            self.raw_sensor_values,
            state,
            info
        )

    # ----------------------------------------------------
    # Action Space
    # ----------------------------------------------------

    @property
    def action_size(self):

        return len(self.action_ids)

    # ----------------------------------------------------
    # Observation Space
    # ----------------------------------------------------

    @property
    def state_size(self):

        return CONFIG["observation"]["sensor_count"] + 2
