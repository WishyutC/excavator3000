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
        self.decision_count += 1
        total_reward = 0.0
        accumulated_breakdown = {}
        repeated_steps = 0
        state = self.previous_state
        status = None
        progress_update = self.progress_tracker.snapshot()

        for _ in range(self.action_repeat):
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
                stuck=self.progress_tracker.is_stuck(self.step_count)
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
            "robot_action": robot_action
        })

        return state, total_reward, status.done, info

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
