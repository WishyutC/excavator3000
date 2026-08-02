from robot_controller import RobotController
from config import CONFIG
from episode_manager import EpisodeManager
from reward_calculator import RewardCalculator


class RCEnvironment:

    def __init__(self):

        self.robot = RobotController()
        self.episode_manager = EpisodeManager(self.robot)
        self.reward_calculator = RewardCalculator()

        self.step_count = 0

        self.previous_state = None
        self.raw_sensor_values = None

    # ----------------------------------------------------
    # Reset Environment
    # ----------------------------------------------------

    def reset(self):

        self.robot.reset()

        self.step_count = 0

        self.raw_sensor_values = self.robot.get_raw_sensor_values()
        state = self.robot.get_observation(self.raw_sensor_values)

        self.previous_state = state

        return state

    # ----------------------------------------------------
    # Execute Action
    # ----------------------------------------------------

    def step(self, action):

        self.robot.apply_action(action)

        if self.robot.step() == -1:
            status = self.episode_manager.simulation_stopped_status()
            return None, 0.0, True, status.to_info()

        self.raw_sensor_values = self.robot.get_raw_sensor_values()
        state = self.robot.get_observation(self.raw_sensor_values)
        self.step_count += 1

        status = self.episode_manager.evaluate(
            self.raw_sensor_values,
            self.step_count
        )
        reward_result = self.reward_calculator.calculate(
            state,
            status.reason,
            self.step_count
        )
        reward = reward_result.total

        self.previous_state = state

        info = status.to_info()
        info["reward_breakdown"] = reward_result.to_info()

        return state, reward, status.done, info

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

        self.robot.update_hud(
            episode,
            step,
            action,
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

        return 7

    # ----------------------------------------------------
    # Observation Space
    # ----------------------------------------------------

    @property
    def state_size(self):

        return CONFIG["observation"]["sensor_count"] + 2
