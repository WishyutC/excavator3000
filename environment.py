from robot_controller import RobotController
from config import CONFIG


class RCEnvironment:

    def __init__(self):

        self.robot = RobotController()

        self.max_steps = CONFIG["environment"]["max_steps"]
        self.step_count = 0

        self.previous_state = None

    # ----------------------------------------------------
    # Reset Environment
    # ----------------------------------------------------

    def reset(self):

        self.robot.reset()

        self.step_count = 0

        state = self.robot.get_state()

        self.previous_state = state

        return state

    # ----------------------------------------------------
    # Execute Action
    # ----------------------------------------------------

    def step(self, action):

        self.robot.apply_action(action)

        if self.robot.step() == -1:
            return None, 0, True, {}

        state = self.robot.get_state()

        reward = self.compute_reward(state)

        done = self.is_done(state)

        self.previous_state = state

        self.step_count += 1

        info = {}

        return state, reward, done, info

    # ----------------------------------------------------
    # Reward Function
    # ----------------------------------------------------

    def compute_reward(self, state):

        reward = 0.0

        front = state[0]
        back = state[1]
        left = state[2]
        right = state[3]
        lf = state[4]
        rf = state[5]
        lb = state[6]
        rb = state[7]

        # ------------------------
        # Collision
        # ------------------------

        if max(state) > CONFIG["environment"]["collision_threshold"]:
            return -100.0

        # ------------------------
        # Encourage forward motion
        # ------------------------

        reward += 0.5

        # ------------------------
        # Penalty near obstacles
        # ------------------------

        reward -= front / 8000.0
        reward -= lf / 12000.0
        reward -= rf / 12000.0

        # ------------------------
        # Small living reward
        # ------------------------

        reward += 0.05

        return reward

    # ----------------------------------------------------
    # Episode Finished?
    # ----------------------------------------------------

    def is_done(self, state):

        if self.robot.collision():
            return True

        if self.step_count >= self.max_steps:
            return True

        return False

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
        state
    ):

        self.robot.update_hud(
            episode,
            step,
            action,
            reward,
            total_reward,
            state
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

        return 8
