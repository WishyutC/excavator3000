from controller import Supervisor
import random
import math

TIME_STEP = 32
MAX_SPEED = 12.56
COLLISION_THRESHOLD = 3900


class RobotController:

    def __init__(self):

        self.robot = Supervisor()
        self.node = self.robot.getSelf()

        if self.node is None:
            raise RuntimeError("Could not get the e-puck Supervisor node.")

        self.translation = self.node.getField("translation")
        self.rotation = self.node.getField("rotation")

        if self.translation is None or self.rotation is None:
            raise RuntimeError(
                "Could not access robot translation or rotation fields."
            )

        # Save original height
        self.start_position = self.translation.getSFVec3f()
        self.robot_height = self.start_position[1]

        # Motors
        self.left_motor = self.robot.getDevice("left wheel motor")
        self.right_motor = self.robot.getDevice("right wheel motor")

        if self.left_motor is None or self.right_motor is None:
            raise RuntimeError("Could not find e-puck wheel motors.")

        self.left_motor.setPosition(float("inf"))
        self.right_motor.setPosition(float("inf"))

        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

        # Sensors
        sensor_names = [
            "distance sensor front",
            "distance sensor back",
            "distance sensor left",
            "distance sensor right",
            "distance sensor left front",
            "distance sensor right front",
            "distance sensor left back",
            "distance sensor right back"
        ]

        self.sensors = []

        for name in sensor_names:

            sensor = self.robot.getDevice(name)

            if sensor is None:
                raise RuntimeError(
                    f"Could not find distance sensor: {name}"
                )

            sensor.enable(TIME_STEP)

            self.sensors.append(sensor)

        # Discrete actions
        self.action_table = {
            0: (4.5, 4.5),       # Forward
            1: (2.0, 5.0),       # Turn left
            2: (5.0, 2.0),       # Turn right
            3: (-4.0, -4.0),     # Reverse
            4: (0.0, 0.0),       # Stop
            5: (-2.0, -4.0),     # Reverse left
            6: (-4.0, -2.0)      # Reverse right
        }

        print("Robot Controller Ready")

    def step(self):
        return self.robot.step(TIME_STEP)

    def get_state(self):

        return [
            sensor.getValue()
            for sensor in self.sensors
        ]

    def apply_action(self, action):

        if action not in self.action_table:
            raise ValueError(
                f"Invalid action {action}. "
                f"Valid actions: 0-{len(self.action_table) - 1}"
            )

        left, right = self.action_table[action]

        left = max(-MAX_SPEED, min(MAX_SPEED, left))
        right = max(-MAX_SPEED, min(MAX_SPEED, right))

        self.left_motor.setVelocity(left)
        self.right_motor.setVelocity(right)

    def stop(self):

        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

    def collision(self):

        return max(self.get_state()) > COLLISION_THRESHOLD

    def reset(self):

        self.stop()

        # Random starting position
        x = random.uniform(-0.4, 0.4)
        z = random.uniform(-0.4, 0.4)

        self.translation.setSFVec3f([
            x,
            self.robot_height,
            z
        ])

        # Random heading
        angle = random.uniform(
            -math.pi,
            math.pi
        )

        self.rotation.setSFRotation([
            0,
            1,
            0,
            angle
        ])

        # Remove previous velocity / momentum
        self.node.resetPhysics()

        # Allow sensors and physics to update
        for _ in range(5):

            if self.robot.step(TIME_STEP) == -1:
                break

    def close(self):

        self.stop()