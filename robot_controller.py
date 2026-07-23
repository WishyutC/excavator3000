from controller import Supervisor
from config import CONFIG
import json
import random
import math
from pathlib import Path
import socket
import subprocess
import sys

TIME_STEP = CONFIG["simulation"]["time_step_ms"]
COLLISION_THRESHOLD = CONFIG["environment"]["collision_threshold"]
OBSERVER_CONFIG = CONFIG["observer"]


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

        # Webots' default ENU coordinates use Z as the vertical axis.
        self.start_position = self.translation.getSFVec3f()
        self.robot_height = self.start_position[2]

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

        device_max_speed = min(
            self.left_motor.getMaxVelocity(),
            self.right_motor.getMaxVelocity()
        )
        configured_max_speed = CONFIG["simulation"]["max_motor_speed"]
        speed_scale = CONFIG["robot"]["drive"]["speed_scale"]

        if not 0.0 < speed_scale <= 1.0:
            raise ValueError("robot.drive.speed_scale must be between 0 and 1.")

        self.motor_max_speed = min(device_max_speed, configured_max_speed)
        self.drive_speed = self.motor_max_speed * speed_scale

        action_ratios = CONFIG["robot"]["drive"]["action_ratios"]

        def wheel_speeds(action_name):
            left_ratio, right_ratio = action_ratios[action_name]
            return (
                left_ratio * self.drive_speed,
                right_ratio * self.drive_speed
            )

        # Discrete actions expressed as ratios of the safe drive speed.
        self.action_table = {
            0: wheel_speeds("forward"),
            1: wheel_speeds("turn_left"),
            2: wheel_speeds("turn_right"),
            3: wheel_speeds("reverse"),
            4: wheel_speeds("stop"),
            5: wheel_speeds("reverse_left"),
            6: wheel_speeds("reverse_right")
        }

        self.action_names = {
            0: "FORWARD",
            1: "TURN LEFT",
            2: "TURN RIGHT",
            3: "REVERSE",
            4: "STOP",
            5: "REVERSE LEFT",
            6: "REVERSE RIGHT"
        }

        self.hud_socket = None
        self.hud_process = None
        self.hud_update_count = 0

        if OBSERVER_CONFIG["enabled"]:
            self.hud_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

            if OBSERVER_CONFIG["auto_launch"]:
                self._start_hud()

        estimated_speed = (
            self.drive_speed * CONFIG["robot"]["wheel_radius_m"]
        )
        print(
            "Robot Controller Ready | "
            f"drive {self.drive_speed:.2f} rad/s | "
            f"estimated {estimated_speed:.3f} m/s"
        )

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

        left = max(-self.motor_max_speed, min(self.motor_max_speed, left))
        right = max(-self.motor_max_speed, min(self.motor_max_speed, right))

        self.left_motor.setVelocity(left)
        self.right_motor.setVelocity(right)

    def get_action_name(self, action):

        return self.action_names.get(action, "UNKNOWN")

    @staticmethod
    def _lookup_rows(sensor):

        try:
            table = sensor.getLookupTable()
        except (AttributeError, TypeError):
            return []

        if not table:
            return []

        if isinstance(table[0], (list, tuple)):
            return [tuple(row[:3]) for row in table if len(row) >= 3]

        return [
            tuple(table[index:index + 3])
            for index in range(0, len(table) - 2, 3)
        ]

    def sensor_distance(self, sensor_index, value):

        # Webots lookup rows are: distance (m), output value, noise.
        rows = self._lookup_rows(self.sensors[sensor_index])

        if not rows:
            return None

        points = sorted(
            ((float(row[1]), float(row[0])) for row in rows),
            key=lambda point: point[0]
        )

        if value <= points[0][0]:
            return points[0][1]

        if value >= points[-1][0]:
            return points[-1][1]

        for (value_a, distance_a), (value_b, distance_b) in zip(
            points,
            points[1:]
        ):
            if value_a <= value <= value_b:
                if value_b == value_a:
                    return min(distance_a, distance_b)

                ratio = (value - value_a) / (value_b - value_a)
                return distance_a + ratio * (distance_b - distance_a)

        return None

    def _start_hud(self):

        hud_path = Path(__file__).with_name("hud_gui.py")

        try:
            self.hud_process = subprocess.Popen([
                sys.executable,
                str(hud_path),
                "--host",
                OBSERVER_CONFIG["host"],
                "--port",
                str(OBSERVER_CONFIG["port"])
            ])
        except OSError as error:
            print(f"Could not start HUD window: {error}")

    def update_hud(
        self,
        episode,
        step,
        action,
        reward,
        total_reward,
        state
    ):

        if not OBSERVER_CONFIG["enabled"] or self.hud_socket is None:
            return

        self.hud_update_count += 1
        update_interval = max(1, OBSERVER_CONFIG["update_every_steps"])

        if self.hud_update_count % update_interval != 0:
            return

        left_speed, right_speed = self.action_table[action]
        node_velocity = self.node.getVelocity()
        linear_speed = math.hypot(node_velocity[0], node_velocity[1])
        angular_speed = node_velocity[5]
        wheel_radius = CONFIG["robot"]["wheel_radius_m"]

        packet = {
            "type": "telemetry",
            "episode": episode,
            "step": step,
            "action": action,
            "action_name": self.get_action_name(action),
            "left_speed": left_speed,
            "right_speed": right_speed,
            "linear_speed_m_s": linear_speed,
            "angular_speed_rad_s": angular_speed,
            "max_linear_speed_m_s": self.motor_max_speed * wheel_radius,
            "target_linear_speed_m_s": (
                (left_speed + right_speed) * 0.5 * wheel_radius
            ),
            "reward": reward,
            "total_reward": total_reward,
            "sensors": [
                {
                    "raw": value,
                    "distance_m": self.sensor_distance(index, value)
                }
                for index, value in enumerate(state)
            ]
        }

        try:
            message = json.dumps(packet).encode("utf-8")
            self.hud_socket.sendto(
                message,
                (OBSERVER_CONFIG["host"], OBSERVER_CONFIG["port"])
            )
        except OSError as error:
            print(f"HUD telemetry error: {error}")

    def stop(self):

        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

    def collision(self):

        return max(self.get_state()) > COLLISION_THRESHOLD

    def reset(self):

        self.stop()

        if CONFIG["environment"]["random_heading"]:
            angle = random.uniform(-math.pi, math.pi)
        else:
            angle = 0.0

        if CONFIG["environment"]["random_start"]:
            robot = CONFIG["robot"]
            environment = CONFIG["environment"]
            arena = environment["arena_size_m"]
            clearance = environment["respawn_wall_clearance_m"]

            half_length = robot["length_m"] / 2.0
            half_width = robot["width_m"] / 2.0

            # Axis-aligned extents of the rotated rectangular footprint.
            extent_x = (
                abs(math.cos(angle)) * half_length
                + abs(math.sin(angle)) * half_width
            )
            extent_y = (
                abs(math.sin(angle)) * half_length
                + abs(math.cos(angle)) * half_width
            )

            x_limit = arena["x"] / 2.0 - extent_x - clearance
            y_limit = arena["y"] / 2.0 - extent_y - clearance

            if x_limit <= 0.0 or y_limit <= 0.0:
                raise ValueError(
                    "Arena is too small for the configured robot dimensions "
                    "and respawn clearance."
                )

            x = random.uniform(-x_limit, x_limit)
            y = random.uniform(-y_limit, y_limit)
        else:
            x = self.start_position[0]
            y = self.start_position[1]

        self.translation.setSFVec3f([
            x,
            y,
            self.robot_height
        ])

        # Heading rotates around the vertical Z axis in the ENU world.
        self.rotation.setSFRotation([
            0,
            0,
            1,
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

        if self.hud_socket is not None:
            if OBSERVER_CONFIG["close_with_controller"]:
                try:
                    message = json.dumps({"type": "shutdown"}).encode("utf-8")
                    self.hud_socket.sendto(
                        message,
                        (OBSERVER_CONFIG["host"], OBSERVER_CONFIG["port"])
                    )
                except OSError:
                    pass

            self.hud_socket.close()

        if (
            self.hud_process is not None
            and OBSERVER_CONFIG["close_with_controller"]
        ):
            try:
                self.hud_process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                self.hud_process.terminate()
