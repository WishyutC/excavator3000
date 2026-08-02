"""Hardware-friendly construction of bounded model observations."""

import math


def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


def proximity_from_distance(distance_m, max_distance_m):
    """Return 0 for clear space and 1 for an obstacle at the sensor."""

    if max_distance_m <= 0.0:
        raise ValueError("sensor_max_distance_m must be greater than zero.")

    if distance_m is None or not math.isfinite(distance_m):
        return 0.0

    return clamp(1.0 - distance_m / max_distance_m, 0.0, 1.0)


def signed_forward_speed(node_velocity, orientation):
    """Project world linear velocity onto the robot's local +X axis."""

    if len(node_velocity) < 3:
        raise ValueError("Node velocity must contain at least 3 values.")

    if len(orientation) < 9:
        raise ValueError("Node orientation must contain 9 values.")

    # Webots returns a row-major local-to-world rotation matrix. Its first
    # column is the robot's local +X (forward) axis in world coordinates.
    forward_x = orientation[0]
    forward_y = orientation[3]
    forward_z = orientation[6]

    return (
        node_velocity[0] * forward_x
        + node_velocity[1] * forward_y
        + node_velocity[2] * forward_z
    )


def build_observation(
    sensor_distances_m,
    node_velocity,
    orientation,
    observation_config
):
    """Build 8 proximity inputs plus normalized speed and turn rate."""

    expected_sensors = observation_config["sensor_count"]

    if len(sensor_distances_m) != expected_sensors:
        raise ValueError(
            f"Expected {expected_sensors} sensor distances, "
            f"received {len(sensor_distances_m)}."
        )

    if len(node_velocity) < 6:
        raise ValueError("Node velocity must contain 6 values.")

    max_forward_speed = observation_config["max_forward_speed_m_s"]
    max_turn_rate = observation_config["max_turn_rate_rad_s"]

    if max_forward_speed <= 0.0:
        raise ValueError("max_forward_speed_m_s must be greater than zero.")

    if max_turn_rate <= 0.0:
        raise ValueError("max_turn_rate_rad_s must be greater than zero.")

    observation = [
        proximity_from_distance(
            distance,
            observation_config["sensor_max_distance_m"]
        )
        for distance in sensor_distances_m
    ]

    forward_speed = signed_forward_speed(node_velocity, orientation)
    observation.append(clamp(
        forward_speed / max_forward_speed,
        -1.0,
        1.0
    ))
    observation.append(clamp(
        node_velocity[5] / max_turn_rate,
        -1.0,
        1.0
    ))

    return observation
