"""Central configuration for the Webots controller, observer, and training."""


CONFIG = {
    "program": {
        # Current modes: "test". Future modes: "train" and "evaluate".
        "mode": "test",
        "terminal_output": True,
        "test_action": 0
    },

    "simulation": {
        "time_step_ms": 32,
        # Requested ceiling; the real Webots motor limit still takes priority.
        "max_motor_speed": 12.9
    },

    "robot": {
        "width_m": 0.145,
        "length_m": 0.255,
        "wheel_radius_m": 0.0205,
        "drive": {
            # 0.95 means use 95% of the motor's actual maximum velocity.
            "speed_scale": 0.95,
            "action_ratios": {
                "forward": [1.0, 1.0],
                "turn_left": [0.45, 1.0],
                "turn_right": [1.0, 0.45],
                "reverse": [-0.80, -0.80],
                "stop": [0.0, 0.0],
                "reverse_left": [-0.40, -0.80],
                "reverse_right": [-0.80, -0.40]
            }
        }
    },

    "observation": {
        # DQN input: 8 obstacle proximities + forward speed + turn rate.
        "sensor_count": 8,
        # Matches the lookup-table range in car_env_test_v2.wbt.
        "sensor_max_distance_m": 0.5,
        # Physical normalization limits; values outside them are clipped.
        "max_forward_speed_m_s": 0.129,
        "max_turn_rate_rad_s": 2.0
    },

    "environment": {
        "max_steps": 500,
        "collision_threshold": 3900,
        # Keep the mapped start position until track-safe spawn zones are added.
        "random_start": False,
        "random_heading": False,
        # car_env_test_v2.wbt uses RectangleArena floorSize 10 10.
        "arena_size_m": {
            "x": 10.0,
            "y": 10.0
        },
        "respawn_wall_clearance_m": 0.05,
        "goal": {
            "enabled": True,
            "def": "GOAL",
            # Matches the thin green finish line in car_env_test_v2.wbt.
            "size_m": {
                "x": 1.7,
                "y": 0.1
            },
            "tolerance_m": 0.03
        },
        "reward": {
            "collision": -100.0,
            "timeout": -50.0,
            "goal_base": 100.0,
            "goal_time_bonus": 50.0,
            "safe_motion_scale": 0.03,
            "danger_penalty_scale": 0.20,
            "time_penalty_start": 0.005,
            "time_penalty_growth": 0.020,
            "stuck_speed_threshold": 0.05,
            "stuck_penalty": 0.020
        }
    },

    "observer": {
        # Disable this during high-speed training to remove GUI overhead.
        "enabled": True,
        "auto_launch": True,
        "host": "127.0.0.1",
        "port": 8765,
        "update_every_steps": 1,
        "close_with_controller": True,
        "window_title": "Webots RL Agent Observer",
        "proximity_m": {
            "danger": 0.05,
            "warning": 0.12,
            "caution": 0.25
        }
    },

    "logging": {
        # File logging will be added with the full observer dashboard.
        "enabled": False,
        "format": "csv",
        "directory": "runs"
    },

    "training": {
        # DQN training remains disabled while its network/trainer are developed.
        "enabled": False,
        "algorithm": "dqn",
        "device": "auto",
        "seed": 42,
        "save_directory": "models",
        "replay_buffer": {
            "type": "uniform",
            "capacity": 50_000,
            "batch_size": 64,
            "learning_starts": 2_000
        }
    }
}
