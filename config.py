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

    "environment": {
        "max_steps": 500,
        "collision_threshold": 3900,
        "random_start": True,
        "random_heading": True,
        # car_env_test_v2.wbt uses RectangleArena floorSize 2 2.
        "arena_size_m": {
            "x": 2.0,
            "y": 2.0
        },
        "respawn_wall_clearance_m": 0.05
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
        # Reserved for the future PPO implementation.
        "enabled": False,
        "algorithm": "ppo",
        "device": "auto",
        "seed": 42,
        "save_directory": "models"
    }
}
