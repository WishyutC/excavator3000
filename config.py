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
        "max_motor_speed": 12.56
    },

    "environment": {
        "max_steps": 500,
        "collision_threshold": 3900,
        "random_start": True,
        "random_start_range_m": 0.4
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
