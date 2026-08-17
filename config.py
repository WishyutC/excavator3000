"""Central configuration for the Webots controller, observer, and training."""

import os


CONFIG = {
    "program": {
        # Modes: "test", "diagnostic", "train", or "evaluate".
        "mode": "train",
        "terminal_output": False,
        "test_action": 2
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
        # Physics steps. With action_repeat=4, the agent makes at most 750
        # decisions during a 3,000-step episode.
        "max_steps": 3_000,
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
            "timeout": -75.0,
            "goal_base": 100.0,
            "goal_time_bonus": 50.0,
            "safe_motion_scale": 0.01,
            "danger_penalty_scale": 0.2,
            "time_penalty_start": 0.005,
            "time_penalty_growth": 0.02,
            "stuck_speed_threshold": 0.05,
            "stuck_penalty": 0.020,
            "stuck_terminal": -60.0
        },
        "progress": {
            # Privileged training signal only. Waypoint coordinates never enter
            # the observation vector exported to the ESP32 policy.
            "enabled": True,
            "checkpoint_radius_m": 0.55,
            "checkpoint_reward": 5.0,
            "distance_reward_scale": 2.0,
            "distance_delta_clip_m": 0.03,
            "waypoints": [
                [-2.85, -0.75],
                [-2.85, 1.70],
                [-2.85, 3.10],
                [0.50, 3.10],
                [3.05, 3.10],
                [3.05, 0.00],
                [3.05, -4.05],
                [1.25, -4.05],
                [-0.10, -4.05],
                [-0.10, -2.00]
            ]
        },
        "stuck_detection": {
            "enabled": True,
            "no_progress_steps": 400,
            "minimum_progress_m": 0.03
        }
    },

    "observer": {
        # Disable this during high-speed training to remove GUI overhead.
        "enabled": True,
        "enabled_in_training": False,
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
        "enabled": True,
        "format": "csv",
        "directory": "runs/curriculum_v2"
    },

    "training": {
        "algorithm": "dqn",
        "device": "auto",
        "seed": 42,
        "episodes": 10000,
        "gamma": 0.99,
        "learning_rate": 0.0003,
        "hidden_sizes": [64, 64],
        # Basic curriculum: forward, left, and right only.
        "action_ids": [0, 1, 2],
        "action_repeat": 4,
        "double_dqn": True,
        "reward_scale": 0.01,
        "epsilon": {
            "start": 1.0,
            "end": 0.05,
            "decay_steps": 1_000_000
        },
        "train_every_steps": 1,
        "gradient_clip_norm": 10.0,
        "target_update_steps": 1000,
        "save_directory": "models/curriculum_v2",
        "checkpoint_name": "dqn_latest.pt",
        "best_checkpoint_name": "dqn_best.pt",
        "candidate_checkpoint_name": "dqn_candidate.pt",
        "save_every_episodes": 50,
        "resume": False,
        "replay_buffer": {
            "type": "uniform",
            "capacity": 50000,
            "batch_size": 64,
            "learning_starts": 2000
        }
    },

    "evaluation": {
        "episodes": 20,
        "checkpoint": "models/curriculum_v2/dqn_best.pt"
    },

    "diagnostics": {
        "turn_steps": 80,
        "minimum_turn_rate_rad_s": 0.1
    }
}


# Environment overrides support headless diagnostics and controlled smoke runs
# without rewriting the saved configuration.
if os.environ.get("RL_PROGRAM_MODE"):
    CONFIG["program"]["mode"] = os.environ["RL_PROGRAM_MODE"].lower()
if os.environ.get("RL_TRAINING_EPISODES"):
    CONFIG["training"]["episodes"] = int(os.environ["RL_TRAINING_EPISODES"])
