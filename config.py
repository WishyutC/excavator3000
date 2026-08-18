"""Central configuration for the Webots controller, observer, and training."""

import os


CONFIG = {
    "program": {
        # Modes: "test", "diagnostic", "train", or "evaluate".
        "mode": "evaluate",
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
                # Tight differential turns for the track's 90-degree bends.
                "turn_left": [0.0, 0.80],
                "turn_right": [0.80, 0.0],
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
        # Matches the extended lookup-table range in car_env_test_v2.wbt.
        # The 1.4 m corridor needs >0.7 m coverage to disambiguate corners.
        "sensor_max_distance_m": 0.8,
        # Physical normalization limits; values outside them are clipped.
        "max_forward_speed_m_s": 0.129,
        "max_turn_rate_rad_s": 2.0
    },

    "environment": {
        # Physics steps. The full 10-checkpoint route is longer than 3,000
        # steps; stuck detection still ends policies that stop progressing.
        "max_steps": 10_000,
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
            "timeout": -100.0,
            "goal_base": 100.0,
            "goal_time_bonus": 50.0,
            "curriculum_goal_base": 80.0,
            "curriculum_goal_time_bonus": 40.0,
            "safe_motion_scale": 0.01,
            "danger_penalty_scale": 0.2,
            "clear_space_steering_penalty_scale": 0.10,
            "time_penalty_start": 0.005,
            "time_penalty_growth": 0.02,
            "stuck_speed_threshold": 0.05,
            "stuck_penalty": 0.020,
            # This remains worse than collision after roughly 100 decisions
            # of gamma discount, preventing a deliberate delayed-stop policy.
            "stuck_terminal": -200.0
        },
        "progress": {
            # Privileged training signal only. Waypoint coordinates never enter
            # the observation vector exported to the ESP32 policy.
            "enabled": True,
            # A learnable capture region for uniform replay, while remaining
            # 20 cm tighter than the original premature 0.55 m transition.
            "checkpoint_radius_m": 0.35,
            "checkpoint_reward": 25.0,
            "distance_reward_scale": 10.0,
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
        "directory": "runs/curriculum_v4_macro_final"
    },

    "training": {
        "algorithm": "dqn",
        "device": "auto",
        "seed": 42,
        "episodes": 20_000,
        # At action_repeat=4 the full route spans far more than 100 decisions;
        # gamma=0.999 keeps later checkpoints relevant to earlier actions.
        "gamma": 0.999,
        "learning_rate": 0.0002,
        "hidden_sizes": [64, 64],
        # Basic curriculum: forward, left, and right only.
        "action_ids": [0, 1, 2],
        "action_repeat": 4,
        # A turn is one semantic DQN action. The low-level controller commits
        # to the bend and clears the corner before requesting another action,
        # preventing left/right oscillation when several sensors fire at once.
        # On the ESP32 this angle-based maneuver can be implemented with an
        # IMU, wheel encoders, or a hardware-tuned equivalent duration.
        "turn_macro": {
            "enabled": True,
            "target_angle_rad": 1.75,
            "exit_heading_angle_rad": 1.57,
            "maximum_turn_steps": 80,
            "exit_forward_steps": 560,
            "straighten_after_steps": 500,
            "centering_gain": 0.10,
            "centering_integral_gain": 0.0,
            "centering_integral_limit": 25.0,
            "centering_integral_decay": 0.96,
            "heading_hold_gain": 1.0,
            "minimum_inner_wheel_ratio": 0.60
        },
        "double_dqn": True,
        "reward_scale": 0.01,
        # Enabled per guided curriculum stage; zero in ordinary DQN training.
        "expert_imitation_weight": 0.0,
        "epsilon": {
            "start": 1.0,
            "end": 0.05,
            "decay_steps": 1_000_000
        },
        # Store every transition but update the network every four actions,
        # the standard DQN cadence for lower compute and less correlated SGD.
        "train_every_steps": 4,
        "gradient_clip_norm": 10.0,
        "target_update_steps": 1000,
        "save_directory": "models/curriculum_v4_macro_final",
        "checkpoint_name": "dqn_latest.pt",
        "best_checkpoint_name": "dqn_best.pt",
        "candidate_checkpoint_name": "dqn_candidate.pt",
        "save_every_episodes": 50,
        "resume": False,
        "curriculum": {
            "enabled": True,
            "start_stage_index": 6,
            "initial_policy_checkpoint": (
                "models/curriculum_v4_macro/stage_05_cp6/dqn_latest.pt"
            ),
            "check_interval_episodes": 100,
            "success_window_episodes": 100,
            "training_success_rate": 0.75,
            "evaluation_episodes": 50,
            "evaluation_success_rate": 0.70,
            "expert_heading_tolerance_rad": 0.12,
            "sensor_expert_front_threshold": 0.35,
            "sensor_expert_turn_checkpoints": [1, 3, 5, 7, 9],
            # Each new stage transfers policy weights only. Optimizer state,
            # replay memory, and epsilon schedule restart for the new task.
            "stages": [
                {
                    "name": "stage_01_cp1",
                    "target_checkpoint": 1,
                    # Bootstrap the same 3-output network with successful
                    # straight-driving demonstrations before turns unlock.
                    "forced_action": 0,
                    "minimum_episodes": 150,
                    "maximum_episodes": 400,
                    "epsilon_start": 1.0,
                    "epsilon_decay_steps": 60_000
                },
                {
                    "name": "stage_02_cp2",
                    "target_checkpoint": 2,
                    "expert_policy": "sensor",
                    "minimum_episodes": 300,
                    "maximum_episodes": 300,
                    "epsilon_start": 0.20,
                    "epsilon_decay_steps": 100_000
                },
                {
                    "name": "stage_03_cp3",
                    "target_checkpoint": 3,
                    "expert_policy": "sensor",
                    "minimum_episodes": 500,
                    "maximum_episodes": 1_800,
                    "epsilon_start": 0.40,
                    "epsilon_decay_steps": 200_000
                },
                {
                    "name": "stage_04_cp4",
                    "target_checkpoint": 4,
                    "expert_policy": "sensor",
                    "minimum_episodes": 300,
                    "maximum_episodes": 300,
                    "epsilon_start": 0.45,
                    "epsilon_decay_steps": 300_000
                },
                {
                    "name": "stage_05_cp6",
                    "target_checkpoint": 6,
                    "expert_policy": "sensor",
                    "minimum_episodes": 100,
                    "maximum_episodes": 600,
                    "greedy_check_interval_episodes": 100,
                    "epsilon_start": 0.45,
                    "epsilon_decay_steps": 400_000
                },
                {
                    "name": "stage_06_cp8",
                    "target_checkpoint": 8,
                    "expert_policy": "sensor",
                    "minimum_episodes": 100,
                    "maximum_episodes": 300,
                    "greedy_check_interval_episodes": 100,
                    "epsilon_start": 0.45,
                    "epsilon_decay_steps": 500_000
                },
                {
                    "name": "stage_07_cp10",
                    "target_checkpoint": 10,
                    "expert_policy": "sensor",
                    "minimum_episodes": 100,
                    "maximum_episodes": 300,
                    "greedy_check_interval_episodes": 100,
                    "epsilon_start": 0.45,
                    "epsilon_decay_steps": 600_000
                },
                {
                    "name": "stage_08_goal",
                    "target_checkpoint": None,
                    "expert_policy": "sensor",
                    "minimum_episodes": 100,
                    "maximum_episodes": 300,
                    "greedy_check_interval_episodes": 100,
                    "epsilon_start": 0.35,
                    "epsilon_decay_steps": 600_000
                }
            ]
        },
        "replay_buffer": {
            "type": "uniform",
            "capacity": 50000,
            "batch_size": 64,
            "learning_starts": 2000
        }
    },

    "evaluation": {
        "episodes": 50,
        "checkpoint": (
            "models/curriculum_v4_macro_final/"
            "stage_08_goal/dqn_latest.pt"
        ),
        "curriculum_target_checkpoint": None
    },

    "diagnostics": {
        "turn_steps": 80,
        "minimum_turn_rate_rad_s": 0.1,
        "sensor_expert_trace": False
    }
}


# Environment overrides support headless diagnostics and controlled smoke runs
# without rewriting the saved configuration.
if os.environ.get("RL_PROGRAM_MODE"):
    CONFIG["program"]["mode"] = os.environ["RL_PROGRAM_MODE"].lower()
if os.environ.get("RL_TRAINING_EPISODES"):
    CONFIG["training"]["episodes"] = int(os.environ["RL_TRAINING_EPISODES"])
