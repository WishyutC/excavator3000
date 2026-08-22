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
        # All layouts use the same RectangleArena in car_env_test_v2.wbt.
        # "survival_mix" changes among three obstacle layouts every episode.
        "map_selector": "survival_mix",
        "maps": {
            "survival_mix": {
                "label": "Three-map survival mix",
                "episode_type": "survival",
                "goal_enabled": False,
                "progress_enabled": False,
                "random_start": True,
                "random_heading": True,
                "turn_macro_enabled": False,
                # Training draws randomly; evaluation cycles in this order so
                # every map receives the same number of benchmark episodes.
                "map_pool": [
                    "obstacle_field",
                    "tight_corridors",
                    "dense_pinch_points"
                ],
                "training_selection": "random",
                "evaluation_selection": "round_robin",
                "spawn_clearance_m": 0.12,
                "spawn_attempts": 2000,
                "obstacles": []
            },
            "race_track": {
                "label": "Original goal track",
                "episode_type": "goal",
                "goal_enabled": True,
                "progress_enabled": True,
                "random_start": False,
                "random_heading": False,
                "turn_macro_enabled": True,
                "spawn_clearance_m": 0.05,
                "spawn_attempts": 500,
                "obstacles": []
            },
            "obstacle_field": {
                "label": "Random-shape survival field",
                "episode_type": "survival",
                "goal_enabled": False,
                "progress_enabled": False,
                "random_start": True,
                "random_heading": True,
                "turn_macro_enabled": False,
                "spawn_clearance_m": 0.25,
                "spawn_attempts": 1000,
                # Mixed lengths, widths, and rotations create an open-ended
                # avoidance field rather than another memorized race lane.
                "obstacles": [
                    {"center": [-3.25, 2.85], "size": [1.25, 0.55], "angle_rad": 0.35},
                    {"center": [-1.05, 3.25], "size": [0.55, 1.55], "angle_rad": -0.45},
                    {"center": [1.55, 3.10], "size": [1.65, 0.45], "angle_rad": 0.70},
                    {"center": [3.45, 1.70], "size": [0.65, 1.40], "angle_rad": 0.15},
                    {"center": [-3.45, 0.55], "size": [0.45, 1.70], "angle_rad": -0.25},
                    {"center": [-0.65, 1.05], "size": [1.80, 0.40], "angle_rad": 0.25},
                    {"center": [1.75, 0.65], "size": [0.55, 1.65], "angle_rad": -0.65},
                    {"center": [-2.10, -1.35], "size": [1.55, 0.50], "angle_rad": -0.75},
                    {"center": [0.25, -1.70], "size": [0.55, 1.80], "angle_rad": 0.40},
                    {"center": [3.10, -1.45], "size": [1.35, 0.50], "angle_rad": -0.30},
                    {"center": [-3.20, -3.25], "size": [0.75, 1.30], "angle_rad": 0.55},
                    {"center": [-0.75, -3.55], "size": [1.50, 0.45], "angle_rad": 0.10},
                    {"center": [2.35, -3.30], "size": [0.50, 1.45], "angle_rad": -0.50},
                    # Four diagonal corner gates touch both arena walls, and
                    # eight staggered wall teeth break the easy perimeter loop.
                    {"center": [-4.35, 4.35], "size": [1.40, 0.50], "angle_rad": 0.7854, "color": [0.65, 0.18, 0.08]},
                    {"center": [4.35, 4.35], "size": [1.40, 0.50], "angle_rad": -0.7854, "color": [0.65, 0.18, 0.08]},
                    {"center": [-4.35, -4.35], "size": [1.40, 0.50], "angle_rad": -0.7854, "color": [0.65, 0.18, 0.08]},
                    {"center": [4.35, -4.35], "size": [1.40, 0.50], "angle_rad": 0.7854, "color": [0.65, 0.18, 0.08]},
                    {"center": [-2.35, 4.45], "size": [0.55, 1.50], "angle_rad": 0.0, "color": [0.75, 0.30, 0.08]},
                    {"center": [2.25, 4.45], "size": [0.55, 1.50], "angle_rad": 0.0, "color": [0.75, 0.30, 0.08]},
                    {"center": [-1.90, -4.45], "size": [0.55, 1.50], "angle_rad": 0.0, "color": [0.75, 0.30, 0.08]},
                    {"center": [2.80, -4.45], "size": [0.55, 1.50], "angle_rad": 0.0, "color": [0.75, 0.30, 0.08]},
                    {"center": [-4.45, -1.85], "size": [1.50, 0.55], "angle_rad": 0.0, "color": [0.75, 0.30, 0.08]},
                    {"center": [-4.45, 2.00], "size": [1.50, 0.55], "angle_rad": 0.0, "color": [0.75, 0.30, 0.08]},
                    {"center": [4.45, -2.20], "size": [1.50, 0.55], "angle_rad": 0.0, "color": [0.75, 0.30, 0.08]},
                    {"center": [4.45, 1.50], "size": [1.50, 0.55], "angle_rad": 0.0, "color": [0.75, 0.30, 0.08]}
                ]
            },
            "tight_corridors": {
                "label": "Tight alternating corridors",
                "episode_type": "survival",
                "goal_enabled": False,
                "progress_enabled": False,
                "random_start": True,
                "random_heading": True,
                "turn_macro_enabled": False,
                "spawn_clearance_m": 0.12,
                "spawn_attempts": 2000,
                # Long bars leave alternating end gates. The 1.15 m lanes are
                # narrow relative to the 0.8 m sensor range but remain wide
                # enough for the 25.5 x 14.5 cm vehicle to turn safely.
                "obstacles": [
                    {"center": [-0.60, 3.60], "size": [7.80, 0.45], "angle_rad": 0.0, "color": [0.10, 0.35, 0.62]},
                    {"center": [0.60, 2.00], "size": [7.80, 0.45], "angle_rad": 0.0, "color": [0.10, 0.35, 0.62]},
                    {"center": [-0.60, 0.40], "size": [7.80, 0.45], "angle_rad": 0.0, "color": [0.10, 0.35, 0.62]},
                    {"center": [0.60, -1.20], "size": [7.80, 0.45], "angle_rad": 0.0, "color": [0.10, 0.35, 0.62]},
                    {"center": [-0.60, -2.80], "size": [7.80, 0.45], "angle_rad": 0.0, "color": [0.10, 0.35, 0.62]},
                    {"center": [-4.35, 4.35], "size": [1.35, 0.45], "angle_rad": 0.7854, "color": [0.08, 0.48, 0.72]},
                    {"center": [4.35, 4.35], "size": [1.35, 0.45], "angle_rad": -0.7854, "color": [0.08, 0.48, 0.72]},
                    {"center": [-4.35, -4.35], "size": [1.35, 0.45], "angle_rad": -0.7854, "color": [0.08, 0.48, 0.72]},
                    {"center": [4.35, -4.35], "size": [1.35, 0.45], "angle_rad": 0.7854, "color": [0.08, 0.48, 0.72]}
                ]
            },
            "dense_pinch_points": {
                "label": "Dense staggered pinch points",
                "episode_type": "survival",
                "goal_enabled": False,
                "progress_enabled": False,
                "random_start": True,
                "random_heading": True,
                "turn_macro_enabled": False,
                "spawn_clearance_m": 0.12,
                "spawn_attempts": 3000,
                # Staggered blocks form short sight lines, diagonal approaches,
                # and repeated 0.7-1.0 m gaps without creating sealed rooms.
                "obstacles": [
                    {"center": [-3.55, 3.45], "size": [1.15, 0.95], "angle_rad": 0.20, "color": [0.42, 0.16, 0.58]},
                    {"center": [-1.55, 3.55], "size": [1.00, 1.25], "angle_rad": -0.18, "color": [0.42, 0.16, 0.58]},
                    {"center": [0.45, 3.35], "size": [1.20, 0.95], "angle_rad": 0.30, "color": [0.42, 0.16, 0.58]},
                    {"center": [2.65, 3.55], "size": [1.05, 1.20], "angle_rad": -0.25, "color": [0.42, 0.16, 0.58]},
                    {"center": [-2.55, 1.65], "size": [1.25, 1.00], "angle_rad": -0.30, "color": [0.52, 0.20, 0.66]},
                    {"center": [-0.35, 1.55], "size": [1.00, 1.30], "angle_rad": 0.18, "color": [0.52, 0.20, 0.66]},
                    {"center": [1.75, 1.65], "size": [1.30, 0.95], "angle_rad": -0.12, "color": [0.52, 0.20, 0.66]},
                    {"center": [3.70, 1.40], "size": [0.85, 1.25], "angle_rad": 0.25, "color": [0.52, 0.20, 0.66]},
                    {"center": [-3.65, -0.35], "size": [0.90, 1.30], "angle_rad": -0.18, "color": [0.60, 0.24, 0.62]},
                    {"center": [-1.55, -0.25], "size": [1.30, 0.90], "angle_rad": 0.28, "color": [0.60, 0.24, 0.62]},
                    {"center": [0.55, -0.45], "size": [1.00, 1.30], "angle_rad": -0.22, "color": [0.60, 0.24, 0.62]},
                    {"center": [2.70, -0.25], "size": [1.25, 0.95], "angle_rad": 0.15, "color": [0.60, 0.24, 0.62]},
                    {"center": [-2.65, -2.35], "size": [1.20, 1.00], "angle_rad": 0.22, "color": [0.48, 0.14, 0.52]},
                    {"center": [-0.45, -2.25], "size": [1.00, 1.30], "angle_rad": -0.25, "color": [0.48, 0.14, 0.52]},
                    {"center": [1.65, -2.45], "size": [1.30, 0.90], "angle_rad": 0.20, "color": [0.48, 0.14, 0.52]},
                    {"center": [3.65, -2.30], "size": [0.85, 1.25], "angle_rad": -0.15, "color": [0.48, 0.14, 0.52]},
                    {"center": [-3.55, -4.10], "size": [1.15, 0.70], "angle_rad": -0.12, "color": [0.35, 0.10, 0.46]},
                    {"center": [-1.40, -4.00], "size": [1.00, 0.90], "angle_rad": 0.18, "color": [0.35, 0.10, 0.46]},
                    {"center": [0.75, -4.10], "size": [1.20, 0.70], "angle_rad": -0.20, "color": [0.35, 0.10, 0.46]},
                    {"center": [2.95, -4.00], "size": [1.00, 0.90], "angle_rad": 0.16, "color": [0.35, 0.10, 0.46]}
                ]
            },
            "chessboard": {
                "label": "Chessboard survival grid",
                "episode_type": "survival",
                "goal_enabled": False,
                "progress_enabled": False,
                "random_start": True,
                "random_heading": True,
                "turn_macro_enabled": False,
                "spawn_clearance_m": 0.20,
                "spawn_attempts": 1000,
                # Alternating occupied squares leave several connected routes
                # through the same 10 x 10 metre arena.
                "obstacles": [
                    {"center": [-3.20, -3.20], "size": [0.72, 0.72], "angle_rad": 0.0},
                    {"center": [0.00, -3.20], "size": [0.72, 0.72], "angle_rad": 0.0},
                    {"center": [3.20, -3.20], "size": [0.72, 0.72], "angle_rad": 0.0},
                    {"center": [-1.60, -1.60], "size": [0.72, 0.72], "angle_rad": 0.0},
                    {"center": [1.60, -1.60], "size": [0.72, 0.72], "angle_rad": 0.0},
                    {"center": [-3.20, 0.00], "size": [0.72, 0.72], "angle_rad": 0.0},
                    {"center": [0.00, 0.00], "size": [0.72, 0.72], "angle_rad": 0.0},
                    {"center": [3.20, 0.00], "size": [0.72, 0.72], "angle_rad": 0.0},
                    {"center": [-1.60, 1.60], "size": [0.72, 0.72], "angle_rad": 0.0},
                    {"center": [1.60, 1.60], "size": [0.72, 0.72], "angle_rad": 0.0},
                    {"center": [-3.20, 3.20], "size": [0.72, 0.72], "angle_rad": 0.0},
                    {"center": [0.00, 3.20], "size": [0.72, 0.72], "angle_rad": 0.0},
                    {"center": [3.20, 3.20], "size": [0.72, 0.72], "angle_rad": 0.0}
                ]
            }
        },
        # These effective values are set from the selected map profile below.
        "random_start": False,
        "random_heading": False,
        "timeout_is_success": False,
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
            "survival_complete": 100.0,
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
        "enabled_in_evaluation": False,
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
        "directory": "runs/survival_mix_v4_baseline_eval_300"
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
        # Reserved for the later obstacle transfer-training run. Evaluation
        # loads the race checkpoint below and never writes model weights.
        "save_directory": "models/obstacle_transfer_v4_2000",
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
        "episodes": 300,
        "log_episodes": True,
        "summary_name": "evaluation_summary.json",
        "checkpoint": (
            "models/curriculum_v4_macro_final/"
            "stage_08_goal/dqn_best.pt"
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
if os.environ.get("RL_EVALUATION_EPISODES"):
    CONFIG["evaluation"]["episodes"] = int(
        os.environ["RL_EVALUATION_EPISODES"]
    )
if os.environ.get("RL_LOGGING_DIRECTORY"):
    CONFIG["logging"]["directory"] = os.environ["RL_LOGGING_DIRECTORY"]
if os.environ.get("RL_MAP_SELECTOR"):
    CONFIG["environment"]["map_selector"] = os.environ[
        "RL_MAP_SELECTOR"
    ].lower()


def _apply_selected_map_profile():
    environment = CONFIG["environment"]
    selected = environment["map_selector"]
    maps = environment["maps"]
    if selected not in maps:
        available = ", ".join(sorted(maps))
        raise ValueError(
            f'Unknown environment.map_selector "{selected}". '
            f"Available maps: {available}."
        )

    profile = maps[selected]
    episode_type = profile.get("episode_type", "goal")
    environment["random_start"] = bool(profile.get("random_start", False))
    environment["random_heading"] = bool(
        profile.get("random_heading", False)
    )
    environment["respawn_wall_clearance_m"] = float(
        profile.get(
            "spawn_clearance_m",
            environment["respawn_wall_clearance_m"]
        )
    )
    environment["goal"]["enabled"] = bool(
        profile.get("goal_enabled", episode_type == "goal")
    )
    environment["progress"]["enabled"] = bool(
        profile.get("progress_enabled", episode_type == "goal")
    )
    environment["timeout_is_success"] = episode_type == "survival"
    CONFIG["training"]["turn_macro"]["enabled"] = bool(
        profile.get("turn_macro_enabled", False)
    )


_apply_selected_map_profile()
