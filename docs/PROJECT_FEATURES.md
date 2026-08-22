---
title: Excavator3000 Project Features and Training Guide
author: Excavator3000 Team
date: 2026-08-22
---

# Excavator3000

## Project status

Excavator3000 is a Webots reinforcement-learning controller intended to become
the high-level driving brain of an ESP32 RC vehicle. The learned policy chooses
forward, left, or right. The hardware layer remains responsible for calibrated
TT-motor PWM, acceleration limits, steering balance, and emergency stopping.

| Phase | Status | Result |
|---|---|---|
| 1. Environment foundation | Complete | Sensors, normalized observation, reset, goal, collision, rewards, HUD, and explicit termination |
| 2. Race-track DQN | Complete and validated | Replay learning, checkpoints, curriculum, evaluation, CSV metrics, and a full-goal policy |
| 3. Obstacle generalization | In progress | Three-map randomized survival pool, safe random spawns, held-out map, and balanced evaluation |
| 4. Model conversion | Not started | Export, TensorFlow Lite conversion, quantization, and numerical comparison |
| 5. ESP32 integration | Not started | Real-sensor calibration, inference, motor tuning, and safety layer |

## Model interface

The policy receives a stable 10-float observation: eight normalized proximity
values, signed normalized forward speed, and signed normalized turn rate. Raw
Webots readings stay available for collision detection and observer telemetry.
The DQN is a compact multilayer perceptron:

```text
10 inputs -> Dense 64 + ReLU -> Dense 64 + ReLU -> 3 Q-values
```

The three outputs correspond to forward, left, and right. This network has
5,059 float32 parameters (about 19.8 KiB before conversion). Replay memory and
the PyTorch optimizer are training-only and will not be deployed to the ESP32.

## Environment modes

The same 10 x 10 metre Webots world supports these selectors:

| Selector | Use |
|---|---|
| `race_track` | Original goal route with ten curriculum checkpoints |
| `survival_mix` | Training/evaluation pool that rotates among three obstacle maps |
| `obstacle_field` | Irregular field with blocked perimeter shortcuts |
| `tight_corridors` | Alternating narrow lanes and end gates |
| `dense_pinch_points` | Dense staggered blocks and short sight lines |
| `chessboard` | Held-out generalization test, excluded from training |

Survival maps are generated at reset as locked Webots `Solid` nodes with
matching `boundingObject` boxes. The race walls and goal are hidden while a
survival map is active. Random spawns account for the real 25.5 x 14.5 cm car
footprint, wall clearance, obstacle clearance, and a random heading.

Training selects a different pool map randomly after each episode. Evaluation
uses deterministic round-robin selection so every map receives an equal sample.
See [SURVIVAL_MAPS.md](SURVIVAL_MAPS.md) for layout and benchmark details.

## Episode semantics

The maximum episode length is 10,000 physics steps. Track mode can end with
goal/curriculum success, collision, stuck detection, timeout, or simulation
stop. Survival mode intentionally has no goal or waypoint shaping and ends with:

| Reason | Meaning |
|---|---|
| `collision` | A proximity sensor crosses the collision threshold |
| `survival_complete` | The policy remains alive for all 10,000 steps |
| `stuck` | Progress has stopped long enough to trigger the detector |
| `simulation_stopped` | Webots stops the controller |

Collision takes priority over success. Survival mode removes the track's
elapsed-time penalty so a late collision is not worse than an immediate crash.
Danger, stopping, unnecessary steering, and collision penalties remain active.

The track-specific committed-turn macro is disabled automatically on survival
maps. That gives the policy frequent local decisions for obstacle avoidance.

## DQN training foundation

The current implementation uses Double DQN with:

- discount factor 0.999;
- Adam learning rate 0.0002;
- Huber loss and gradient clipping;
- 50,000-transition uniform replay memory;
- batch size 64 and 2,000-transition warmup;
- one optimizer update every four agent steps;
- target-network synchronization every 1,000 updates;
- epsilon decay from 1.00 to 0.05 over 1,000,000 environment steps.

Checkpoints contain the online and target networks, optimizer state, counters,
architecture metadata, and the episode summary. `dqn_latest.pt` is resumable;
`dqn_best.pt` is the strongest saved candidate. Curriculum completion evaluates
both candidates and preserves the stronger deployment checkpoint.

Model and experiment folders are ignored by Git. Use a unique `save_directory`
and logging directory for every experiment so the validated race baseline is
never overwritten.

## Observer and dashboard

The Tkinter observer receives localhost UDP telemetry and shows eight sensor
blocks around the vehicle, estimated distance, speed, action, map, reward,
episode state, epsilon, loss, and replay usage. It is enabled for interactive
testing and disabled by default during headless training and evaluation.

The web dashboard reads the training CSV and provides practical controls for
starting/pausing Webots and editing supported `config.py` values. Its map field
includes every individual selector plus `survival_mix`.

## Main modules

| File | Responsibility |
|---|---|
| `rl_controller.py` | Webots entry point and mode dispatch |
| `robot_controller.py` | Supervisor, motors, sensors, pose reset, and telemetry |
| `map_manager.py` | Dynamic map nodes, pool rotation, and safe random spawning |
| `environment.py` | RL reset/step/action/observation interface |
| `episode_manager.py` | Collision, goal, timeout, stuck, and survival termination |
| `progress_tracker.py` | Race curriculum checkpoint progress |
| `reward_calculator.py` | Track/survival-aware reward calculation |
| `dqn_agent.py` | Exploration, Double DQN learning, and checkpoints |
| `curriculum_trainer.py` | Staged track curriculum and candidate selection |
| `dqn_trainer.py` | Ordinary training and balanced evaluation orchestration |
| `training_logger.py` | Episode CSV including active map name |
| `hud_gui.py` | External observer window |
| `dashboard.py` | Training dashboard server and HTTP routes |
| `dashboard_process.py` | Webots process controls |
| `config.py` | Central program configuration and environment overrides |

## Running and validation

Install PC training dependencies:

```powershell
python -m pip install -r requirements-training.txt
```

Run the automated checks:

```powershell
python -m unittest discover -s tests -v
python -m py_compile config.py curriculum_trainer.py dqn_agent.py dqn_network.py dqn_trainer.py environment.py episode_manager.py map_manager.py progress_tracker.py reward_calculator.py rl_controller.py robot_controller.py training_logger.py
```

The suite currently contains 78 tests. Real Webots smoke tests additionally
validated all three pool maps, full-length survival episodes, and map rotation.

## Current evidence and next milestone

The curriculum-v4 best checkpoint can complete the full ten-checkpoint race
route. A frozen 2,000-episode obstacle-field benchmark achieved 39.1% survival
with no obstacle-specific retraining. This establishes a useful baseline but is
not sufficient for autonomous obstacle avoidance.

The next milestone is to finish the balanced three-map evaluation, train a new
policy on `survival_mix`, and compare it on both the training maps and held-out
`chessboard`. After obstacle behavior is stable, export and quantize the compact
network, verify converted outputs against PyTorch, then calibrate the ESP32
sensor and motor layers.

## Sim-to-real boundary

The neural network decides *what* maneuver to perform; it does not directly
emit TT-motor PWM. Deployment must preserve the observation order and
normalization, then independently implement motor dead-zone compensation,
left/right balance, acceleration limits, and emergency collision protection.
The real 200-250 RPM motors with 68 mm wheels do not need to match Webots raw
speed exactly, but normalized speed and action meaning must stay consistent.
