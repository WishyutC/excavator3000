# Excavator3000

Excavator3000 is a Webots and PyTorch reinforcement-learning project for a
self-driving ESP32 RC vehicle. The neural network acts as the vehicle's
high-level brain: it reads eight obstacle sensors plus motion feedback and
chooses **forward**, **left**, or **right**. The future ESP32 motor layer will
translate those decisions into calibrated TT-motor PWM and steering commands.

The simulation environment, DQN pipeline, replay memory, checkpointing,
dashboard, observer HUD, race curriculum, randomized obstacle maps, and
evaluation system are implemented. The repository is now prepared for its first
100,000-episode obstacle-avoidance training run.

> The validated race model is not yet the final ESP32 obstacle-avoidance model.
> Survival training, final evaluation, model conversion, quantization, and
> hardware calibration still remain.

## Current progress

| Area | Status |
|---|---|
| Webots controller and sensor integration | Complete |
| Normalized state and discrete action spaces | Complete |
| Reset, random spawn, and episode lifecycle | Complete |
| Goal, checkpoint, collision, stuck, and timeout detection | Complete |
| Modular reward function | Complete |
| Uniform replay buffer | Complete |
| Double DQN training pipeline | Complete |
| CSV logs, run manifest, and checkpoints | Complete |
| Tkinter observer HUD | Complete |
| Local/Cloudflare-compatible web dashboard | Complete |
| Ten-stage race curriculum and goal policy | Complete and validated |
| Three-map randomized survival environment | Complete and validated |
| Frozen pre-training survival benchmark | Complete: 300 episodes |
| 100,000-episode obstacle training | Configured, not started |
| Hyperparameter tuning | Pending first survival-training result |
| Held-out-map evaluation | Pending trained survival model |
| TensorFlow Lite conversion and quantization | Not started |
| ESP32 deployment and motor calibration | Not started |

## How the system works

```text
Webots world
   |
   +-- 8 proximity sensors
   +-- forward speed
   +-- turn rate
   |
   v
10-value normalized observation
   |
   v
Double DQN: 10 -> 64 -> 64 -> 3 Q-values
   |
   +-- action 0: forward
   +-- action 1: turn left
   +-- action 2: turn right
   |
   v
Webots wheel command now / calibrated ESP32 motor command later
```

The network has 5,059 float32 parameters, approximately 19.8 KiB of raw
weights before conversion. The replay buffer, optimizer, training code, Webots
Supervisor, dashboard, and HUD stay on the PC; only the final inference model
and input/output logic belong on the ESP32.

## Repository and Webots layout

This Git repository contains the controller, not the complete Webots project.
The expected local layout is:

```text
Bot_sim/
|-- worlds/
|   `-- car_env_test_v2.wbt
`-- controllers/
    `-- rl_controller/        <- this repository
        |-- README.md
        |-- config.py
        |-- rl_controller.py
        `-- ...
```

The robot node in `car_env_test_v2.wbt` must use the `rl_controller` controller
and provide Supervisor access. Race mode additionally expects a
`DEF GOAL Solid` node. Survival mode dynamically creates locked obstacle
`Solid` nodes with matching `boundingObject` boxes inside the same world.

## Requirements

- Windows 10 or 11
- Webots installed in the standard location, or `WEBOTS_EXE` configured
- Python compatible with the pinned requirements
- Git
- NVIDIA CUDA is optional; training also supports CPU
- Cloudflare Tunnel is optional and only needed for remote dashboard access

The current training environment has been verified with Python 3.13 and an
NVIDIA RTX 5060 Ti.

## Installation

Clone the controller into the Webots project structure shown above:

```powershell
git clone https://github.com/WishyutC/excavator3000.git `
  "C:\Users\titan\OneDrive\Documents\Bot_sim\controllers\rl_controller"
cd "C:\Users\titan\OneDrive\Documents\Bot_sim\controllers\rl_controller"
```

Install training and dashboard dependencies:

```powershell
python -m pip install -r requirements-training.txt
python -m pip install -r requirements-dashboard.txt
```

Verify PyTorch and CUDA:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Run the automated checks:

```powershell
python -m unittest discover -s tests -v
python -m py_compile config.py curriculum_trainer.py dashboard.py dashboard_config.py dashboard_data.py dashboard_process.py dqn_agent.py dqn_network.py dqn_trainer.py environment.py episode_manager.py hud_gui.py map_manager.py observation.py progress_tracker.py replay_buffer.py reward_calculator.py rl_controller.py robot_controller.py training_logger.py
```

The current suite contains 80 tests.

## Central configuration

All operating settings live in [`config.py`](config.py). Important groups are:

| Group | Controls |
|---|---|
| `program` | `test`, `diagnostic`, `train`, or `evaluate` mode |
| `simulation` | Webots time step and motor-speed ceiling |
| `robot` | Vehicle dimensions, speed scale, and wheel ratios |
| `observation` | Sensor range and speed/turn normalization |
| `environment` | Map, episode length, reset, collision, rewards, and progress |
| `observer` | HUD launch and telemetry behavior |
| `logging` | Run identity, description, baseline reference, and CSV directory |
| `training` | DQN, replay, epsilon, checkpoints, curriculum, and resume behavior |
| `evaluation` | Checkpoint, episode count, logging, and summary filename |

Supported one-run environment overrides are:

```text
RL_PROGRAM_MODE
RL_TRAINING_EPISODES
RL_EVALUATION_EPISODES
RL_LOGGING_DIRECTORY
RL_MAP_SELECTOR
```

Example PowerShell smoke override:

```powershell
$env:RL_PROGRAM_MODE = "train"
$env:RL_TRAINING_EPISODES = "3"
$env:RL_LOGGING_DIRECTORY = "runs/smoke_test"
$env:RL_MAP_SELECTOR = "survival_mix"
```

Remove the overrides afterward so they do not silently affect later runs:

```powershell
Remove-Item Env:RL_PROGRAM_MODE -ErrorAction SilentlyContinue
Remove-Item Env:RL_TRAINING_EPISODES -ErrorAction SilentlyContinue
Remove-Item Env:RL_LOGGING_DIRECTORY -ErrorAction SilentlyContinue
Remove-Item Env:RL_MAP_SELECTOR -ErrorAction SilentlyContinue
```

## Environment modes and maps

| Selector | Episode type | Purpose |
|---|---|---|
| `race_track` | Goal | Original ten-checkpoint curriculum route |
| `survival_mix` | Survival | Random pool of the three training maps |
| `obstacle_field` | Survival | Irregular obstacles and blocked perimeter shortcuts |
| `tight_corridors` | Survival | Alternating narrow lanes and end gates |
| `dense_pinch_points` | Survival | Dense blocks, short sight lines, and tight gaps |
| `chessboard` | Survival | Held-out generalization map; never used for training |

For `survival_mix`, training randomly selects a map at every reset and prevents
the same map from appearing twice consecutively. Position and heading are also
randomized. Spawn validation includes the complete 25.5 x 14.5 cm vehicle
footprint, obstacle clearance, and arena-wall clearance.

Evaluation uses round-robin map selection. This produces an equal episode count
per map and makes comparisons repeatable.

## State space

The policy receives exactly ten normalized floats in a fixed order:

| Index | Input | Range |
|---:|---|---:|
| 0 | Front proximity | 0 to 1 |
| 1 | Back proximity | 0 to 1 |
| 2 | Left proximity | 0 to 1 |
| 3 | Right proximity | 0 to 1 |
| 4 | Left-front proximity | 0 to 1 |
| 5 | Right-front proximity | 0 to 1 |
| 6 | Left-back proximity | 0 to 1 |
| 7 | Right-back proximity | 0 to 1 |
| 8 | Signed forward speed | -1 to 1 |
| 9 | Signed turn rate | -1 to 1 |

Map name, spawn coordinates, waypoint coordinates, and other privileged Webots
state are not provided to the model. This forces it to learn local behavior
that can transfer to a real vehicle.

## Episode outcomes and rewards

Race episodes can finish through goal success, curriculum-target success,
collision, stuck detection, timeout, or simulation stop. Survival episodes use:

| Reason | Result |
|---|---|
| `collision` | Failure; a proximity reading exceeds the collision threshold |
| `stuck` | Failure; movement/progress remains too low for too long |
| `survival_complete` | Success; the car survives all 10,000 physics steps |
| `simulation_stopped` | Webots/controller shutdown |

The survival reward combines:

- +100 for completing the full episode;
- -100 for collision;
- -200 for becoming stuck;
- a small safe-forward-motion reward;
- an increasing penalty near obstacles;
- a penalty for unnecessary steering in clear space.

Survival mode disables race waypoint rewards and the elapsed-time penalty. This
prevents a late collision from becoming worse than an immediate crash.

## Observer HUD

[`hud_gui.py`](hud_gui.py) is an external Tkinter observer that receives JSON
telemetry over UDP at `127.0.0.1:8765`. It displays:

- eight sensor blocks around the vehicle;
- raw readings, estimated distance, and danger colors;
- speedometer and turn rate;
- active action and map;
- episode, reward, and termination state;
- epsilon, replay size, training updates, and loss.

The HUD is enabled for interactive testing. It is disabled by default during
training and evaluation because GUI updates reduce simulation speed.

## Web training dashboard

The dashboard provides live CSV visualization, approved `config.py` controls,
and start/pause/resume/stop controls for headless Webots.

Start it by double-clicking `Start_dashboard.bat`, or run:

```powershell
python dashboard.py
```

Then open `http://127.0.0.1:8080` and enter the token printed in the terminal.
The token is stored locally in `.dashboard-token` and must not be shared.

Dashboard process controls:

- **Start** launches Webots with `--batch --mode=fast --no-rendering`.
- **Pause** suspends the Webots process tree without closing it.
- **Resume** continues a paused process.
- **Stop** terminates Webots; progress after the last checkpoint may be lost.

Configuration edits apply when Webots starts again. Restart the dashboard after
changing `logging.directory`, because its CSV reader selects the log when the
dashboard server starts.

For remote access, keep the authenticated dashboard running and point a
Cloudflare Tunnel at:

```text
http://localhost:8080
```

Example Quick Tunnel:

```powershell
cloudflared tunnel --url http://localhost:8080
```

See [`docs/DASHBOARD.md`](docs/DASHBOARD.md) for security and operating notes.

## Run Webots directly

The recommended long-run method is the dashboard. To launch Webots directly
from PowerShell:

```powershell
& "C:\Program Files\Webots\msys64\mingw64\bin\webots.exe" `
  --batch --mode=fast --no-rendering --stdout --stderr `
  "C:\Users\titan\OneDrive\Documents\Bot_sim\worlds\car_env_test_v2.wbt"
```

In PowerShell, `&` must appear before the quoted executable path. Do not place
it between the Webots arguments and world path.

For visual testing, open the world normally in Webots, select `test` mode in
`config.py`, and press Play. The observer HUD will open automatically.

## Prepared 100,000-episode training run

The repository is currently configured for:

```text
program.mode = train
environment.map_selector = survival_mix
training.episodes = 100000
training.curriculum.enabled = False
training.resume = True
observer.enabled_in_training = False
```

Outputs are isolated from every previous experiment:

```text
runs/obstacle_avoidance_survival_mix_100k_v1/
|-- training.csv
`-- run_manifest.json

models/obstacle_avoidance_survival_mix_100k_v1/
|-- dqn_latest.pt
|-- dqn_best.pt
`-- dqn_candidate.pt
```

or hyperparameters during the experiment; that would mix different experiments
inside one CSV.
Before starting, confirm no other Webots process owns the world. Then use the
dashboard **Start** button or the direct headless command. Resume mode is
outage-safe: it starts fresh when the dedicated directory has no checkpoint and
automatically restores `dqn_latest.pt` after checkpoints begin. Do not change
rewards or hyperparameters during the experiment; that would mix different
experiments inside one CSV.
or hyperparameters during the experiment; that would mix different experiments
inside one CSV.

One hundred thousand episodes can represent hundreds of millions of physics
steps. Use pause/resume and checkpoints instead of expecting one uninterrupted
session.

The complete frozen configuration and reporting plan are in
[`docs/OBSTACLE_AVOIDANCE_100K_TRAINING_PLAN.md`](docs/OBSTACLE_AVOIDANCE_100K_TRAINING_PLAN.md).

## Resume interrupted training

`dqn_latest.pt` is saved every 50 episodes and at a clean training shutdown. To
continue after Webots or the computer stops:

1. Confirm `models/obstacle_avoidance_survival_mix_100k_v1/dqn_latest.pt`
   exists.
2. Keep the same model and logging directories.
3. Set `training.resume` to `True`.
4. Keep `training.episodes` at the final target, not the number of additional
   episodes.
5. Restart Webots.

The checkpoint restores online/target networks, optimizer state, replay memory,
epsilon and training counters. Before appending new data, the logger removes any
CSV rows newer than the restored checkpoint and records that rollback in
`resume_events.jsonl`. Never rename or clear the active checkpoint or CSV before
resuming.

To intentionally begin a different experiment, set `resume = False` and choose
new model and log directory names.

## Logs and checkpoints

Every completed episode appends one CSV row containing:

- episode and map name;
- physics steps and DQN decision count;
- total and mean reward;
- termination reason and success;
- epsilon, replay size, optimizer updates, and latest loss;
- forward/left/right action percentages;
- track progress fields used by race experiments.

`run_manifest.json` is created once when a run starts. It records the complete
effective configuration, run description, Python version, and CSV schema. It is
not overwritten when training resumes, preserving the original experiment
definition for reports and presentations.

Checkpoint meanings:

| File | Meaning |
|---|---|
| `dqn_latest.pt` | Latest resumable training state |
| `dqn_best.pt` | Highest-return successful episode observed so far |
| `dqn_candidate.pt` | Strongest non-successful candidate |

The final deployable checkpoint must be selected through frozen evaluation, not
only by one episode's reward.

Generated `models/` and `runs/` data are ignored by Git. Historical experiments
are preserved outside the active directories in `model-old-unused/` and
`run-old-unused/`.

## Evaluate a checkpoint

For evaluation:

1. Set `program.mode` to `evaluate`.
2. Set `evaluation.checkpoint` to the exact `.pt` file.
3. Set a new `logging.directory` so training data is never overwritten.
4. Choose `survival_mix` for equal round-robin testing across training maps or
   `chessboard` for held-out generalization.
5. Set `evaluation.episodes`.
6. Restart Webots in headless mode.

Evaluation disables exploration and learning. It writes episode rows plus
`evaluation_summary.json`, containing overall and per-map success, reward,
steps, and termination counts.

## Baseline results

Before survival training, the frozen race curriculum-v4 model completed a
300-episode balanced `survival_mix` evaluation:

| Map | Survival | Collision | Average steps |
|---|---:|---:|---:|
| `tight_corridors` | 77% | 23% | 8,801.81 |
| `obstacle_field` | 40% | 60% | 5,876.63 |
| `dense_pinch_points` | 8% | 92% | 2,780.38 |
| **Overall** | **41.67%** | **58.33%** | **5,819.61** |

This is the official comparison point for the new model. It shows that race
training transferred some useful behavior, while dense obstacle avoidance still
requires dedicated training.

The earlier 2,000-episode single-layout benchmark achieved 39.1% survival on
`obstacle_field`. Both reports are kept locally under `runs/` and excluded from
Git because experiment artifacts can become large.

## Hyperparameter tuning workflow

The current parameters are a controlled first baseline, not a final optimum.
After the 100k run:

1. evaluate the selected checkpoint on the same 300-episode `survival_mix`
   protocol;
2. compare overall and per-map survival against 41.67%;
3. evaluate separately on held-out `chessboard`;
4. inspect reward, loss, epsilon, episode length, and action distribution;
5. change one small parameter group for a new run;
6. train from a fresh network with new artifact directories;
7. repeat the identical evaluations.

Tune epsilon decay, learning rate, reward balance, and action repeat first.
Avoid changing network size or discount factor until evidence shows they are the
limiting factors.

## Source overview

| File | Responsibility |
|---|---|
| `rl_controller.py` | Webots entry point and mode dispatch |
| `robot_controller.py` | Supervisor, motors, sensors, reset, and telemetry |
| `map_manager.py` | Dynamic maps, random rotation, and safe spawning |
| `environment.py` | RL reset/step/action/observation interface |
| `episode_manager.py` | Goal, collision, timeout, stuck, and survival outcomes |
| `observation.py` | Hardware-friendly normalized state construction |
| `progress_tracker.py` | Race curriculum checkpoint progress |
| `reward_calculator.py` | Modular track and survival rewards |
| `replay_buffer.py` | Uniform fixed-capacity replay memory |
| `dqn_network.py` | Compact PyTorch Q-network |
| `dqn_agent.py` | Exploration, Double DQN learning, and checkpoints |
| `dqn_trainer.py` | Training and evaluation orchestration |
| `curriculum_trainer.py` | Staged race curriculum and candidate evaluation |
| `training_logger.py` | Episode CSV and immutable run manifest |
| `hud_gui.py` | External Tkinter telemetry HUD |
| `dashboard.py` | Authenticated dashboard server |
| `dashboard_process.py` | Headless Webots process controls |
| `dashboard_data.py` | CSV metrics and rolling summaries |
| `dashboard_config.py` | Safe editing of approved configuration fields |
| `config.py` | Central configuration and map definitions |

## Documentation

| Document | Purpose |
|---|---|
| [`PROJECT_FEATURES.md`](docs/PROJECT_FEATURES.md) | Current architecture, features, and roadmap |
| [`MODEL_ARCHITECTURE.md`](docs/MODEL_ARCHITECTURE.md) | Exact input preprocessing, network layers, Q-value outputs, and ESP32 contract |
| [`ESP32 deployment package`](esp32%20model%20deployment/README.md) | Shareable race-model copy, TFLite export, verification report, and conversion tools |
| [`OBSTACLE_AVOIDANCE_100K_TRAINING_PLAN.md`](docs/OBSTACLE_AVOIDANCE_100K_TRAINING_PLAN.md) | Frozen 100k experiment design and presentation evidence |
| [`SURVIVAL_MAPS.md`](docs/SURVIVAL_MAPS.md) | Map generation, spawn rules, and survival semantics |
| [`MACRO_CURRICULUM_V4_REPORT.md`](docs/MACRO_CURRICULUM_V4_REPORT.md) | Validated race curriculum-v4 results |
| [`DASHBOARD.md`](docs/DASHBOARD.md) | Dashboard and Cloudflare Tunnel usage |
| [`TRAINING_UPGRADE_LOG.md`](docs/TRAINING_UPGRADE_LOG.md) | Historical DQN training improvements |
| [`CURRICULUM_V3_REPORT.md`](docs/CURRICULUM_V3_REPORT.md) | Historical curriculum-v3 experiments |

## ESP32 deployment boundary

The final policy will choose a maneuver; it will not directly output motor PWM.
The embedded integration must provide:

- the same eight sensor positions and normalization order;
- measured forward-speed and turn-rate normalization;
- calibrated action-to-PWM or action-to-RPM mapping;
- TT-motor dead-zone compensation and left/right balancing;
- acceleration limiting;
- emergency stop logic independent of the neural network;
- converted-model numerical checks against PyTorch;
- TensorFlow Lite Micro memory and latency validation.

The real vehicle uses 200-250 RPM TT motors and 68 mm wheels. Raw Webots motor
speed does not need to equal real free speed exactly, but sensor normalization,
speed awareness, action meaning, and safety behavior must remain consistent.

## Safety and experiment rules

- Do not commit `.pt`, run CSV, dashboard tokens, or local experiment archives.
- Do not overwrite a validated model with a new experiment.
- Do not mix changed hyperparameters into an existing run directory.
- Do not train on `chessboard`; keep it unseen for generalization testing.
- Do not treat a high reward alone as proof of safe driving.
- Always compare overall and per-map survival, especially
  `dense_pinch_points`.
- Keep a hardware emergency stop outside the learned policy.

## License

No license file is currently included. Add an explicit license before treating
the repository as open-source or redistributing it outside the project team.
