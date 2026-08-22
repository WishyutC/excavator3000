# Excavator3000 obstacle-avoidance training plan

## Experiment identity

| Item | Value |
|---|---|
| Run name | `obstacle_avoidance_survival_mix_100k_v1` |
| Purpose | Train the first production-oriented local obstacle-avoidance policy |
| Algorithm | Double Deep Q-Network (Double DQN) |
| Training budget | 100,000 episodes |
| Training maps | `obstacle_field`, `tight_corridors`, `dense_pinch_points` |
| Held-out test map | `chessboard` |
| Initialization | Fresh network; no race checkpoint and no replay history |
| Training log | `runs/obstacle_avoidance_survival_mix_100k_v1/training.csv` |
| Run manifest | `runs/obstacle_avoidance_survival_mix_100k_v1/run_manifest.json` |
| Checkpoints | `models/obstacle_avoidance_survival_mix_100k_v1/` |

This document freezes the intended configuration before training begins. It is
the experiment reference for the final technical report and presentation.

## Research question

Can a compact three-action DQN learn reusable obstacle avoidance from local
sensor and motion inputs when its Webots layout, position, and heading change
on every episode?

The policy is not trained to reach a map-specific goal. Its objective is to
continue driving without colliding or becoming stuck. This matches the planned
smart-tractor use case, where the neural network supplies high-level movement
decisions and the ESP32 hardware layer controls the physical motors.

## Pre-training baseline

The frozen race curriculum-v4 checkpoint was evaluated for 300 episodes before
obstacle training. Round-robin evaluation assigned exactly 100 episodes to each
training map.

| Map | Survived | Collided | Survival rate | Average steps |
|---|---:|---:|---:|---:|
| `tight_corridors` | 77 | 23 | 77% | 8,801.81 |
| `obstacle_field` | 40 | 60 | 40% | 5,876.63 |
| `dense_pinch_points` | 8 | 92 | 8% | 2,780.38 |
| **Overall** | **125** | **175** | **41.67%** | **5,819.61** |

The source report is
`runs/survival_mix_v4_baseline_eval_300/evaluation_summary.json`. The strong
difference between maps is important: aggregate survival alone can hide failure
on dense obstacles.

## Environment randomization

`environment.map_selector` is `survival_mix`. At every training reset:

1. one of the three training maps is selected randomly;
2. the immediately previous map is excluded from that draw;
3. a collision-free X/Y position is sampled;
4. a random heading from -180 to +180 degrees is sampled;
5. the 25.5 x 14.5 cm vehicle footprint and map-specific clearance are checked;
6. Webots physics and sensor state are reset.

Map selection is random rather than a fixed curriculum, so the policy cannot
depend on a known episode order. The CSV `map_name` field allows the final
report to verify the actual map distribution. Evaluation remains round-robin
for fair comparisons.

## State and action spaces

The model input is a 10-float vector:

- eight normalized proximity readings;
- signed normalized forward speed;
- signed normalized turn rate.

The output contains three Q-values:

| Action | Meaning |
|---:|---|
| 0 | Drive forward |
| 1 | Turn left |
| 2 | Turn right |

Track waypoints, map identity, spawn coordinates, and privileged Webots state
are not model inputs. This prevents the policy from memorizing a map and keeps
the interface suitable for the ESP32.

## Episode and reward configuration

- Maximum length: 10,000 Webots physics steps.
- `collision`: terminal reward -100.
- `stuck`: terminal reward -200.
- `survival_complete`: terminal reward +100.
- Safe forward motion receives a small positive shaping reward.
- Nearby obstacles receive an increasing danger penalty.
- Steering in clear space is penalized to reduce unnecessary oscillation.
- The race-track elapsed-time penalty and waypoint rewards are disabled.

Removing the time penalty is essential for survival training. Otherwise, a
late collision could produce a worse return than an immediate crash and teach
the agent to terminate early.

## DQN hyperparameters

| Parameter | Value | Reason for initial choice |
|---|---:|---|
| Hidden layers | 64, 64 | Compact enough for later embedded conversion |
| Discount factor | 0.999 | Preserves the value of long survival episodes |
| Learning rate | 0.0002 | Conservative updates for a noisy randomized task |
| Replay capacity | 50,000 transitions | Mixes experience across maps and spawns |
| Batch size | 64 | Stable learning with moderate GPU/CPU cost |
| Learning warmup | 2,000 transitions | Prevents learning from a nearly empty buffer |
| Train frequency | Every 4 decisions | Reduces correlated updates and compute cost |
| Target update | Every 1,000 optimizer steps | Stabilizes Bellman targets |
| Reward scale | 0.01 | Keeps neural-network targets numerically controlled |
| Gradient clipping | 10.0 | Limits unstable gradient spikes |
| Epsilon start | 1.00 | Full exploration at the beginning |
| Epsilon end | 0.05 | Retains limited exploration late in training |
| Epsilon decay | 1,000,000 environment decisions | Avoids committing too early to the race policy's behavior |
| Action repeat | 4 physics steps | Reduces control noise while retaining local reactions |
| Seed | 42 | Makes software-side randomness reproducible |

These values form the first obstacle-training baseline. They are not yet
claimed to be optimal. Hyperparameter tuning must compare fresh runs under the
same evaluation protocol rather than changing values during this run.

## Checkpoint and recovery plan

Training starts with `resume = False`, so existing race and baseline artifacts
cannot be overwritten. The controller writes:

- `dqn_latest.pt` every 50 episodes and at clean shutdown;
- `dqn_best.pt` when a successful episode exceeds the previous successful
  return;
- `dqn_candidate.pt` for the strongest non-successful episode.

If Webots or the computer stops, set `training.resume = True` and restart using
the same model and log directories. The latest checkpoint restores network,
optimizer, epsilon, and training counters. Never rename or clear the active CSV
when resuming.

## Evidence captured for reports and presentations

`training.csv` stores one row per completed episode with:

- global and stage episode numbers;
- active map;
- physics steps and DQN decisions;
- total and mean reward;
- termination reason and success flag;
- epsilon, replay size, optimizer-step count, and latest loss;
- forward/left/right action percentages;
- race progress fields, which remain zero for survival maps.

`run_manifest.json` is created once at the start and preserves the complete
effective configuration, run description, Python version, and CSV schema. It is
not overwritten on resume.

The final report should show:

1. survival rate over rolling windows of 100 and 1,000 episodes;
2. per-map survival and collision rates;
3. average and median episode length per map;
4. reward trend, with survival rate treated as the primary metric;
5. epsilon and loss trends;
6. action distribution to expose forward-only or steering-collapse behavior;
7. final round-robin evaluation against the 41.67% baseline;
8. held-out `chessboard` performance to measure generalization.

## Before pressing Start

The saved configuration is prepared with:

```text
program.mode = train
environment.map_selector = survival_mix
training.episodes = 100000
training.curriculum.enabled = False
training.resume = False
observer.enabled_in_training = False
```

Use headless fast mode for the long run. Confirm that the first episodes create
the new CSV, run manifest, and checkpoint directory. Do not use the held-out
`chessboard` map for training, and do not change rewards or hyperparameters
mid-run; changes would make the 100,000 episodes scientifically inconsistent.

One hundred thousand episodes can represent hundreds of millions of Webots
physics steps. The run may take a long time even in fast, non-rendered mode.
Checkpointing and resume support are therefore part of the experiment design,
not optional conveniences.
