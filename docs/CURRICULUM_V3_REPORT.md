# Excavator3000 DQN Curriculum v3 Report

Status: **superseded**

This report records the v3 investigation that isolated the stage-5 design
blocker. The blocker was resolved by the segmented turn controller, and the
completed 8-stage results are documented in
[`MACRO_CURRICULUM_V4_REPORT.md`](MACRO_CURRICULUM_V4_REPORT.md).

## Baseline preservation

No baseline data was deleted. The runtime artifacts were renamed and preserved as:

- `runs_baseline_v1` / `models_baseline_v1`
- `runs_baseline_v2` / `models_baseline_v2`
- `runs_old` / `models_old`

The active v3 artifacts use `runs/curriculum_v3` and
`models/curriculum_v3`.

## Implemented foundation

- Eight-stage checkpoint curriculum with policy-only transfer.
- Fresh optimizer, replay buffer, counters, and epsilon schedule per stage.
- Explicit `curriculum_complete` termination and reward.
- Collision priority over real-goal and curriculum success.
- Latest-stage checkpoint selection, avoiding lucky early random checkpoints.
- Safe curriculum resume from an already validated stage.
- Sensor-guided demonstrations plus an imitation auxiliary loss; this does not
  change the deployed DQN architecture.
- Dashboard stage, stage-episode, success, collision, stuck, and progress data.
- Dashboard start/pause/resume/stop control remains available on port 8080.
- Headless fast Webots launch with controller stdout/stderr forwarding.
- Measured tight-turn primitives: left `+0.955 rad/s`, right `-0.908 rad/s`.

## Validated training results

Every promoted policy was evaluated greedily for 50 episodes with no forced
action, waypoint expert, sensor expert, or epsilon exploration.

| Stage | Guided episodes | Greedy success | Mean eval reward | Progress |
|---|---:|---:|---:|---:|
| `stage_01_cp1` | 200 | 50/50 (100%) | 130.88 | 1/10 |
| `stage_02_cp2` | 300 | 50/50 (100%) | 168.65 | 2/10 |
| `stage_03_cp3` | 500 | 50/50 (100%) | 206.07 | 3/10 |
| `stage_04_cp4` | 300 | 50/50 (100%) | 255.55 | 4/10 |

The active CSV contains 1,300 validated guided episodes through stage 4 plus
300 preserved stage-5 continuation episodes. Every guided episode reached its
configured stage target with zero collision and zero stuck terminations.

### Validated checkpoint hashes (SHA-256)

- Stage 1: `5220E48082393503ABCE256C68A7E9B26871C730BA5CD4652BC3BBAC285ABB24`
- Stage 2: `44F775A35390A9B056E493287671B6A6BFDD03D3A3B1C8056CA8F5F64CF84098`
- Stage 3: `62B3D5A38EA9FD8E41D26DFCFB70351102E08E8C4827D83EABC4CD3261E82AAD`
- Stage 4: `07C83292580DA461D288FF83478FAA084D6942C629E62EC2B6FF2CC1A310C826`

The non-promoted stage-5 diagnostic checkpoint is preserved with SHA-256
`E778E3092F69D147BE2F1553AC13CCDD5E9E13853A0BA94FE58408EB250CFB9C`.

## Stage 5 observation blocker

Stage 4 was solved with a training-only, waypoint-aligned teacher and passed
50/50 expert-free greedy episodes. Stage 5 then received 100 guided episodes,
failed 0/50 greedy episodes at checkpoint 5, continued from that checkpoint
for 300 more guided episodes, and again failed 0/50 at checkpoint 5.

A focused greedy diagnostic terminated as `stuck` at position
`(3.589, 3.511)`, facing south toward checkpoint 6. The policy was still
selecting right-turn action 2 with normalized turn rate `-0.841`, even though
the correct action was forward. The current observation contains obstacle
proximities, speed, and instantaneous turn rate, but no accumulated turn angle
or action history. The memoryless MLP therefore cannot distinguish "continue
turning" from "the turn is complete; drive forward" when the old corner wall
remains visible.

Additional stage-5 episodes did not improve the result, establishing that this
is missing state rather than insufficient training time or reward tuning.

## Required design decision

One of the following is needed before stages 5–8 can be validly completed:

1. **Calibrated turn macro-action (recommended):** the DQN selects left or
   right once, and the ESP32 motor controller completes the turn using a tuned
   time or gyro angle before requesting another DQN decision. This keeps the
   exported network at 10 inputs and matches the hardware-team tuning split.
2. **IMU heading/turn-progress input:** add one or two normalized orientation
   features and retrain the input layer. This gives the DQN direct awareness of
   accumulated rotation but adds a hardware sensor requirement.
3. **Recurrent policy:** replace the MLP with a memory-capable network. This is
   heavier to train and deploy on ESP32.

Training is paused rather than promoting a policy that only succeeds because
of hidden simulator information. The dashboard stays running, stages 1–4 are
validated, and both failed stage-5 checkpoints are preserved for diagnosis.
