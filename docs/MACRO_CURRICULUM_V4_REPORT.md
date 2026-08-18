# Macro Curriculum V4 Report

## Outcome

The compact DQN now completes the Webots track and reaches the actual `GOAL`
node. The final policy passed 50 out of 50 greedy goal evaluations without the
training expert.

This is one shared neural network. The curriculum checkpoints are progressive
training targets and saved snapshots, not separate models. Each stage transfers
the same policy weights into the next stage.

## Why the previous policy failed

The earlier primitive action repeated a turn for only four physics steps. Near
checkpoint 5, multiple distance sensors became active together and the DQN
could rapidly change its decision. The robot visibly shook and the stage-5
policy scored 0 out of 50 greedy successes at every periodic gate, including
after 600 additional guided episodes.

Extra episodes did not solve this control-level ambiguity.

## Segmented turn controller

The DQN action space remains:

1. forward
2. turn left
3. turn right

A turn decision now invokes one low-level maneuver:

1. commit to the requested direction;
2. rotate to an approximately 100-degree oversteer angle;
3. travel diagonally long enough to translate the 14.5 cm by 25.5 cm chassis
   into the next lane;
4. straighten to approximately 90 degrees;
5. use distance-sensor feedback and heading hold for exit stabilization;
6. return control to the DQN for its next high-level decision.

This prevents contradictory sensor frames from reversing a turn before the
maneuver finishes. The same design can be deployed on the ESP32 using an IMU,
wheel encoders, or hardware-calibrated timing and TT-motor commands.

The model dimensions did not change:

- observation: 8 distance sensors, normalized forward speed, normalized turn
  rate;
- actions: 3;
- hidden layers: 64 and 64;
- trainable parameters: 5,059.

## Validated curriculum results

| Stage | Target | Training episodes | Final greedy result | Mean reward | Progress |
|---|---:|---:|---:|---:|---:|
| 1 | checkpoint 1 | 200 | 50/50 | 130.90 | 10% |
| 2 | checkpoint 2 | 300 | 50/50 | 170.66 | 20% |
| 3 | checkpoint 3 | 500 | 50/50 | 208.70 | 30% |
| 4 | checkpoint 4 | 300 | 50/50 | 255.15 | 40% |
| 5 | checkpoint 6 | 100 | 50/50 | 341.93 | 60% |
| 6 | checkpoint 8 | 100 | 50/50 | 424.65 | 80% |
| 7 | checkpoint 10 | 200 | 50/50 | 485.89 | 100% |
| 8 | actual goal | 200 | 50/50 | 529.56 | 100% |

Stage 7 scored 0/50 at episode 100 and passed 50/50 at episode 200. The final
goal stage scored 25/50 at episode 100 and passed 50/50 at episode 200. These
results show why periodic policy-only gates are necessary: successful expert
demonstrations alone do not prove that the DQN learned the behavior.

## Final artifacts

- Final policy: `models/curriculum_v4_macro_final/stage_08_goal/dqn_latest.pt`
- Final summary: `runs/curriculum_v4_macro_final/curriculum_summary.json`
- Clean stage 6-8 training log: `runs/curriculum_v4_macro_final/training.csv`
- Preserved failed stage-5 experiment:
  `runs/curriculum_v3_attempt38_stage5_extra600_all_gates_failed`

The default program mode is now `evaluate`, pointed at the final goal policy,
so opening the world tests the completed policy instead of accidentally
starting another training run.

## Scope of the result

The result proves that the policy can complete the current Webots map from its
configured start position. It does not yet prove generalization to arbitrary
tracks or the real RC car. The next training phase should introduce multiple
maps, randomized safe start poses, sensor noise, motor variation, and calibrated
real-world maneuver timing before TFLite/ESP32 deployment.
