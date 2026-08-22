# Single-world survival maps

## Outcome

`car_env_test_v2.wbt` now supports six selectors without duplicating the world
file:

| Selector | Purpose | Goal | Spawn |
|---|---|---:|---|
| `survival_mix` | Three-map per-episode training/evaluation pool | Disabled | Random safe pose and heading |
| `race_track` | Original ten-checkpoint goal route | Enabled | Original fixed pose |
| `obstacle_field` | Irregular obstacles plus blocked edges/corners | Disabled | Random safe pose and heading |
| `tight_corridors` | Alternating narrow roads and end gates | Disabled | Random safe pose and heading |
| `dense_pinch_points` | Dense staggered blocks and short sight lines | Disabled | Random safe pose and heading |
| `chessboard` | Repeated grid obstacles and connected corridors | Disabled | Random safe pose and heading |

All generated survival layouts use the existing 10 x 10 metre
`RectangleArena`. The
controller hides the original race walls and creates the selected layout as
locked Webots `Solid` nodes with matching `boundingObject` boxes. This keeps the
robot, sensors, floor, controller, and world file shared across every map.

The obstacle field includes four diagonal corner gates and eight staggered
wall-connected barriers. They interrupt the previously clear perimeter route,
so circling beside the arena wall is no longer a low-risk survival strategy.

`survival_mix` uses `obstacle_field`, `tight_corridors`, and
`dense_pinch_points`. Training randomly chooses a different map after every
episode. Evaluation rotates through them in a fixed order for an equal sample
on each layout. `chessboard` remains outside the pool as a held-out test.

## Select a map

Edit this value in `config.py`:

```python
"map_selector": "survival_mix",
```

Valid values are `survival_mix`, `race_track`, `obstacle_field`,
`tight_corridors`, `dense_pinch_points`, and `chessboard`.

The training dashboard also exposes this as **World layout** under the
Environment group. A saved dashboard change takes effect the next time Webots
starts. The observer HUD displays the active map in its agent-state heading.

For a one-run override in PowerShell, use:

```powershell
$env:RL_MAP_SELECTOR = "chessboard"
& "C:\Program Files\Webots\msys64\mingw64\bin\webots.exe" --batch --mode=fast --no-rendering "C:\Users\titan\OneDrive\Documents\Bot_sim\worlds\car_env_test_v2.wbt"
```

Remove the override later with:

```powershell
Remove-Item Env:RL_MAP_SELECTOR
```

## Random spawning

Each survival reset samples:

- X and Y position across the arena;
- a heading from -180 to +180 degrees;
- enough clearance for the full 25.5 x 14.5 cm vehicle footprint;
- additional map-specific safety clearance from every obstacle and outer wall.

The check uses a conservative circle around the rectangular vehicle, so none of
its corners can begin inside a bounding object. It retries up to the configured
`spawn_attempts` and raises a clear error instead of accepting an unsafe pose.

## Survival episode semantics

These maps intentionally have no finish-line goal and no race-track waypoint
reward. An episode ends as:

- `collision` when a proximity sensor crosses the collision threshold;
- `survival_complete` when the car lasts for `environment.max_steps`;
- `simulation_stopped` when Webots stops.

`survival_complete` counts as success and receives the configurable
`environment.reward.survival_complete` reward. Survival maps remove the track's
elapsed-time penalty, because otherwise a late collision can cost more than an
early crash. Unsafe proximity, stopping, unnecessary steering, and collision
penalties remain active.

The track-specific 90-degree turn macro is disabled automatically on survival
maps. The DQN therefore makes frequent forward/left/right decisions suitable
for local obstacle avoidance.

## Training warning

The existing checkpoint was trained for the original race track. Selecting a
survival layout changes the task; it does not make that checkpoint an obstacle
avoidance policy. Before starting a survival run:

1. Select `survival_mix`.
2. Set `program.mode` to `train`.
3. Disable `training.curriculum.enabled`, because the current curriculum uses
   race-track checkpoints.
4. Set `training.resume` to `False` for a new policy.
5. Use new `logging.directory` and `training.save_directory` values so the
   validated race checkpoint and CSV are not mixed with survival artifacts.

A good curriculum is to learn on `survival_mix` and reserve `chessboard` for
validation that was not seen during training.

The completed 2,000-episode frozen curriculum-v4 benchmark is preserved in
`runs/obstacle_v4_frozen_eval_2000`. The balanced pre-training benchmark writes
300 episodes—100 per training map—to
`runs/survival_mix_v4_baseline_eval_300`. Every CSV row identifies its active
map, and `evaluation_summary.json` reports both overall and per-map performance.
The observer HUD is disabled for long headless evaluations; the web dashboard
can follow the CSV without slowing Webots.

## Validation evidence

The map-pool implementation was checked at four levels:

- 10,000 generated spawn poses were accepted safely on each training map;
- both new tight layouts passed headless Webots steering diagnostics;
- a real three-episode Webots rotation smoke test selected all three maps and
  survived the full 10,000 steps on each;
- the complete automated suite passes 78 tests, including map generation,
  spawn safety, rotation, termination, CSV logging, summaries, and dashboard
  configuration.

The frozen race-track curriculum-v4 checkpoint was also evaluated for 2,000
episodes on the first obstacle layout. It survived 782 episodes (39.1%),
collided in 1,218 (60.9%), and averaged 6,062.92 steps. This is a baseline, not
the final obstacle policy: it shows useful transfer from track driving while
also confirming that dedicated survival training is necessary.

The balanced 300-episode benchmark uses the same frozen checkpoint and assigns
100 episodes to each map in `survival_mix`. Its live CSV and final
`evaluation_summary.json` are written under
`runs/survival_mix_v4_baseline_eval_300`. Generated experiment artifacts remain
outside Git by design.

## Adding another layout

Add another entry under `environment.maps` in `config.py` and include its name
in the dashboard selector. Each obstacle is described by `center`, `size`, and
`angle_rad`; optional `height_m` and RGB `color` fields customize its appearance.
No additional `.wbt` file is required.
