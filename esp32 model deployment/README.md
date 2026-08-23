# Excavator3000 ESP32 model deployment package

This directory contains a deployment copy and float32 TensorFlow Lite export of
the requested curriculum-v4 race checkpoint:

```text
source: models/curriculum_v4_macro_final/stage_08_goal/dqn_latest.pt
```

## Files

| File | Purpose |
|---|---|
| `dqn_latest.pt` | Untouched deployment copy of the requested PyTorch checkpoint |
| `excavator3000_race_v4_float32.tflite` | Standard float32 TFLite inference graph |
| `verification_report.json` | Architecture, hashes, sizes, and numerical checks |
| `export_tflite.py` | Reproducible PyTorch checkpoint export driver |
| `TfliteDenseExporter.java` | Local FlatBuffer graph builder and structural verifier |

The `.pt` and `.tflite` binaries are ignored by Git. Copy the entire directory
directly when sharing it with the hardware teammate.

## Important limitation

This is the **race-track latest checkpoint**, exactly as requested. It is not
the newer obstacle-avoidance policy currently being trained, and `dqn_latest.pt`
is not necessarily stronger than the race experiment's `dqn_best.pt`.

Use this package for integration testing of sensors, TensorFlow Lite Micro,
action selection, and motor mapping. Do not treat it as the final autonomous
obstacle-avoidance release.

## Model contract

- Input type: float32
- Input shape: `[1, 10]`
- Architecture: `10 -> 64 ReLU -> 64 ReLU -> 3 linear Q-values`
- Output type: float32
- Output shape: `[1, 3]`
- Output order: forward, turn left, turn right
- Selection: choose the index with the largest Q-value

Input order:

```text
front, back, left, right,
left-front, right-front, left-back, right-back,
normalized signed forward speed, normalized signed turn rate
```

See `docs/MODEL_ARCHITECTURE.md` for normalization formulas and the full ESP32
inference contract.

## Verification

The exporter checks the TFLite FlatBuffer identifier, model version, tensor
shapes, operator count, standard `FULLY_CONNECTED` operators, serialized
weights, output error, and action agreement against 256 deterministic PyTorch
reference inputs.

Because a Windows LiteRT interpreter was not installed in this workspace, the
verification executes the parsed TFLite operators rather than the official
desktop runtime. The hardware teammate should still run a TFLite Micro smoke
test on the target board before connecting motors.

## Re-export

Android Studio supplies the local TensorFlow Lite schema and FlatBuffers Java
libraries used by the exporter. From the repository root:

```powershell
python ".\esp32 model deployment\export_tflite.py"
```

The command overwrites only the deployment copies, never the source checkpoint.
