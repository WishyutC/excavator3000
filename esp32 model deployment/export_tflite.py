"""Export the curriculum-v4 race DQN into a standard float32 TFLite graph."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
DEPLOYMENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from dqn_network import DQNNetwork


DEFAULT_SOURCE = (
    ROOT / "models" / "curriculum_v4_macro_final"
    / "stage_08_goal" / "dqn_latest.pt"
)
DEFAULT_TFLITE = DEPLOYMENT_DIR / "excavator3000_race_v4_float32.tflite"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def locate_android_jars(android_studio):
    root = Path(android_studio)
    library = root / "plugins" / "android" / "lib"
    schema = library / "tensorflow-lite-metadata-0.1.0-rc2.jar"
    flatbuffers = library / "flatbuffers-java-1.12.0.jar"
    javac = root / "jbr" / "bin" / "javac.exe"
    java = root / "jbr" / "bin" / "java.exe"
    required = (schema, flatbuffers, javac, java)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing Android Studio tools: " + ", ".join(missing))
    return schema, flatbuffers, javac, java


def write_float_arrays(path, arrays):
    with Path(path).open("wb") as file:
        for array in arrays:
            values = np.asarray(array, dtype="<f4").reshape(-1)
            file.write(values.tobytes(order="C"))


def build_reference_vectors(model, path, count=256):
    generator = np.random.default_rng(3000)
    inputs = np.empty((count, 10), dtype=np.float32)
    inputs[:, :8] = generator.uniform(0.0, 1.0, (count, 8))
    inputs[:, 8:] = generator.uniform(-1.0, 1.0, (count, 2))
    inputs[0] = 0.0
    inputs[1, :8] = 1.0
    inputs[1, 8:] = 0.0
    inputs[2, :8] = 0.5
    inputs[2, 8:] = (-1.0, 1.0)

    with torch.no_grad():
        outputs = model(torch.from_numpy(inputs)).cpu().numpy()
    combined = np.concatenate((inputs, outputs), axis=1)
    np.savetxt(path, combined, delimiter=",", fmt="%.9g")
    return count


def parse_verification(output):
    line = next(
        (item for item in output.splitlines() if item.startswith("VERIFY ")),
        None
    )
    if line is None:
        raise RuntimeError("Java exporter did not return verification metrics.")
    values = {}
    for field in line.removeprefix("VERIFY ").split():
        key, value = field.split("=", 1)
        values[key] = float(value) if key != "cases" else int(value)
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_TFLITE)
    parser.add_argument(
        "--android-studio",
        default=r"C:\Program Files\Android\Android Studio"
    )
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {source}")

    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    architecture = {
        "state_size": int(checkpoint["state_size"]),
        "action_size": int(checkpoint["action_size"]),
        "hidden_sizes": [int(value) for value in checkpoint["hidden_sizes"]]
    }
    if architecture != {
        "state_size": 10,
        "action_size": 3,
        "hidden_sizes": [64, 64]
    }:
        raise ValueError(f"Unexpected checkpoint architecture: {architecture}")

    model = DQNNetwork(10, 3, (64, 64))
    model.load_state_dict(checkpoint["online_network"])
    model.eval()
    state = model.state_dict()
    arrays = [
        state["model.0.weight"].numpy(),
        state["model.0.bias"].numpy(),
        state["model.2.weight"].numpy(),
        state["model.2.bias"].numpy(),
        state["model.4.weight"].numpy(),
        state["model.4.bias"].numpy()
    ]

    schema, flatbuffers, javac, java = locate_android_jars(args.android_studio)
    classpath = f"{schema};{flatbuffers}"
    output.parent.mkdir(parents=True, exist_ok=True)
    copied_checkpoint = output.parent / "dqn_latest.pt"
    if source != copied_checkpoint.resolve():
        shutil.copy2(source, copied_checkpoint)

    with tempfile.TemporaryDirectory(prefix="excavator3000_tflite_") as temp:
        temp = Path(temp)
        weights = temp / "weights.bin"
        vectors = temp / "verification_vectors.csv"
        classes = temp / "classes"
        classes.mkdir()
        write_float_arrays(weights, arrays)
        case_count = build_reference_vectors(model, vectors)

        source_java = output.parent / "TfliteDenseExporter.java"
        subprocess.run(
            [str(javac), "-cp", classpath, "-d", str(classes), str(source_java)],
            check=True
        )
        completed = subprocess.run(
            [
                str(java), "-cp", f"{classes};{classpath}",
                "TfliteDenseExporter", str(weights), str(output), str(vectors)
            ],
            check=True,
            capture_output=True,
            text=True
        )
        metrics = parse_verification(completed.stdout)

    if metrics["cases"] != case_count:
        raise RuntimeError("Verification did not evaluate every reference case.")
    if metrics["max_abs_error"] > 1e-4:
        raise RuntimeError(f"TFLite graph error is too large: {metrics}")
    if metrics["action_agreement"] != 1.0:
        raise RuntimeError(f"Action selection mismatch: {metrics}")

    report = {
        "model": "Excavator3000 curriculum-v4 race policy",
        "format": "TensorFlow Lite FlatBuffer",
        "precision": "float32",
        "source_checkpoint": copied_checkpoint.name,
        "source_checkpoint_sha256": sha256(copied_checkpoint),
        "source_checkpoint_bytes": copied_checkpoint.stat().st_size,
        "tflite_file": output.name,
        "tflite_sha256": sha256(output),
        "tflite_bytes": output.stat().st_size,
        "checkpoint_episode": int(checkpoint.get("episode", 0)),
        "environment_steps": int(checkpoint.get("environment_steps", 0)),
        "training_steps": int(checkpoint.get("training_steps", 0)),
        "architecture": architecture,
        "parameter_count": model.parameter_count,
        "input": {
            "name": "observation",
            "shape": [1, 10],
            "dtype": "float32"
        },
        "output": {
            "name": "q_values",
            "shape": [1, 3],
            "dtype": "float32",
            "order": ["forward", "turn_left", "turn_right"]
        },
        "verification": {
            **metrics,
            "method": (
                "Parsed TFLite FlatBuffer weights and operators were executed "
                "against deterministic PyTorch reference vectors."
            ),
            "desktop_litert_runtime_executed": False
        }
    }
    report_path = output.parent / "verification_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
