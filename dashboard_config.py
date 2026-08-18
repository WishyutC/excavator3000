"""Safe, formatting-preserving access to selected CONFIG values."""

import ast
from pathlib import Path
import tempfile
import threading


CONFIG_LOCK = threading.Lock()


FIELDS = {
    "program.mode": {
        "label": "Program mode", "type": "select",
        "choices": ["train", "evaluate", "test"], "group": "Run"
    },
    "environment.max_steps": {
        "label": "Maximum episode steps", "type": "int",
        "minimum": 100, "maximum": 50_000, "group": "Environment"
    },
    "environment.collision_threshold": {
        "label": "Collision sensor threshold", "type": "int",
        "minimum": 1, "maximum": 10_000, "group": "Environment"
    },
    "robot.drive.speed_scale": {
        "label": "Motor speed scale", "type": "float",
        "minimum": 0.05, "maximum": 1.0, "step": 0.01,
        "group": "Environment"
    },
    "environment.reward.collision": {
        "label": "Collision reward", "type": "float",
        "minimum": -1_000.0, "maximum": 0.0, "step": 1.0,
        "group": "Rewards"
    },
    "environment.reward.timeout": {
        "label": "Timeout reward", "type": "float",
        "minimum": -1_000.0, "maximum": 0.0, "step": 1.0,
        "group": "Rewards"
    },
    "environment.reward.stuck_terminal": {
        "label": "Stuck reward", "type": "float",
        "minimum": -1_000.0, "maximum": 0.0, "step": 1.0,
        "group": "Rewards"
    },
    "environment.reward.goal_base": {
        "label": "Goal base reward", "type": "float",
        "minimum": 0.0, "maximum": 1_000.0, "step": 1.0,
        "group": "Rewards"
    },
    "environment.reward.goal_time_bonus": {
        "label": "Goal time bonus", "type": "float",
        "minimum": 0.0, "maximum": 1_000.0, "step": 1.0,
        "group": "Rewards"
    },
    "environment.reward.curriculum_goal_base": {
        "label": "Curriculum target reward", "type": "float",
        "minimum": 0.0, "maximum": 1_000.0, "step": 1.0,
        "group": "Rewards"
    },
    "environment.reward.curriculum_goal_time_bonus": {
        "label": "Curriculum time bonus", "type": "float",
        "minimum": 0.0, "maximum": 1_000.0, "step": 1.0,
        "group": "Rewards"
    },
    "environment.reward.safe_motion_scale": {
        "label": "Safe-motion scale", "type": "float",
        "minimum": 0.0, "maximum": 2.0, "step": 0.001,
        "group": "Rewards"
    },
    "environment.reward.danger_penalty_scale": {
        "label": "Danger penalty scale", "type": "float",
        "minimum": 0.0, "maximum": 5.0, "step": 0.01,
        "group": "Rewards"
    },
    "environment.reward.clear_space_steering_penalty_scale": {
        "label": "Clear-space steering penalty", "type": "float",
        "minimum": 0.0, "maximum": 2.0, "step": 0.001,
        "group": "Rewards"
    },
    "environment.reward.time_penalty_start": {
        "label": "Initial time penalty", "type": "float",
        "minimum": 0.0, "maximum": 1.0, "step": 0.001,
        "group": "Rewards"
    },
    "environment.reward.time_penalty_growth": {
        "label": "Time-penalty growth", "type": "float",
        "minimum": 0.0, "maximum": 1.0, "step": 0.001,
        "group": "Rewards"
    },
    "environment.progress.checkpoint_reward": {
        "label": "Checkpoint reward", "type": "float",
        "minimum": 0.0, "maximum": 100.0, "step": 0.5,
        "group": "Progress"
    },
    "environment.progress.distance_reward_scale": {
        "label": "Distance progress scale", "type": "float",
        "minimum": 0.0, "maximum": 20.0, "step": 0.1,
        "group": "Progress"
    },
    "environment.stuck_detection.no_progress_steps": {
        "label": "Stuck termination steps", "type": "int",
        "minimum": 10, "maximum": 50_000, "group": "Progress"
    },
    "training.episodes": {
        "label": "Target episode", "type": "int",
        "minimum": 1, "maximum": 10_000_000, "group": "DQN"
    },
    "training.gamma": {
        "label": "Discount factor (gamma)", "type": "float",
        "minimum": 0.0, "maximum": 0.9999, "step": 0.001,
        "group": "DQN"
    },
    "training.learning_rate": {
        "label": "Learning rate", "type": "float",
        "minimum": 0.000001, "maximum": 0.1, "step": 0.0001,
        "group": "DQN"
    },
    "training.action_repeat": {
        "label": "Action repeat", "type": "int",
        "minimum": 1, "maximum": 20, "group": "DQN"
    },
    "training.double_dqn": {
        "label": "Double DQN", "type": "bool", "group": "DQN"
    },
    "training.reward_scale": {
        "label": "Learning reward scale", "type": "float",
        "minimum": 0.0001, "maximum": 1.0, "step": 0.001,
        "group": "DQN"
    },
    "training.epsilon.start": {
        "label": "Starting epsilon", "type": "float",
        "minimum": 0.0, "maximum": 1.0, "step": 0.01, "group": "DQN"
    },
    "training.epsilon.end": {
        "label": "Final epsilon", "type": "float",
        "minimum": 0.0, "maximum": 1.0, "step": 0.01, "group": "DQN"
    },
    "training.epsilon.decay_steps": {
        "label": "Epsilon decay steps", "type": "int",
        "minimum": 1, "maximum": 100_000_000, "group": "DQN"
    },
    "training.target_update_steps": {
        "label": "Target update interval", "type": "int",
        "minimum": 1, "maximum": 10_000_000, "group": "DQN"
    },
    "training.save_every_episodes": {
        "label": "Checkpoint interval", "type": "int",
        "minimum": 1, "maximum": 100_000, "group": "DQN"
    },
    "training.resume": {
        "label": "Resume latest checkpoint", "type": "bool", "group": "DQN"
    },
    "training.curriculum.enabled": {
        "label": "Staged curriculum", "type": "bool", "group": "Curriculum"
    },
    "training.curriculum.training_success_rate": {
        "label": "Training success threshold", "type": "float",
        "minimum": 0.0, "maximum": 1.0, "step": 0.01,
        "group": "Curriculum"
    },
    "training.curriculum.evaluation_episodes": {
        "label": "Evaluation episodes", "type": "int",
        "minimum": 1, "maximum": 10_000, "group": "Curriculum"
    },
    "training.curriculum.evaluation_success_rate": {
        "label": "Evaluation success threshold", "type": "float",
        "minimum": 0.0, "maximum": 1.0, "step": 0.01,
        "group": "Curriculum"
    },
    "training.replay_buffer.capacity": {
        "label": "Replay capacity", "type": "int",
        "minimum": 64, "maximum": 10_000_000, "group": "Replay"
    },
    "training.replay_buffer.batch_size": {
        "label": "Replay batch size", "type": "int",
        "minimum": 1, "maximum": 65_536, "group": "Replay"
    },
    "training.replay_buffer.learning_starts": {
        "label": "Replay warmup steps", "type": "int",
        "minimum": 1, "maximum": 10_000_000, "group": "Replay"
    }
}


def _config_node(tree):
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == "CONFIG"
               for target in node.targets):
            return node.value
    raise ValueError("CONFIG assignment was not found.")


def _child_node(dictionary_node, key):
    if not isinstance(dictionary_node, ast.Dict):
        raise ValueError(f"Cannot access {key!r} inside a non-dictionary value.")
    for key_node, value_node in zip(dictionary_node.keys, dictionary_node.values):
        if isinstance(key_node, ast.Constant) and key_node.value == key:
            return value_node
    raise KeyError(key)


def _path_node(root, dotted_path):
    node = root
    for part in dotted_path.split("."):
        node = _child_node(node, part)
    return node


def read_config(config_path):
    source = Path(config_path).read_text(encoding="utf-8")
    root = _config_node(ast.parse(source))
    return ast.literal_eval(root)


def value_at(config, dotted_path):
    value = config
    for part in dotted_path.split("."):
        value = value[part]
    return value


def _coerce(path, raw_value):
    field = FIELDS[path]
    field_type = field["type"]
    if field_type == "bool":
        if not isinstance(raw_value, bool):
            raise ValueError(f"{field['label']} must be true or false.")
        value = raw_value
    elif field_type == "int":
        if isinstance(raw_value, bool):
            raise ValueError(f"{field['label']} must be an integer.")
        value = int(raw_value)
    elif field_type == "float":
        if isinstance(raw_value, bool):
            raise ValueError(f"{field['label']} must be a number.")
        value = float(raw_value)
    elif field_type == "select":
        value = str(raw_value)
        if value not in field["choices"]:
            raise ValueError(f"Invalid value for {field['label']}.")
    else:
        raise ValueError(f"Unsupported field type: {field_type}")

    if "minimum" in field and value < field["minimum"]:
        raise ValueError(f"{field['label']} must be at least {field['minimum']}.")
    if "maximum" in field and value > field["maximum"]:
        raise ValueError(f"{field['label']} must be at most {field['maximum']}.")
    return value


def _validate_relationships(config):
    epsilon = config["training"]["epsilon"]
    if epsilon["end"] > epsilon["start"]:
        raise ValueError("Final epsilon cannot be greater than starting epsilon.")

    replay = config["training"]["replay_buffer"]
    if replay["batch_size"] > replay["learning_starts"]:
        raise ValueError("Replay warmup must be at least the batch size.")
    if replay["learning_starts"] > replay["capacity"]:
        raise ValueError("Replay warmup cannot exceed replay capacity.")


def update_config(config_path, requested_values):
    unknown = set(requested_values) - set(FIELDS)
    if unknown:
        raise ValueError(f"Unsupported configuration field: {sorted(unknown)[0]}")

    updates = {
        path: _coerce(path, value)
        for path, value in requested_values.items()
    }

    path = Path(config_path)
    with CONFIG_LOCK:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        root = _config_node(tree)
        current = ast.literal_eval(root)

        for dotted_path, value in updates.items():
            target = current
            parts = dotted_path.split(".")
            for part in parts[:-1]:
                target = target[part]
            target[parts[-1]] = value
        _validate_relationships(current)

        lines = source.splitlines(keepends=True)
        offsets = []
        position = 0
        for line in lines:
            offsets.append(position)
            position += len(line)

        replacements = []
        for dotted_path, value in updates.items():
            node = _path_node(root, dotted_path)
            start = offsets[node.lineno - 1] + node.col_offset
            end = offsets[node.end_lineno - 1] + node.end_col_offset
            replacements.append((start, end, repr(value)))

        for start, end, replacement in sorted(replacements, reverse=True):
            source = source[:start] + replacement + source[end:]

        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", newline="", delete=False, dir=path.parent
        ) as temporary:
            temporary.write(source)
            temporary_path = Path(temporary.name)
        temporary_path.replace(path)

    return read_config(path)


def public_fields(config_path):
    config = read_config(config_path)
    result = []
    for path, metadata in FIELDS.items():
        try:
            value = value_at(config, path)
        except KeyError:
            # Keep the dashboard compatible with older config files that do
            # not yet contain newly introduced optional settings.
            continue
        result.append({"path": path, "value": value, **metadata})
    return result
