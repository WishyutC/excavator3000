"""Sequential checkpoint curriculum with fresh replay at every stage."""

from collections import deque
from copy import deepcopy
import csv
import json
from pathlib import Path

from config import CONFIG
from dqn_agent import DQNAgent
from dqn_trainer import DQNTrainer
from training_logger import TrainingLogger


def _stage_training_config(base_config, stage, stage_index):
    config = deepcopy(base_config)
    config["seed"] = int(base_config["seed"]) + int(stage_index)
    config["episodes"] = int(stage["maximum_episodes"])
    config["epsilon"]["start"] = float(stage["epsilon_start"])
    config["epsilon"]["decay_steps"] = int(stage["epsilon_decay_steps"])
    config["forced_action"] = stage.get("forced_action")
    config["expert_policy"] = stage.get("expert_policy")
    guided = (
        config["forced_action"] is not None
        or config["expert_policy"] is not None
    )
    config["expert_imitation_weight"] = float(
        stage.get("expert_imitation_weight", 1.0 if guided else 0.0)
    )
    config["save_directory"] = str(
        Path(base_config["save_directory"]) / stage["name"]
    )
    config["resume"] = False
    return config


def _success_rate(summaries):
    if not summaries:
        return 0.0
    return sum(summary.success for summary in summaries) / len(summaries)


def _mean(summaries, attribute):
    if not summaries:
        return 0.0
    return sum(float(getattr(item, attribute)) for item in summaries) / len(summaries)


def _write_summary(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, allow_nan=False),
        encoding="utf-8"
    )


def _last_logged_episode(path):
    if not path.exists():
        return 0
    last = 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                last = max(last, int(row.get("episode", 0)))
            except (TypeError, ValueError):
                continue
    return last


def _evaluate_saved_candidates(agent, trainer, stage_config, episodes):
    """Reload deployment candidates and select the strongest saved policy."""

    directory = Path(stage_config["save_directory"])
    names = (
        stage_config["checkpoint_name"],
        stage_config["best_checkpoint_name"]
    )
    checks = []
    selected_path = None
    selected_summaries = []
    selected_score = (-1.0, float("-inf"))

    for name in dict.fromkeys(names):
        path = directory / name
        if not path.exists():
            continue
        agent.load_checkpoint(path, load_optimizer=False)
        summaries = trainer.evaluate(episodes, verbose=False)
        rate = _success_rate(summaries)
        mean_reward = _mean(summaries, "total_reward")
        checks.append({
            "checkpoint": str(path),
            "evaluation_episodes": len(summaries),
            "success_rate": rate,
            "mean_reward": mean_reward,
            "mean_progress": _mean(summaries, "track_progress")
        })
        score = (rate, mean_reward)
        if score > selected_score:
            selected_path = path
            selected_summaries = summaries
            selected_score = score

    if selected_path is None:
        raise RuntimeError("No saved curriculum checkpoint was found.")
    agent.load_checkpoint(selected_path, load_optimizer=False)
    return selected_path, selected_summaries, checks


def run_staged_curriculum(environment, training_config=None):
    """Train and greedily validate every configured curriculum stage."""

    base = training_config or CONFIG["training"]
    curriculum = base["curriculum"]
    stages = curriculum["stages"]
    if not stages:
        raise ValueError("The staged curriculum requires at least one stage.")

    check_interval = int(curriculum["check_interval_episodes"])
    window_size = int(curriculum["success_window_episodes"])
    training_threshold = float(curriculum["training_success_rate"])
    evaluation_episodes = int(curriculum["evaluation_episodes"])
    evaluation_threshold = float(curriculum["evaluation_success_rate"])
    if check_interval <= 0 or window_size <= 0 or evaluation_episodes <= 0:
        raise ValueError("Curriculum intervals and windows must be positive.")

    run_directory = Path(CONFIG["logging"]["directory"])
    logger = TrainingLogger(path=run_directory)
    summary_path = run_directory / "curriculum_summary.json"
    start_stage_index = int(curriculum.get("start_stage_index", 1))
    if not 1 <= start_stage_index <= len(stages):
        raise ValueError("curriculum.start_stage_index is outside the stage list.")

    if start_stage_index == 1:
        payload = {
            "completed": False,
            "completed_stages": 0,
            "total_stages": len(stages),
            "stages": []
        }
        global_episode_offset = 0
        previous_policy = None
    else:
        if not summary_path.exists():
            raise RuntimeError("Curriculum resume summary was not found.")
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        if int(payload.get("completed_stages", 0)) < start_stage_index - 1:
            raise RuntimeError("Curriculum resume stages are not validated.")
        # Discard a previous failed/incomplete result for the stage being
        # retried while preserving every earlier validated stage.
        payload["stages"] = [
            item for item in payload.get("stages", [])
            if int(item.get("index", 0)) < start_stage_index
        ]
        payload["completed"] = False
        payload.pop("failed_stage", None)
        payload.pop("final_checkpoint", None)
        global_episode_offset = _last_logged_episode(logger.path)
        configured_policy = curriculum.get("initial_policy_checkpoint")
        if not configured_policy:
            raise RuntimeError("Curriculum resume policy was not configured.")
        previous_policy = Path(configured_policy)
        if not previous_policy.exists():
            raise RuntimeError(
                f"Curriculum resume policy was not found: {previous_policy}"
            )

    for stage_index, stage in enumerate(
        stages[start_stage_index - 1:],
        start=start_stage_index
    ):
        stage_name = str(stage["name"])
        target = stage["target_checkpoint"]
        minimum_episodes = int(stage["minimum_episodes"])
        maximum_episodes = int(stage["maximum_episodes"])
        greedy_check_interval = int(
            stage.get("greedy_check_interval_episodes", 0)
        )
        if not 0 < minimum_episodes <= maximum_episodes:
            raise ValueError(
                f"Invalid episode limits for curriculum stage {stage_name}."
            )
        if greedy_check_interval < 0:
            raise ValueError("Greedy check interval cannot be negative.")

        environment.set_curriculum_stage(stage_name, target)
        stage_config = _stage_training_config(base, stage, stage_index)
        agent = DQNAgent(
            environment.state_size,
            environment.action_size,
            training_config=stage_config
        )
        if previous_policy is not None:
            agent.load_policy_weights(previous_policy)

        print(
            f"CURRICULUM START | {stage_name} | target {target} | "
            f"episodes {minimum_episodes}-{maximum_episodes} | "
            f"epsilon {stage_config['epsilon']['start']:.2f}->"
            f"{stage_config['epsilon']['end']:.2f}"
        )
        trainer = DQNTrainer(
            environment,
            agent,
            training_config=stage_config,
            logger=logger,
            episode_offset=global_episode_offset
        )
        recent = deque(maxlen=window_size)
        trained = 0
        stopped_early = False
        evaluation_summaries = []
        evaluation_rate = 0.0
        evaluation_at_episode = None
        greedy_checks = []

        while trained < maximum_episodes:
            target_episode = min(trained + check_interval, maximum_episodes)
            summaries = trainer.train(
                episodes=target_episode,
                start_episode=trained
            )
            if not summaries:
                raise RuntimeError(
                    f"Webots stopped during curriculum stage {stage_name}."
                )
            recent.extend(summaries)
            trained = target_episode
            rate = _success_rate(recent)
            print(
                f"CURRICULUM WINDOW | {stage_name} | episode {trained} | "
                f"success {rate:.1%} over {len(recent)}"
            )
            if (
                greedy_check_interval > 0
                and trained >= minimum_episodes
                and trained % greedy_check_interval == 0
            ):
                evaluation_summaries = trainer.evaluate(
                    evaluation_episodes,
                    verbose=False
                )
                evaluation_rate = _success_rate(evaluation_summaries)
                evaluation_at_episode = trained
                check = {
                    "episode": trained,
                    "evaluation_episodes": len(evaluation_summaries),
                    "success_rate": evaluation_rate,
                    "mean_reward": _mean(
                        evaluation_summaries,
                        "total_reward"
                    ),
                    "mean_progress": _mean(
                        evaluation_summaries,
                        "track_progress"
                    )
                }
                greedy_checks.append(check)
                print("CURRICULUM_GREEDY_CHECK_JSON=" + json.dumps(check))
                if (
                    len(evaluation_summaries) == evaluation_episodes
                    and evaluation_rate >= evaluation_threshold
                ):
                    stopped_early = trained < maximum_episodes
                    break
            elif (
                greedy_check_interval == 0
                and trained >= minimum_episodes
                and len(recent) >= window_size
                and rate >= training_threshold
            ):
                stopped_early = trained < maximum_episodes
                break

        # Validate and transfer the policy produced at the end of this stage.
        # A single fast success can occur during early random exploration, so
        # the per-episode reward "best" checkpoint is not a reliable stage
        # selection criterion.
        stage_checkpoint, evaluation_summaries, deployment_checks = (
            _evaluate_saved_candidates(
                agent,
                trainer,
                stage_config,
                evaluation_episodes
            )
        )
        evaluation_rate = _success_rate(evaluation_summaries)

        stage_result = {
            "index": stage_index,
            "name": stage_name,
            "target_checkpoint": target,
            "episodes_trained": trained,
            "stopped_early": stopped_early,
            "training_window_success_rate": _success_rate(recent),
            "evaluation_episodes": len(evaluation_summaries),
            "evaluation_success_rate": evaluation_rate,
            "evaluation_mean_reward": _mean(
                evaluation_summaries,
                "total_reward"
            ),
            "evaluation_mean_progress": _mean(
                evaluation_summaries,
                "track_progress"
            ),
            "greedy_checks": greedy_checks,
            "deployment_checks": deployment_checks,
            "stage_checkpoint": str(stage_checkpoint),
            "passed": (
                len(evaluation_summaries) == evaluation_episodes
                and evaluation_rate >= evaluation_threshold
            )
        }
        payload["stages"].append(stage_result)
        print("CURRICULUM_STAGE_JSON=" + json.dumps(stage_result))

        if not stage_result["passed"]:
            payload["failed_stage"] = stage_name
            _write_summary(summary_path, payload)
            print(
                f"CURRICULUM STOP | {stage_name} failed greedy evaluation "
                f"({evaluation_rate:.1%} < {evaluation_threshold:.1%})."
            )
            return payload

        payload["completed_stages"] = stage_index
        previous_policy = stage_checkpoint
        global_episode_offset += trained
        _write_summary(summary_path, payload)

    payload["completed"] = True
    payload["final_checkpoint"] = str(previous_policy)
    _write_summary(summary_path, payload)
    print("CURRICULUM_COMPLETE_JSON=" + json.dumps(payload))
    return payload
