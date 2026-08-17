"""Webots entry point for test, DQN training, and evaluation modes."""

import json
from pathlib import Path

from config import CONFIG
from environment import RCEnvironment


def run_test(environment):
    episode = 0

    while True:
        episode += 1
        state = environment.reset()
        total_reward = 0.0
        done = False
        step = 0
        info = {"termination_reason": "running", "is_success": False}

        if CONFIG["program"]["terminal_output"]:
            print(f"\n===== TEST EPISODE {episode} =====")

        while not done:
            action = CONFIG["program"]["test_action"]
            next_state, reward, done, info = environment.step(action)

            if next_state is None:
                print("Webots simulation stopped.")
                return

            total_reward += reward
            step += 1
            environment.update_hud(
                episode,
                step,
                action,
                reward,
                total_reward,
                next_state,
                info
            )

            if CONFIG["program"]["terminal_output"]:
                print(
                    f"Episode: {episode} | Step: {step} | Action: {action} | "
                    f"Reward: {reward:.3f} | Total: {total_reward:.3f}"
                )
                print("Observation:", [round(value, 3) for value in next_state])

            state = next_state

        if CONFIG["program"]["terminal_output"]:
            print(
                f"Episode {episode} finished | Steps: {step} | "
                f"Total Reward: {total_reward:.3f} | "
                f"Reason: {info['termination_reason']}"
            )


def create_agent(environment):
    from dqn_agent import DQNAgent

    agent = DQNAgent(environment.state_size, environment.action_size)
    print(
        f"DQN ready | device {agent.device} | "
        f"parameters {agent.online_network.parameter_count:,}"
    )
    return agent


def run_turn_diagnostic(environment):
    """Verify both physical turn actions in the real Webots controller."""

    turn_steps = int(CONFIG["diagnostics"]["turn_steps"])
    minimum_rate = float(CONFIG["diagnostics"]["minimum_turn_rate_rad_s"])
    results = {}

    for action, label in ((1, "left"), (2, "right")):
        environment.robot.reset()
        rates = []
        environment.robot.apply_action(action)
        for _ in range(turn_steps):
            if environment.robot.step() == -1:
                break
            rates.append(environment.robot.get_turn_rate())
        environment.robot.stop()
        mean_rate = sum(rates) / len(rates) if rates else 0.0
        results[label] = {
            "steps": len(rates),
            "mean_turn_rate_rad_s": mean_rate,
            "passed": abs(mean_rate) >= minimum_rate
        }

    passed = all(item["passed"] for item in results.values())
    if passed and results["left"]["mean_turn_rate_rad_s"] * results["right"]["mean_turn_rate_rad_s"] >= 0:
        passed = False
        results["direction_check"] = "left and right did not rotate oppositely"

    print("TURN_DIAGNOSTIC_JSON=" + json.dumps({"passed": passed, **results}))
    if not passed:
        raise RuntimeError("Turn diagnostic failed. Review wheel action ratios.")


def run_training(environment):
    from dqn_trainer import DQNTrainer

    training_config = CONFIG["training"]
    agent = create_agent(environment)
    start_episode = 0

    loaded = None
    if training_config["resume"]:
        checkpoint = (
            Path(training_config["save_directory"])
            / training_config["checkpoint_name"]
        )
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"Training resume checkpoint does not exist: {checkpoint}"
            )
        loaded = agent.load_checkpoint(checkpoint)
        start_episode = int(loaded.get("episode", 0))
        print(f"Resumed training from episode {start_episode}: {checkpoint}")

    trainer = DQNTrainer(environment, agent)
    if loaded is not None:
        extra = loaded.get("extra", {})
        trainer.best_success_reward = float(
            extra.get("best_success_reward", float("-inf"))
        )
        trainer.best_candidate_reward = float(
            extra.get("best_candidate_reward", float("-inf"))
        )
    trainer.train(
        episodes=training_config["episodes"],
        start_episode=start_episode
    )


def run_evaluation(environment):
    from dqn_trainer import DQNTrainer

    evaluation_config = CONFIG["evaluation"]
    checkpoint = Path(evaluation_config["checkpoint"])
    if not checkpoint.exists():
        raise FileNotFoundError(f"Evaluation checkpoint not found: {checkpoint}")

    agent = create_agent(environment)
    agent.load_checkpoint(checkpoint, load_optimizer=False)
    trainer = DQNTrainer(environment, agent)
    summaries = trainer.evaluate(evaluation_config["episodes"])

    if summaries:
        successes = sum(summary.success for summary in summaries)
        average_reward = sum(
            summary.total_reward for summary in summaries
        ) / len(summaries)
        print(
            f"Evaluation complete | success {successes}/{len(summaries)} | "
            f"average reward {average_reward:+.3f}"
        )


def main():
    mode = CONFIG["program"]["mode"].lower()
    runners = {
        "test": run_test,
        "diagnostic": run_turn_diagnostic,
        "train": run_training,
        "evaluate": run_evaluation
    }
    if mode not in runners:
        raise ValueError(
            f'Unsupported program mode "{mode}". '
            "Use test, diagnostic, train, or evaluate."
        )

    environment = RCEnvironment()
    try:
        runners[mode](environment)
    except KeyboardInterrupt:
        print(f"\n{mode.capitalize()} mode stopped by user.")
    finally:
        environment.close()


if __name__ == "__main__":
    main()
