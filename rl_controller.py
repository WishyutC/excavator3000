"""Webots entry point for test, DQN training, and evaluation modes."""

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


def run_training(environment):
    from dqn_trainer import DQNTrainer

    training_config = CONFIG["training"]
    agent = create_agent(environment)
    start_episode = 0

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
        "train": run_training,
        "evaluate": run_evaluation
    }
    if mode not in runners:
        raise ValueError(
            f'Unsupported program mode "{mode}". Use test, train, or evaluate.'
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
