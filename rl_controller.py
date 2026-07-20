from environment import RCEnvironment
from config import CONFIG


def main():
    env = RCEnvironment()

    episode = 0

    try:
        while True:
            episode += 1

            state = env.reset()

            total_reward = -1
            done = False
            step = 0

            if CONFIG["program"]["terminal_output"]:
                print(f"\n===== EPISODE {episode} =====")

            while not done:

                # Fixed action used while verifying the Webots setup.
                action = CONFIG["program"]["test_action"]

                next_state, reward, done, info = env.step(action)

                if next_state is None:
                    print("Webots simulation stopped.")
                    return

                total_reward += reward
                step += 1

                env.update_hud(
                    episode,
                    step,
                    action,
                    reward,
                    total_reward,
                    next_state
                )

                if CONFIG["program"]["terminal_output"]:
                    print(
                        f"Episode: {episode} | "
                        f"Step: {step} | "
                        f"Action: {action} | "
                        f"Reward: {reward:.3f} | "
                        f"Total Reward: {total_reward:.3f}"
                    )

                    print(
                        "Sensors:",
                        [
                            round(value, 1)
                            for value in next_state
                        ]
                    )

                state = next_state

            if CONFIG["program"]["terminal_output"]:
                print(
                    f"Episode {episode} finished | "
                    f"Steps: {step} | "
                    f"Total Reward: {total_reward:.3f}"
                )

    except KeyboardInterrupt:
        print("\nEnvironment test stopped by user.")

    finally:
        env.close()


if __name__ == "__main__":
    main()
