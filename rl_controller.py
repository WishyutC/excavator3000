from environment import RCEnvironment


def main():
    env = RCEnvironment()

    episode = 0

    try:
        while True:
            episode += 1

            state = env.reset()

            total_reward = 0.0
            done = False
            step = 0

            print(f"\n===== EPISODE {episode} =====")

            while not done:

                # Drive forward continuously to verify the Webots setup.
                action = 0

                next_state, reward, done, info = env.step(action)

                if next_state is None:
                    print("Webots simulation stopped.")
                    return

                total_reward += reward
                step += 1

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
