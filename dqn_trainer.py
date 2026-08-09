"""Training and evaluation orchestration around the Webots environment."""

from dataclasses import dataclass
from pathlib import Path

from config import CONFIG
from training_logger import TrainingLogger


@dataclass(frozen=True)
class EpisodeSummary:
    episode: int
    steps: int
    total_reward: float
    termination_reason: str
    success: bool
    epsilon: float
    buffer_size: int
    training_steps: int
    loss: float | None

    def to_dict(self):
        return self.__dict__.copy()


class DQNTrainer:
    def __init__(self, environment, agent, training_config=None, logger=None):
        self.environment = environment
        self.agent = agent
        self.config = training_config or CONFIG["training"]
        self.logger = logger or TrainingLogger()
        self.best_reward = float("-inf")

    def train(self, episodes=None, start_episode=0):
        episode_count = int(episodes or self.config["episodes"])
        summaries = []

        for episode in range(int(start_episode) + 1, episode_count + 1):
            summary, simulation_stopped = self._run_episode(
                episode,
                training=True
            )
            if simulation_stopped:
                break

            summaries.append(summary)
            self.logger.log_episode(summary.to_dict())
            self._save_training_checkpoints(summary)
            self._print_summary(summary, "TRAIN")

        if summaries:
            self._save_checkpoint(
                self.config["checkpoint_name"],
                summaries[-1]
            )
        return summaries

    def evaluate(self, episodes=None):
        episode_count = int(episodes or CONFIG["evaluation"]["episodes"])
        summaries = []

        for episode in range(1, episode_count + 1):
            summary, simulation_stopped = self._run_episode(
                episode,
                training=False
            )
            if simulation_stopped:
                break
            summaries.append(summary)
            self._print_summary(summary, "EVAL")

        return summaries

    def _run_episode(self, episode, training):
        state = self.environment.reset()
        total_reward = 0.0
        step = 0
        done = False
        latest_metrics = None
        info = {
            "termination_reason": "running",
            "is_success": False
        }

        while not done:
            action = self.agent.select_action(state, evaluate=not training)
            next_state, reward, done, info = self.environment.step(action)

            if next_state is None:
                return self._summary(
                    episode, step, total_reward, info, latest_metrics
                ), True

            if training:
                self.agent.remember(
                    state,
                    action,
                    reward,
                    next_state,
                    done,
                    info["termination_reason"]
                )
                update = self.agent.learn()
                if update is not None:
                    latest_metrics = update

            step += 1
            total_reward += reward
            info["training"] = self._training_info(latest_metrics, training)
            self.environment.update_hud(
                episode,
                step,
                action,
                reward,
                total_reward,
                next_state,
                info
            )
            state = next_state

        return self._summary(
            episode, step, total_reward, info, latest_metrics
        ), False

    def _summary(self, episode, steps, total_reward, info, metrics):
        return EpisodeSummary(
            episode=episode,
            steps=steps,
            total_reward=total_reward,
            termination_reason=info["termination_reason"],
            success=bool(info.get("is_success", False)),
            epsilon=float(self.agent.epsilon),
            buffer_size=len(self.agent.replay_buffer),
            training_steps=self.agent.training_steps,
            loss=metrics.loss if metrics is not None else None
        )

    def _training_info(self, metrics, training):
        info = {
            "mode": "train" if training else "evaluate",
            "epsilon": 0.0 if not training else self.agent.epsilon,
            "buffer_size": len(self.agent.replay_buffer),
            "training_steps": self.agent.training_steps,
            "device": str(self.agent.device)
        }
        if metrics is not None:
            info.update(metrics.to_info())
        return info

    def _save_training_checkpoints(self, summary):
        if summary.total_reward > self.best_reward:
            self.best_reward = summary.total_reward
            self._save_checkpoint(self.config["best_checkpoint_name"], summary)

        interval = int(self.config["save_every_episodes"])
        if interval > 0 and summary.episode % interval == 0:
            self._save_checkpoint(self.config["checkpoint_name"], summary)

    def _save_checkpoint(self, filename, summary):
        path = Path(self.config["save_directory"]) / filename
        self.agent.save_checkpoint(
            path,
            episode=summary.episode,
            extra={"episode_summary": summary.to_dict()}
        )

    @staticmethod
    def _print_summary(summary, label):
        loss = "warming up" if summary.loss is None else f"{summary.loss:.5f}"
        print(
            f"[{label}] episode {summary.episode} | "
            f"steps {summary.steps} | reward {summary.total_reward:+.3f} | "
            f"reason {summary.termination_reason} | "
            f"epsilon {summary.epsilon:.3f} | loss {loss}"
        )
