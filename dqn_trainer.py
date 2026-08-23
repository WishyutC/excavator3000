"""Training and evaluation orchestration around the Webots environment."""

from dataclasses import dataclass
from pathlib import Path

from config import CONFIG
from training_logger import TrainingLogger


@dataclass(frozen=True)
class EpisodeSummary:
    episode: int
    stage_episode: int
    curriculum_stage: str
    curriculum_target_checkpoint: int | None
    map_name: str
    steps: int
    decisions: int
    total_reward: float
    mean_reward_per_step: float
    termination_reason: str
    success: bool
    epsilon: float
    buffer_size: int
    training_steps: int
    loss: float | None
    min_goal_distance_m: float | None
    final_goal_distance_m: float | None
    track_progress: float
    checkpoints_reached: int
    checkpoint_count: int
    action_forward_pct: float
    action_left_pct: float
    action_right_pct: float

    def to_dict(self):
        return self.__dict__.copy()


class DQNTrainer:
    def __init__(
        self,
        environment,
        agent,
        training_config=None,
        logger=None,
        episode_offset=0
    ):
        self.environment = environment
        self.agent = agent
        self.config = training_config or CONFIG["training"]
        self.logger = logger or TrainingLogger()
        self.episode_offset = int(episode_offset)
        self.best_success_reward = float("-inf")
        self.best_candidate_reward = float("-inf")

    def train(self, episodes=None, start_episode=0):
        episode_count = int(episodes or self.config["episodes"])
        summaries = []

        for stage_episode in range(int(start_episode) + 1, episode_count + 1):
            episode = self.episode_offset + stage_episode
            summary, simulation_stopped = self._run_episode(
                episode,
                stage_episode,
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

    def evaluate(self, episodes=None, verbose=True, log_episodes=False):
        episode_count = int(episodes or CONFIG["evaluation"]["episodes"])
        summaries = []

        for episode in range(1, episode_count + 1):
            summary, simulation_stopped = self._run_episode(
                episode,
                episode,
                training=False
            )
            if simulation_stopped:
                break
            summaries.append(summary)
            if log_episodes:
                self.logger.log_episode(summary.to_dict())
            if verbose:
                self._print_summary(summary, "EVAL")

        return summaries

    def _run_episode(self, episode, stage_episode, training):
        state = self.environment.reset()
        total_reward = 0.0
        step = 0
        done = False
        latest_metrics = None
        action_counts = [0] * self.agent.action_size
        minimum_goal_distance = None
        info = {
            "termination_reason": "running",
            "is_success": False
        }

        while not done:
            forced_action = self.config.get("forced_action")
            expert_policy = self.config.get("expert_policy")
            if training and forced_action is not None:
                action = int(forced_action)
            elif training and expert_policy == "waypoint":
                action = self.environment.waypoint_expert_action()
            elif training and expert_policy == "sensor":
                action = self.environment.sensor_expert_action(state)
            else:
                action = self.agent.select_action(state, evaluate=not training)
            if not 0 <= action < self.agent.action_size:
                raise ValueError("forced_action is outside the policy action space.")
            action_counts[action] += 1
            next_state, reward, done, info = self.environment.step(action)

            if next_state is None:
                return self._summary(
                    episode,
                    stage_episode,
                    step,
                    total_reward,
                    info,
                    latest_metrics,
                    action_counts,
                    minimum_goal_distance
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
            goal_distance = info.get("goal_distance_m")
            if goal_distance is not None:
                minimum_goal_distance = (
                    goal_distance
                    if minimum_goal_distance is None
                    else min(minimum_goal_distance, goal_distance)
                )
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
            episode,
            stage_episode,
            step,
            total_reward,
            info,
            latest_metrics,
            action_counts,
            minimum_goal_distance
        ), False

    def _summary(
        self,
        episode,
        stage_episode,
        decisions,
        total_reward,
        info,
        metrics,
        action_counts,
        minimum_goal_distance
    ):
        physics_steps = int(info.get("episode_steps", decisions))
        action_total = max(1, sum(action_counts))
        progress = info.get("progress", {})

        def action_percentage(index):
            return (
                100.0 * action_counts[index] / action_total
                if index < len(action_counts)
                else 0.0
            )

        return EpisodeSummary(
            episode=episode,
            stage_episode=stage_episode,
            curriculum_stage=str(info.get("curriculum_stage", "single")),
            curriculum_target_checkpoint=info.get(
                "curriculum_target_checkpoint"
            ),
            map_name=str(getattr(self.environment, "map_name", "unknown")),
            steps=physics_steps,
            decisions=decisions,
            total_reward=total_reward,
            mean_reward_per_step=(
                total_reward / physics_steps if physics_steps else 0.0
            ),
            termination_reason=info["termination_reason"],
            success=bool(info.get("is_success", False)),
            epsilon=float(self.agent.epsilon),
            buffer_size=len(self.agent.replay_buffer),
            training_steps=self.agent.training_steps,
            loss=metrics.loss if metrics is not None else None,
            min_goal_distance_m=minimum_goal_distance,
            final_goal_distance_m=info.get("goal_distance_m"),
            track_progress=float(progress.get("progress_fraction", 0.0)),
            checkpoints_reached=int(progress.get("checkpoints_reached", 0)),
            checkpoint_count=int(progress.get("checkpoint_count", 0)),
            action_forward_pct=action_percentage(0),
            action_left_pct=action_percentage(1),
            action_right_pct=action_percentage(2)
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
        if summary.success and summary.total_reward > self.best_success_reward:
            self.best_success_reward = summary.total_reward
            self._save_checkpoint(self.config["best_checkpoint_name"], summary)
        elif (
            not summary.success
            and summary.total_reward > self.best_candidate_reward
        ):
            self.best_candidate_reward = summary.total_reward
            self._save_checkpoint(
                self.config["candidate_checkpoint_name"],
                summary
            )

        interval = int(self.config["save_every_episodes"])
        if interval > 0 and summary.episode % interval == 0:
            self._save_checkpoint(self.config["checkpoint_name"], summary)

    def _save_checkpoint(self, filename, summary):
        path = Path(self.config["save_directory"]) / filename
        self.agent.save_checkpoint(
            path,
            episode=summary.episode,
            extra={
                "episode_summary": summary.to_dict(),
                "best_success_reward": self.best_success_reward,
                "best_candidate_reward": self.best_candidate_reward
            }
        )

    @staticmethod
    def _print_summary(summary, label):
        loss = "warming up" if summary.loss is None else f"{summary.loss:.5f}"
        print(
            f"[{label}] episode {summary.episode} | "
            f"stage {summary.curriculum_stage}:{summary.stage_episode} | "
            f"steps {summary.steps} | reward {summary.total_reward:+.3f} | "
            f"reason {summary.termination_reason} | "
            f"progress {summary.checkpoints_reached}/{summary.checkpoint_count} | "
            f"actions F/L/R {summary.action_forward_pct:.0f}/"
            f"{summary.action_left_pct:.0f}/{summary.action_right_pct:.0f}% | "
            f"epsilon {summary.epsilon:.3f} | loss {loss}"
        )
