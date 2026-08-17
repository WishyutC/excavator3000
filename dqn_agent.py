"""DQN action selection, replay learning, target updates, and checkpoints."""

from dataclasses import asdict, dataclass
from pathlib import Path
import random

import torch
from torch import nn

from config import CONFIG
from dqn_network import DQNNetwork
from replay_buffer import create_replay_buffer


@dataclass(frozen=True)
class LearningMetrics:
    loss: float
    mean_q: float
    mean_target: float
    gradient_norm: float
    epsilon: float
    buffer_size: int
    training_steps: int

    def to_info(self):
        return asdict(self)


def resolve_device(requested_device):
    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(requested_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but PyTorch cannot access it.")
    return device


class DQNAgent:
    def __init__(self, state_size, action_size, training_config=None):
        self.config = training_config or CONFIG["training"]
        self.state_size = int(state_size)
        self.action_size = int(action_size)
        self.hidden_sizes = tuple(self.config["hidden_sizes"])
        self.device = resolve_device(self.config["device"])

        seed = int(self.config["seed"])
        self.random = random.Random(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        self.online_network = DQNNetwork(
            self.state_size,
            self.action_size,
            self.hidden_sizes
        ).to(self.device)
        self.target_network = DQNNetwork(
            self.state_size,
            self.action_size,
            self.hidden_sizes
        ).to(self.device)
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.target_network.eval()

        self.optimizer = torch.optim.Adam(
            self.online_network.parameters(),
            lr=self.config["learning_rate"]
        )
        self.loss_function = nn.SmoothL1Loss()
        self.replay_buffer = create_replay_buffer(self.config)
        self.environment_steps = 0
        self.training_steps = 0

    @property
    def epsilon(self):
        epsilon_config = self.config["epsilon"]
        decay_steps = max(1, int(epsilon_config["decay_steps"]))
        progress = min(1.0, self.environment_steps / decay_steps)
        return (
            epsilon_config["start"]
            + progress
            * (epsilon_config["end"] - epsilon_config["start"])
        )

    def select_action(self, state, evaluate=False):
        if len(state) != self.state_size:
            raise ValueError(
                f"Expected state size {self.state_size}, received {len(state)}."
            )

        if not evaluate and self.random.random() < self.epsilon:
            return self.random.randrange(self.action_size)

        state_tensor = torch.tensor(
            state,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            return int(self.online_network(state_tensor).argmax(dim=1).item())

    def remember(
        self,
        state,
        action,
        reward,
        next_state,
        done,
        termination_reason="running"
    ):
        self.replay_buffer.add(
            state,
            action,
            reward,
            next_state,
            done,
            termination_reason
        )
        self.environment_steps += 1

    def learn(self):
        if not self.replay_buffer.is_ready:
            return None
        if self.environment_steps % self.config["train_every_steps"] != 0:
            return None

        batch = self.replay_buffer.sample()
        states = torch.tensor(
            batch.states,
            dtype=torch.float32,
            device=self.device
        )
        actions = torch.tensor(
            batch.actions,
            dtype=torch.long,
            device=self.device
        ).unsqueeze(1)
        rewards = torch.tensor(
            batch.rewards,
            dtype=torch.float32,
            device=self.device
        ) * float(self.config.get("reward_scale", 1.0))
        dones = torch.tensor(
            batch.dones,
            dtype=torch.float32,
            device=self.device
        )
        next_states = torch.tensor(
            tuple(
                state if state is not None else (0.0,) * self.state_size
                for state in batch.next_states
            ),
            dtype=torch.float32,
            device=self.device
        )

        selected_q = self.online_network(states).gather(1, actions).squeeze(1)

        with torch.no_grad():
            next_q = self._next_state_values(next_states)
            targets = rewards + self.config["gamma"] * (1.0 - dones) * next_q

        loss = self.loss_function(selected_q, targets)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.online_network.parameters(),
            self.config["gradient_clip_norm"]
        )
        self.optimizer.step()
        self.training_steps += 1

        if self.training_steps % self.config["target_update_steps"] == 0:
            self.sync_target_network()

        return LearningMetrics(
            loss=float(loss.item()),
            mean_q=float(selected_q.detach().mean().item()),
            mean_target=float(targets.mean().item()),
            gradient_norm=float(gradient_norm),
            epsilon=float(self.epsilon),
            buffer_size=len(self.replay_buffer),
            training_steps=self.training_steps
        )

    def _next_state_values(self, next_states):
        """Return target values using Double DQN selection when configured."""

        if self.config.get("double_dqn", False):
            next_actions = self.online_network(next_states).argmax(
                dim=1,
                keepdim=True
            )
            return self.target_network(next_states).gather(
                1,
                next_actions
            ).squeeze(1)
        return self.target_network(next_states).max(dim=1).values

    def sync_target_network(self):
        self.target_network.load_state_dict(self.online_network.state_dict())

    def save_checkpoint(self, path, episode=0, extra=None):
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_size": self.state_size,
            "action_size": self.action_size,
            "hidden_sizes": self.hidden_sizes,
            "online_network": self.online_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "environment_steps": self.environment_steps,
            "training_steps": self.training_steps,
            "double_dqn": bool(self.config.get("double_dqn", False)),
            "reward_scale": float(self.config.get("reward_scale", 1.0)),
            "episode": int(episode),
            "extra": extra or {}
        }, checkpoint_path)
        return checkpoint_path

    def load_checkpoint(self, path, load_optimizer=True):
        checkpoint = torch.load(
            Path(path),
            map_location=self.device,
            weights_only=False
        )

        if checkpoint["state_size"] != self.state_size:
            raise ValueError("Checkpoint state size does not match environment.")
        if checkpoint["action_size"] != self.action_size:
            raise ValueError("Checkpoint action size does not match environment.")
        if tuple(checkpoint["hidden_sizes"]) != self.hidden_sizes:
            raise ValueError("Checkpoint hidden sizes do not match configuration.")

        self.online_network.load_state_dict(checkpoint["online_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        if load_optimizer and "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.environment_steps = int(checkpoint.get("environment_steps", 0))
        self.training_steps = int(checkpoint.get("training_steps", 0))
        return checkpoint
