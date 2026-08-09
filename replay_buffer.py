"""Uniform experience replay for DQN training."""

from collections import deque
from dataclasses import dataclass
import random
from typing import Optional, Sequence, Tuple

from config import CONFIG


State = Tuple[float, ...]


@dataclass(frozen=True, slots=True)
class Transition:
    state: State
    action: int
    reward: float
    next_state: Optional[State]
    done: bool
    termination_reason: str


@dataclass(frozen=True, slots=True)
class ReplayBatch:
    states: Tuple[State, ...]
    actions: Tuple[int, ...]
    rewards: Tuple[float, ...]
    next_states: Tuple[Optional[State], ...]
    dones: Tuple[bool, ...]
    termination_reasons: Tuple[str, ...]

    def __len__(self):
        return len(self.actions)


class UniformReplayBuffer:
    """Fixed-capacity buffer that samples every transition uniformly."""

    def __init__(
        self,
        capacity,
        batch_size,
        learning_starts,
        seed=None
    ):
        if capacity <= 0:
            raise ValueError("Replay capacity must be positive.")
        if batch_size <= 0:
            raise ValueError("Replay batch_size must be positive.")
        if batch_size > capacity:
            raise ValueError("Replay batch_size cannot exceed capacity.")
        if learning_starts < batch_size:
            raise ValueError(
                "Replay learning_starts must be at least batch_size."
            )
        if learning_starts > capacity:
            raise ValueError(
                "Replay learning_starts cannot exceed capacity."
            )

        self.capacity = int(capacity)
        self.batch_size = int(batch_size)
        self.learning_starts = int(learning_starts)
        self._transitions = deque(maxlen=self.capacity)
        self._random = random.Random(seed)

    def __len__(self):
        return len(self._transitions)

    @property
    def is_ready(self):
        """Whether enough experiences exist to begin DQN updates."""
        return len(self) >= self.learning_starts

    def add(
        self,
        state: Sequence[float],
        action: int,
        reward: float,
        next_state: Optional[Sequence[float]],
        done: bool,
        termination_reason="running"
    ):
        """Copy and store one environment transition."""
        reason = getattr(termination_reason, "value", termination_reason)

        transition = Transition(
            state=tuple(float(value) for value in state),
            action=int(action),
            reward=float(reward),
            next_state=(
                tuple(float(value) for value in next_state)
                if next_state is not None
                else None
            ),
            done=bool(done),
            termination_reason=str(reason)
        )
        self._transitions.append(transition)

    def sample(self, batch_size=None):
        """Return an unbiased random batch without removing transitions."""
        requested_size = (
            self.batch_size
            if batch_size is None
            else int(batch_size)
        )

        if requested_size <= 0:
            raise ValueError("Requested replay batch size must be positive.")
        if requested_size > len(self):
            raise ValueError(
                f"Cannot sample {requested_size} transitions from "
                f"a buffer containing {len(self)}."
            )

        transitions = self._random.sample(
            list(self._transitions),
            requested_size
        )

        return ReplayBatch(
            states=tuple(item.state for item in transitions),
            actions=tuple(item.action for item in transitions),
            rewards=tuple(item.reward for item in transitions),
            next_states=tuple(item.next_state for item in transitions),
            dones=tuple(item.done for item in transitions),
            termination_reasons=tuple(
                item.termination_reason
                for item in transitions
            )
        )

    def clear(self):
        self._transitions.clear()


def create_replay_buffer(training_config=None):
    """Build the configured training replay buffer."""
    if training_config is None:
        training_config = CONFIG["training"]
    replay_config = training_config["replay_buffer"]

    if replay_config["type"] != "uniform":
        raise ValueError(
            f'Unsupported replay buffer type: {replay_config["type"]}'
        )

    return UniformReplayBuffer(
        capacity=replay_config["capacity"],
        batch_size=replay_config["batch_size"],
        learning_starts=replay_config["learning_starts"],
        seed=training_config["seed"]
    )
