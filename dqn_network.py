"""Compact fully connected Q-network for the excavator controller."""

from torch import nn


class DQNNetwork(nn.Module):
    """Map the 10-value observation to one Q-value per discrete action."""

    def __init__(self, state_size, action_size, hidden_sizes=(64, 64)):
        super().__init__()

        if state_size <= 0 or action_size <= 0:
            raise ValueError("State and action sizes must be positive.")
        if not hidden_sizes or any(size <= 0 for size in hidden_sizes):
            raise ValueError("DQN hidden sizes must be positive.")

        layers = []
        input_size = int(state_size)

        for hidden_size in hidden_sizes:
            layers.extend((nn.Linear(input_size, hidden_size), nn.ReLU()))
            input_size = hidden_size

        layers.append(nn.Linear(input_size, int(action_size)))
        self.model = nn.Sequential(*layers)

    def forward(self, observation):
        return self.model(observation)

    @property
    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())
