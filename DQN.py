import os
from typing import Dict, List, Tuple

import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from IPython.display import clear_output
from torch.utils.data import DataLoader

from model.lstm import LSTMClassifier
from data.tcp import PcapDataset, split_dataset


class Network(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        """Initialization."""
        super(Network, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        return self.layers(x)


class DQNAgent:
    """DQN Agent interacting with environment.

    """

    def __init__(
            self,
            max_epsilon: float = 1.0,
            min_epsilon: float = 0.1,
            gamma: float = 0.99,
    ):
        """Initialization.

        Args:

        """
        obs_dim = 2 + 3 * 64 * 2
        action_dim = 2

        self.env = LSTMClassifier()
        self.env.load_state_dict(torch.load("model.pth"))

        self.epsilon = max_epsilon
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon

        self.gamma = gamma

        # device: cpu / gpu
        self.device = torch.device(
            "cpu"
        )

        # networks: dqn, dqn_target
        self.dqn = Network(obs_dim, action_dim).to(self.device)
        self.dqn_target = Network(obs_dim, action_dim).to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()

        # optimizer
        self.optimizer = optim.Adam(self.dqn.parameters())

        # transition to store in memory
        self.transition = list()

        # mode: train / test
        self.is_test = False

    def select_action(self, state):
        """Select an action from the input state."""
        # epsilon greedy policy
        if self.epsilon > np.random.random():
            selected_action = torch.Tensor([random.randint(0, 1)])
        else:
            selected_action = self.dqn(
                torch.FloatTensor(state).to(self.device)
            ).argmax()
            selected_action = selected_action.detach().cpu().numpy()

        return selected_action

    def update_model(self, state, action, next_state, reward) -> torch.Tensor:
        """Update the model by gradient descent."""

        loss = self._compute_dqn_loss(state, action, next_state, reward)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def _compute_dqn_loss(self, state, action, next_state, reward) -> torch.Tensor:
        """Return dqn loss."""

        # G_t   = r + gamma * v(s_{t+1})  if state != Terminal
        #       = r                       otherwise
        action = action.long()
        curr_q_value = self.dqn(state.clone().detach()).gather(0, action)
        next_q_value = self.dqn_target(
            next_state.clone().detach()
        ).max(dim=0, keepdim=True)[0].detach()

        target = (reward + self.gamma * next_q_value).to(self.device)

        # calculate dqn loss
        loss = F.smooth_l1_loss(curr_q_value, target)

        return loss

    def _target_hard_update(self):
        """Hard update: target <- local."""
        self.dqn_target.load_state_dict(self.dqn.state_dict())


if __name__ == "__main__":
    agent = DQNAgent()
    dataset = PcapDataset()
    count = 0
    train_dataset, val_dataset = split_dataset(dataset, seed=2024)
    for (inputs, targets) in train_dataset:
        count += 1
        agent.env.reset()
        state = torch.zeros([2 + 3 * 64 * 2])
        idx = 0
        reward = 0
        while idx <= 95:
            out, h, c = agent.env.step_forward(inputs[idx:idx + 5])
            next_state = torch.cat((out.flatten(), h.flatten(), c.flatten()))
            action = agent.select_action(state)
            idx += 5
            reward -= 1
            if action == 1 and torch.max(out, 0)[1] == targets:
                reward += 10
            agent.update_model(state, action, next_state, reward)
            state = next_state
            if action == 1:
                break

        if count % 10 == 0:
            print(count)
            agent._target_hard_update()

    score = 0
    sum_step = 0

    for (inputs, targets) in val_dataset:

        agent.env.reset()
        state = torch.zeros([2 + 3 * 64 * 2])
        idx = 0
        while idx <= 95:

            out, h, c = agent.env.step_forward(inputs[idx:idx + 5])
            idx += 5
            next_state = torch.cat((out.flatten(), h.flatten(), c.flatten()))
            action = agent.select_action(state)
            if action == 1 and torch.max(out, 0)[1] == targets:
                score += 1
                sum_step += idx
                break
            # if idx == 95 and torch.max(out, 0) == targets:
            #     score += 1

print(score, sum_step/80)
