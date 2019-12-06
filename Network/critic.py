import torch
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, s, a):
        x = torch.cat([s, a], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        return x


class CriticNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(CriticNet, self).__init__()
        self.linear1 = nn.Linear(state_dim, 128)
        self.linear2 = nn.Linear(128, 32)
        self.linear3 = nn.Linear(32 + action_dim, 64)
        self.linear4 = nn.Linear(64, 1)

    def forward(self, s, a):
        x = F.relu(self.linear1(s))
        x = F.relu(self.linear2(x))
        x = torch.cat([x, a], 1)
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))

        return x
