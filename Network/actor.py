import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, s):
        x = F.relu(self.linear1(s))
        x = F.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x))

        return x


class ActorNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(ActorNet, self).__init__()
        self.linear1 = nn.Linear(input_size, 64)
        self.linear2 = nn.Linear(64, 32)
        self.linear3_group = nn.ModuleList()
        for each_dst_num in output_size:
            self.linear3_group.append(nn.Linear(32, each_dst_num))

    def forward(self, s):
        x = F.relu(self.linear1(s))
        x = F.relu(self.linear2(x))
        x_cat = []
        for each_dst in self.linear3_group:
            tmp = F.softmax(each_dst(x))
            x_cat.append(tmp)
        out = torch.cat(x_cat, dim=1)
        return out
