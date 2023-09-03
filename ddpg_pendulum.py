import torch
import gym
from torch import nn
from torch.nn import functional as F

env = gym.make('Pendulum-v1')
n_states = env.observation_space
n_actions = env.action_space
print(f"状态数：{n_states}，动作数：{n_actions}")

class Actor(nn.Module):
    def __init__(self, input_dims, hidden_dims, output_dims, init_w=3e-3):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(input_dims, hidden_dims)
        self.linear2 = nn.Linear(hidden_dims, hidden_dims)
        self.linear3 = nn.Linear(hidden_dims, output_dims)
        # 初始化参数
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return F.tanh(self.linear3(x))
    
class Critic(nn.Module):
    def __init__(self, input_dims, hidden_dims, output_dims, init_w=3e-3):
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(input_dims, hidden_dims)
        self.linear2 = nn.Linear(hidden_dims, hidden_dims)
        self.linear3 = nn.Linear(hidden_dims, output_dims)
        # 初始化参数
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return self.linear3(x)