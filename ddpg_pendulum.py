import torch
import gym
from torch import nn
from torch.nn import functional as F
from collections import deque
import random

env = gym.make('Pendulum-v1')
n_states = env.observation_space
n_actions = env.action_space
print(f"状态数：{n_states}，动作数：{n_actions}")

# 演员网络
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

# 评论家网络
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
    
class Config():
    def __init__(self) -> None:
        self.epsilon_start = 0.95
        self.epsilon_end = 0.01
        self.train_epoch = 400
        self.tets_epoch = 200

class replay_buffer():
    def __init__(self, capacity) -> None:
        self.capacity = capacity
        self.buffer = deque(maxlen=self.capacity)
    def push(self, state, action, reward, next_state, done):
        '''将与环境互动的数据加入经验回放区'''
        self.buffer.append([state, action, reward, next_state, done])
    def sample(self, batch_size):
        '''从经验回放区采样数据'''
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        batch = random.sample(self.buffer, batch_size)
        return zip(*batch)
    def clear(self):
        '''清空缓冲区'''
        self.buffer.clear()
    def __len__(self):
        '''获取缓冲区长度'''
        return len(self.buffer)