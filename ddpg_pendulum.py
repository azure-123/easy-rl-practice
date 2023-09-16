import torch
import gym
from torch import nn
from torch.nn import functional as F
from collections import deque
import random
import numpy as np

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

# 设置参数
class Config():
    def __init__(self) -> None:
        self.epsilon_start = 0.95
        self.epsilon_end = 0.01
        self.train_epoch = 400
        self.test_epoch = 200
        self.device = 'gpu'
        self.tau = 1e-2 # 软更新参数

# 经验回放池
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
    
class DDPG():
    def __init__(self, actor, critic, buffer, cfg) -> None:
        self.device = cfg.device
        self.buffer = buffer.to(self.device)
        self.batch_size = cfg.batch_size
        self.actor = actor.to(self.device)
        self.target_actor = actor.to(self.device)
        self.critic = critic.to(self.device)
        self.target_critic = critic.to(self.device)
        self.actor_lr = cfg.actor_lr
        self.critic_lr = cfg.critic_lr
        self.gamma = cfg.gamma
        self.tau = cfg.tau
        # 将演员的参数复制到目标网络当中
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
        # 将评论家的参数复制到目标网络当中
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        # 优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic_lr.parameters(), lr=self.critic_lr)
    def sample_action(self, state):
        action = self.actor(state)
        return action.detach()
    @torch.no_grad()
    def predict_action(self, state):
        action = self.actor(state)
        return action
    def update(self):
        if len(self.buffer) < self.batch_size:
            return
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.buffer.sample(self.batch_size)
        state_batch = torch.tensor(np.array(state_batch), dtype=torch.float32, device=self.device)
        action_batch = torch.tensor(np.array(action_batch), dtype=torch.float32, device=self.device)
        reward_batch = torch.tensor(np.array(reward_batch), dtype=torch.float32, device=self.device)
        next_state_batch = torch.tensor(np.array(next_state_batch), dtype=torch.float32, device=self.device)
        done_batch = torch.tensor(np.array(done_batch), device=self.device)
        # 演员的更新
        actor_loss = self.critic(state_batch, self.actor(state_batch))
        actor_loss = -actor_loss.mean()
        # 评论家的更新
        next_action = self.target_actor(next_state_batch)
        target_value = self.target_critic(next_state_batch, next_action.detach())
        expected_value = reward_batch + (1 - done_batch) * self.gamma * target_value
        actual_value = self.critic(state_batch, action_batch)
        critic_loss = nn.MSELoss(actual_value, expected_value)
        # 更新两个网络
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        # 进行软更新
        for target_param, param in zip(self.target_actor.paramters(), self.actor.parameters()):
            target_param.data.copy_(target_param.data * (1 - self.tau) + param.data * self.tau)
        for target_param, param in zip(self.target_critic.parameters(), self.critic.paramters()):
            target_param.data.copy_(target_param.data * (1 - self.tau) + param.data * self.tau)

def train(cfg, env, agent):
    reward = []
    for step in range(cfg.traih_epoch):
        ep_reward = 0
        state = env.reset()
        while True:
            env.render()
            action = agent.sample_action(state)
            next_state, reward, done, _ = env.step(action)
            ep_reward += reward
            agent.buffer.push(state, action, reward, next_state, done)
            agent.update()
            state = next_state
            if done :
                break
        reward.append(ep_reward)
        if (step + 1) % 20 == 0:
            print(f"回合：{step + 1}/{cfg.train_epoch}，奖励：{ep_reward:.1f}，Epsilon：{agent.epsilon:.3f}")

def test(cfg, env, agent):
    reward = []