import gym
import torch
import random
from torch import nn
from torch.nn import functional as F
import copy
import numpy as np
import math

env = gym.make('CartPole-v1')
n_states = env.observation_space.shape[0] # 获取状态数
n_actions = env.action_space.n # 获取动作数
print(f"状态数：{n_states}，动作数：{n_actions}")

class replay_buffer():
    def __init__(self, capacity) -> None:
        self.buffer = []
        self.capacity = capacity
        self.position = 0
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity: # 若长度不够，说明缓冲区还没有满过，需要先装满容量
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity # 装满，覆盖已有数据
    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones
    def __len__(self):
        return len(self.buffer)
    
class MLP(nn.Module):
    def __init__(self, input_dims, output_dims, hidden_dims):
        super(MLP,self).__init__()
        self.fc1 = nn.Linear(input_dims, hidden_dims)
        self.fc2 = nn.Linear(hidden_dims, hidden_dims)
        self.fc3 = nn.Linear(hidden_dims, output_dims)
    def forward(self, X):
        res = F.relu(self.fc1(X))
        res = F.relu(self.fc2(res))
        return self.fc3(res)
    
class DQN():
    def __init__(self, model, buffer, cfg):
        self.policy_net = model
        self.target_net = copy.deepcopy(self.policy_net)
        self.epsilon = cfg.epsilon_start
        self.buffer = buffer
        self.n_actions = cfg.n_actions
        self.batch_size = cfg.batch_size
        self.lr = cfg.lr
        self.gamma = cfg.gamma
        self.epsilon_start = cfg.epsilon_start
        self.epsilon_end = cfg.epsilon_end
        self.epsilon_decay = cfg.epsilon_decay
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.count = 0
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(param.data)
        self.device = 'cpu'

    def sample_actions(self, state):
        self.count += 1
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            math.exp(-1. * self.count / self.epsilon_decay) 
        if random.random() > self.epsilon:
            with torch.no_grad():
                state = torch.tensor([state], dtype=torch.float32)
                q_value = self.policy_net(state)
                action = q_value.max(1)[1].item()
        else:
            action = random.randint(0, self.n_actions - 1)
        return action

    @torch.no_grad()
    def predict_actions(self, state):
        state = torch.tensor([state], dtype=torch.float32)
        q_value = self.policy_net(state)
        action = q_value.max(1)[1].item()
        return action

    def update(self):
        if len(self.buffer) < self.batch_size:
            return
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.buffer.sample(self.batch_size)
        states = torch.tensor(np.array(state_batch), dtype=torch.float)
        actions = torch.tensor(action_batch).unsqueeze(1)  
        rewards = torch.tensor(reward_batch, dtype=torch.float)
        next_states = torch.tensor(np.array(next_state_batch), dtype=torch.float)
        dones = torch.tensor(np.float32(done_batch))
        q_values = self.policy_net(states).gather(dim=1, index=actions)
        next_q_values = self.target_net(next_states).max(1)[0].detach()
        # 训练设置
        target_values = rewards + self.gamma * next_q_values * (1 - dones)
        loss = nn.MSELoss()(q_values, target_values.unsqueeze(1))
        # 开始训练
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

class config():
    def __init__(self, n_actions):
        self.epsilon = 0.2
        self.n_actions = n_actions
        self.batch_size = 64
        self.lr = 0.0001
        self.gamma = 0.95
        self.num_epochs = 200
        self.update_epochs = 20
        self.timesteps = 100000
        self.epsilon_start = 0.95
        self.epsilon_end = 0.01
        self.epsilon_decay = 500

capacity = 100000
hidden_dims = 64
cfg = config(n_actions)
buffer = replay_buffer(capacity)
net = MLP(n_states, n_actions, hidden_dims)
agent = DQN(net, buffer, cfg)

def train(cfg, env, agent):
    ep_rewards = []
    for epoch in range(cfg.num_epochs):
        state = env.reset()
        ep_reward = 0
        for t in range(cfg.timesteps):
            action = agent.sample_actions(state)
            next_state, reward, done, _ = env.step(action)
            agent.buffer.push(state, action, reward, next_state, done)
            state = next_state
            agent.update()
            ep_reward += reward
            if done:
                break
            
        if (epoch + 1) % cfg.update_epochs == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
        ep_rewards.append(ep_reward)
        if (epoch + 1) % 10 == 0:
            print(f"回合：{epoch + 1}/{cfg.num_epochs}，奖励：{ep_reward:.2f}")
    env.close()

def test(cfg, env, agent):
    ep_rewards = []
    for epoch in range(cfg.num_epochs):
        state = env.reset()
        ep_reward = 0
        for t in range(cfg.timesteps):
            action = agent.predict_actions(state)
            next_state, reward, done, _ = env.step(action)
            ep_reward += reward
            state = next_state
        ep_rewards.append(ep_reward)
        if (epoch + 1) % 10 == 0:
            print(f"回合：{epoch + 1}/{cfg.num_epochs}，奖励：{ep_reward:.2f}")
    env.close()

train(cfg, env, agent)
test(cfg, env, agent)