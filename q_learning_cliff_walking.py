import gym
from envs.gridworld_env import CliffWalkingWrapper
import numpy as np
import random
import math

# 引入gym库中的环境
env = gym.make('CliffWalking-v0')

# 获取状态数和动作数
n_states = env.observation_space.n
n_actions = env.action_space.n
print(f"状态数：{n_states}，动作数：{n_actions}")

# 获取初始状态
state = env.reset()
print(f"初始状态：{state}")

class q_learning():
    def __init__(self, cfg, n_states, n_actions):
        self.epsilon = cfg.epsilon_start
        self.epsilon_start = cfg.epsilon_start
        self.epsilon_end = cfg.epsilon_end
        self.lr = cfg.lr
        self.Q_table = np.zeros(shape=(n_states, n_actions))
        self.n_actions = n_actions
        self.n_states = n_states
        self.gamma = cfg.gamma
        self.count = 0
        self.epsilon_decay = cfg.epsilon_decay
    def sample_actions(self, state):
        self.count += 1
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            math.exp(-1. * self.count / self.epsilon_decay) # epsilon是会递减的，这里选择指数递减
        rand = random.random()
        if rand < self.epsilon:
            action = random.randint(0, self.n_actions - 1)
        else:
            action = np.argmax(self.Q_table[state])
        return action
    def predict_actions(self, state):
        return np.argmax(self.Q_table[state])
    def update_policy(self, state, action, reward, next_state, terminated):
        Q_predict = self.Q_table[state][action]
        if terminated:
            Q_target = reward
        else:
            Q_target = reward + self.gamma * np.max(self.Q_table[next_state])
        self.Q_table[state][action] += self.lr * (Q_target - Q_predict)

class Config():
    def __init__(self):
        self.lr = 0.1
        self.gamma = 0.9
        self.epochs = 1000
        self.epsilon_start = 0.95
        self.epsilon_end = 0.01
        self.epsilon_decay = 300
        self.epochs = 400
        self.test_epochs = 30
config = Config()
q_learner = q_learning(config, n_states, n_actions)

def train(cfg,env,agent):
    print('开始训练！')
    rewards = []  # 记录奖励
    for i_ep in range(cfg.epochs):
        ep_reward = 0  # 记录每个回合的奖励
        state = env.reset()
        while True:
            action = agent.sample_actions(state)  # 根据算法采样一个动作
            next_state, reward, terminated, info = env.step(action)  # 与环境进行一次动作交互
            agent.update_policy(state, action, reward, next_state, terminated)  # Q学习算法更新
            state = next_state  # 更新状态
            ep_reward += reward
            if terminated:
                break
        rewards.append(ep_reward)
        if (i_ep+1)%20==0:
            print(f"回合：{i_ep+1}/400，奖励：{ep_reward:.1f}，Epsilon：{agent.epsilon:.3f}")
    print('完成训练！')
    return {"rewards":rewards}
train(config, env, q_learner)
def test(cfg,env,agent):
    print('开始测试！')
    rewards = []  # 记录所有回合的奖励
    for i_ep in range(cfg.test_epochs):
        ep_reward = 0  # 记录每个episode的reward
        state = env.reset()  # 重置环境, 重新开一局（即开始新的一个回合）
        while True:
            action = agent.predict_actions(state)  # 根据算法选择一个动作
            next_state, reward, terminated, info = env.step(action)  # 与环境进行一个交互
            state = next_state  # 更新状态
            ep_reward += reward
            if terminated:
                break
        rewards.append(ep_reward)
        print(f"回合数：{i_ep+1}/{cfg.test_epochs}, 奖励：{ep_reward:.1f}")
    print('完成测试！')
    return {"rewards":rewards}

test(config, env, q_learner)