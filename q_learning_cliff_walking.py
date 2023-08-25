import gym
from envs.gridworld_env import CliffWalkingWrapper
import numpy as np
import random

# 引入gym库中的环境
env = gym.make('CliffWalking-v0')
env = CliffWalkingWrapper(env)

# 获取状态数和动作数
n_states = env.observation_space.n
n_actions = env.action_space.n
print(f"状态数：{n_states}，动作数：{n_actions}")

# 获取初始状态
state = env.reset()
print(f"初始状态：{state}")

class q_learning():
    def __init__(self, cfg, n_states, n_actions):
        self.epsilon = cfg.epsilon
        self.lr = cfg.lr
        self.Q_table = np.zeros(shape=(n_states, n_actions))
        self.n_actions = n_actions
        self.n_states = n_states
        self.gamma = cfg.gamma
    def sample_actions(self, state):
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
            Q_target = reward + self.gamma * np.argmax(self.Q_table[next_state])
        self.Q_table[state][action] += self.lr * (Q_target - Q_predict)

