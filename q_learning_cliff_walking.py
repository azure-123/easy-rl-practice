import gym
from envs.gridworld_env import CliffWalkingWrapper

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