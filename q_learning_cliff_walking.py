import gym
from envs.gridworld_env import CliffWalkingWrapper

env = gym.make('CliffWalking-v0')
env = CliffWalkingWrapper(env)
