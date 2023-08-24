import gym
from envs.gridworld_env import CliffWalkingWapper

env = gym.make('CliffWalking-v0')
env = CliffWalkingWapper(env)
