import gym
from envs import CliffWalkingWapper

env = gym.make('CliffWalking-v0')
env = CliffWalkingWapper(env)
