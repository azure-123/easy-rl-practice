import gym

env = gym.make('CartPole-v1') # 载入环境
env.reset() # 重置环境
for _ in range(1000):
    env.render() # 显示图形界面
    action = env.action_space.sample() # 随机采样动作
    observation, reward, done, info = env.step(action) # 采取动作
env.close() # 关闭环境