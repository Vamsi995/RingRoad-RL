import gym
import Ring_Road

env = gym.make('ringroad-v0')
env.reset()
while True:
    action = 0
    obs, reward, done, info = env.step(action)
    env.render()