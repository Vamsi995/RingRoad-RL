import gym
from baselines.common import models
from stable_baselines import DQN

from Ring_Road.control.DQN import learn
from Ring_Road.metrics import Metrics


def main():
    env = gym.make("ringroad-v1", enable_render=True)
    model = DQN.load("Models/DQN")

    # for i in range(iterations):
    obs = env.reset()
    done = False
    met = Metrics(env)

    while not done:
        obs, rew, done, _ = env.step()
        met.step()
        env.render()

    met.plot()
    env.close()
