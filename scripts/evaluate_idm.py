import gym
from baselines.common import models
from stable_baselines import DQN

from Ring_Road.control.DQN import learn
from Ring_Road.metrics import Metrics


def main():
    env = gym.make("ringroad-v1", enable_render=False, agent_type="idm")

    obs = env.reset(eval_mode=True)
    done = False
    met = Metrics(env)

    while not done:
        obs, rew, done, _ = env.step(eval_mode=True)
        met.step()
        env.render()
        # print(env.action_steps)

    met.plot()
    env.close()
