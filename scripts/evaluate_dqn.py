import gym
from baselines.common import models
from stable_baselines import DQN

from Ring_Road.control.DQN import learn
from Ring_Road.metrics import Metrics


def main():
    env = gym.make("ringroad-v1", enable_render=True, agent_type="dqn")
    model = DQN.load("Models/DQN/DQN5")

    obs = env.reset(eval_mode=True)
    done = False
    met = Metrics(env)

    while not done:
        action, _states = model.predict(obs)
        obs, rew, done, _ = env.step(action=action, eval_mode=True)
        met.step()
        env.render()

    met.plot()
    env.close()
