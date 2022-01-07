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
    met.register_cars()
    t = 0
    while not done:
        action, _states = model.predict(obs)
        obs, rew, done, _ = env.step(action)
        env.render()
        met.store_v(t)
        met.store_xy(t)
        met.running_mean_vel(t)
        t += 1

    # env.reset()
    # met = Metrics(env)
    # t = 0
    # met.register_cars()
    #
    # while t < TIME_STEPS:
    # # while True:
    #     # keys = pygame.key.get_pressed()
    #     # if keys[K_UP]:
    #     #     for ag in env.agents:
    #     #         ag.acc += 0.1
    #     # if keys[K_DOWN]:
    #     #     for ag in env.agents:
    #     #         ag.acc -= 1
    #
    #     env.step(3)
    #     met.store_xy(t)
    #     met.store_v(t)
    #     met.running_mean_vel(t)
    #     t += 1

    met.plot_positions()
    met.plot_velocities()

