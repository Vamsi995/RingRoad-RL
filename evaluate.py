import pygame
from baselines.common import models
from matplotlib import pyplot as plt
from pygame import K_UP, K_DOWN

from DQN import learn
from constants import FPS
from metrics import Metrics
from simulator import RenderEnv, NoRenderEnv


def main():
    TIME_STEPS = 2000

    env = RenderEnv()
    iterations = 200
    # env = NoRenderEnv()
    act = learn(env, network=models.mlp(num_hidden=64, num_layers=3), total_timesteps=0,
                load_path="Models/Model2.pkl", inbuild_network=True)


    # for i in range(iterations):
    obs = env.reset()
    done = False
    met = Metrics(env)
    met.register_cars()
    t = 0
    while not done:
        if t % FPS == 0:
            action = act(obs[None])[0]
            print(t)
            print("Hits")
        obs, rew, done, _ = env.step(action)
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


if __name__ == "__main__":
    main()