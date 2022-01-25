import gym
import pygame
from pygame import K_LEFT, K_RIGHT, K_DOWN, K_UP
import Ring_Road

env = gym.make("ringroad-v1", enable_render=True, agent_type="man")
obs = env.reset()


def main():
    """
    This is the driver function.

    Returns
    -------
    None
    """


    while True:

        keys = pygame.key.get_pressed()
        if keys[K_DOWN]:
            env.step(1)
            print("dowm")
        elif keys[K_UP]:
            env.step(0)
            print("up")
        # if THREE_LANE:
        #     threelane_controls(env, keys)
        # else:
        #     twolane_controls(env, keys)
        env.render()


if __name__ == '__main__':
    main()