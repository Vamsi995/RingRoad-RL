import gym
from gym import register


class RingRoad(gym.Env):

    def __init__(self):
        self.action_space = None
        self.observation_space = None

        self.simulation_time = 0  # Simulation time
        self.action_steps = 0  # Actions performed
        self.done = False
        pass

    def reset(self):
        pass

    def step(self):
        pass

    def render(self):
        pass


register(
    id='ringroad-v0',
    entry_point='RingRoad.envs:RingRoad',
)