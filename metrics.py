import numpy as np

from constants import radius
from matplotlib import pyplot as plt


class Metrics:
    def __init__(self, env):
        self.env = env
        self.position = {}
        self.velocity = {}
        self.mean_vel = 0
        self.running_mean = []

    def register_cars(self):
        for veh in self.env.env_vehicles:
            self.position[veh.id] = []
            self.velocity[veh.id] = []
        for veh in self.env.agents:
            self.position[veh.id] = []
            self.velocity[veh.id] = []

    def store_xy(self, t):
        for veh in self.env.env_vehicles:
            distance = veh.rad * radius
            self.position[veh.id].append((t, distance))
        for veh in self.env.agents:
            distance = veh.rad * radius
            self.position[veh.id].append((t, distance))

    def store_v(self, t):
        for veh in self.env.env_vehicles:
            self.velocity[veh.id].append((t, veh.v))
        for veh in self.env.agents:
            self.velocity[veh.id].append((t, veh.v))

    def running_mean_vel(self):
        temp = []
        for veh in self.env.env_vehicles:
            temp.append(veh.v)
        for ag in self.env.agents:
            temp.append(ag.v)
        self.running_mean.append(sum(temp)/len(temp))

    def mean_velocity(self, timesteps):
        veh_v = []
        for id, vel in self.velocity.items():
            x, y = zip(*vel)
            veh_v.append(sum(y) / len(y))
        self.mean_vel = sum(veh_v) / len(veh_v)

    def plot_positions(self):
        for veh in self.env.env_vehicles:
            x, y = zip(*self.position[veh.id])
            plt.scatter(x, y)
        plt.show()

    def plot_velocities(self):
        for veh in self.env.env_vehicles:
            x, y = zip(*self.velocity[veh.id])
            plt.plot(x, y, color='gray')
        for ag in self.env.agents:
            x, y = zip(*self.velocity[ag.id])
            plt.plot(x, y, color='r')
        # plt.plot(np.arange(len(self.running_mean)), self.running_mean, color='r')
        plt.show()