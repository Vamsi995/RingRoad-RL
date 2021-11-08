import math

import numpy as np

from constants import radius
from matplotlib import pyplot as plt


class Metrics:
    def __init__(self, env):
        self.env = env
        self.position = {}
        self.velocity = {}
        self.throughput = 0
        self.running_mean = [0]
        self.running_deviation = [0]
        self.total_veh = 0

    def register_cars(self):
        for veh in self.env.env_vehicles:
            self.position[veh.id] = []
            self.velocity[veh.id] = []
        for veh in self.env.agents:
            self.position[veh.id] = []
            self.velocity[veh.id] = []
        self.total_veh = len(self.position)

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

    def running_mean_vel(self, t):
        if t == 0:
            return
        veh_v = []
        for id, vel in self.velocity.items():
            x, y = zip(*vel)
            veh_v.append(sum(y))
        mean = sum(veh_v) / (len(veh_v) * t)
        self.running_mean.append(mean)

        dev = 0
        for id, vel in self.velocity.items():
            x, y = zip(*vel)
            for v in y:
                dev += (mean - v) ** 2

        dev /= ((len(veh_v) * t) - 1)
        self.running_deviation.append(dev ** 0.5)

    def throughput(self):
        self.throughput = self.running_mean[-1] * self.total_veh / (2 * math.pi * radius)

    def plot_positions(self):
        global s
        for veh in self.env.env_vehicles:
            x, y = zip(*self.position[veh.id])
            t, v = zip(*self.velocity[veh.id])
            s = plt.scatter(x, y, c=v, cmap=plt.get_cmap("viridis"))

        plt.colorbar(s,label="Velocity")
        plt.show()

    def plot_velocities(self):
        for veh in self.env.env_vehicles:
            x, y = zip(*self.velocity[veh.id])
            plt.plot(x, y, color='gray')
        for ag in self.env.agents:
            x, y = zip(*self.velocity[ag.id])
            plt.plot(x, y, color='r')
        plt.plot(np.arange(len(self.running_mean)), self.running_mean, color='green')
        # plt.plot(np.arange(len(self.running_deviation)),
        #          [x + y for x, y in zip(self.running_mean, self.running_deviation)], color='b')
        # plt.plot(np.arange(len(self.running_deviation)),
        #          [x - y for x, y in zip(self.running_mean, self.running_deviation)], color='b')

        plt.show()
