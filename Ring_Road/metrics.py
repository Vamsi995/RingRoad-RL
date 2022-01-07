import math

import numpy as np

from Ring_Road.constants import RADIUS_PIX, FPS, ACTION_FREQ
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
        self.register_cars()

    def step(self):
        self.store_v(self.env.action_steps)
        self.store_xy(self.env.action_steps)
        self.running_mean_vel(self.env.action_steps)

    def register_cars(self):
        for veh in self.env.env_veh:
            self.position[veh.id] = []
            self.velocity[veh.id] = []
        for veh in self.env.agents:
            self.position[veh.id] = []
            self.velocity[veh.id] = []
        self.total_veh = len(self.position)

    def store_xy(self, t):
        for veh in self.env.env_veh:
            distance = veh.central_angle * RADIUS_PIX
            self.position[veh.id].append((t, distance))
        for veh in self.env.agents:
            distance = veh.central_angle * RADIUS_PIX
            self.position[veh.id].append((t, distance))

    def store_v(self, t):
        for veh in self.env.env_veh:
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

        if (len(veh_v) * t) - 1 != 0:
            dev /= ((len(veh_v) * t) - 1)
            self.running_deviation.append(dev ** 0.5)

    def throughput(self):
        self.throughput = self.running_mean[-1] * self.total_veh / (2 * math.pi * RADIUS_PIX)

    def findIndexes(self, pos):
        indices = []
        prev = pos[0]
        for i in range(len(pos)):
            if prev > pos[i]:
                indices.append(i)
            prev = pos[i]
        return indices

    def convert_action_steps_to_time(self, x):
        time_sec = len(x) / (FPS // ACTION_FREQ)
        new_x = np.linspace(0, time_sec, len(x))
        return new_x

    def get_sliced_arrays(self, indices, times, pos, vel):

        return_list = []

        for i in range(len(indices)):
            if i == 0:
                ind = indices[i]
                data_tup = (times[0:ind], pos[0:ind], vel[0:ind])
            elif i == len(indices) - 1:
                ind = indices[i]
                data_tup = (times[ind:], pos[ind:], vel[ind:])
            else:
                prev_ind = indices[i - 1]
                curr_ind = indices[i]
                data_tup = (times[prev_ind: curr_ind], pos[prev_ind: curr_ind], vel[prev_ind:curr_ind])

            return_list.append(data_tup)
        return return_list

    def plot(self):
        self.plot_positions()
        self.plot_velocities()
        self.plot_avg_vel()

    def plot_positions(self):
        global s
        plot_data = self.env.env_veh + self.env.agents
        for veh in plot_data:
            x, y = zip(*self.position[veh.id])
            t, v = zip(*self.velocity[veh.id])
            s = plt.scatter(self.convert_action_steps_to_time(x), y, c=v, cmap=plt.get_cmap("viridis"), marker='.')
        plt.colorbar(s, label="Velocity (m/s)")
        plt.xlabel("Time (s)")
        plt.ylabel("Position (m)")
        plt.show()

    def plot_velocities(self):
        for veh in self.env.env_veh:
            x, y = zip(*self.velocity[veh.id])
            plt.plot(self.convert_action_steps_to_time(x), y, color='gray')
        for ag in self.env.agents:
            x, y = zip(*self.velocity[ag.id])
            plt.plot(self.convert_action_steps_to_time(x), y, color='r')
        plt.xlabel("Time (s)")
        plt.ylabel("Velocity (m/s)")
        plt.show()

    def plot_avg_vel(self):

        plt.plot(self.convert_action_steps_to_time(self.running_mean), self.running_mean, color='#1B2ACC')

        plt.fill_between(self.convert_action_steps_to_time(self.running_mean),
                         np.array(self.running_mean) - np.array(self.running_deviation),
                         np.array(self.running_mean) + np.array(self.running_deviation), antialiased=True, alpha=0.2,
                         edgecolor='#1B2ACC', facecolor='#089FFF')

        plt.xlabel("Time (s)")
        plt.ylabel("Spatially-Averaged Velocity (m/s)")
        plt.show()
