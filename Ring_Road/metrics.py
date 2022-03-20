import math
from typing import List

import numpy as np

from Ring_Road.constants import RADIUS_PIX, FPS, ACTION_FREQ, RADIUS
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
            distance = veh.central_angle * RADIUS
            self.position[veh.id].append((t, distance))
        for veh in self.env.agents:
            distance = veh.central_angle * RADIUS
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
        if FPS // ACTION_FREQ == 1:
            time_sec = len(x) / FPS
        else:
            time_sec = len(x) / (FPS // ACTION_FREQ)
        new_x = np.linspace(0, time_sec, len(x))
        return new_x

    def plot(self):
        self.plot_positions()
        self.plot_velocities()
        self.plot_avg_vel()

    def plot_positions(self):
        plt.figure(figsize=(15, 5))
        global s
        plot_data = self.env.env_veh + self.env.agents
        for veh in plot_data:
            x, y = zip(*self.position[veh.id])
            t, v = zip(*self.velocity[veh.id])
            s = plt.scatter(self.convert_action_steps_to_time(x), y, c=v, cmap=plt.get_cmap("viridis"), marker='.')
        plt.colorbar(s, label="Velocity (m/s)")
        plt.xlabel("Time (s)")
        plt.ylabel("Position (m)")
        if self.env.algorithm == "dqn":
            plt.savefig("Plots/DQN/SpaceTime_New1.png")
        elif self.env.algorithm == "ppo":
            plt.savefig("Plots/PPO/SpaceTimeSmoothed_New1.png")
        elif self.env.algorithm == "fs":
            plt.savefig("Plots/FollowerStopper/SpaceTimeSmoothed_New1.png")
        elif self.env.algorithm == "pi":
            plt.savefig("Plots/PISaturation/SpaceTimeSmoothed_New1.png")

    def plot_velocities(self):
        plt.figure(figsize=(15, 5))
        for veh in self.env.env_veh:
            x, y = zip(*self.velocity[veh.id])
            plt.plot(self.convert_action_steps_to_time(x), self.smooth(y, 0.9), color='gray')
        for ag in self.env.agents:
            x, y = zip(*self.velocity[ag.id])
            plt.plot(self.convert_action_steps_to_time(x), self.smooth(y, 0.9), color='r')
        plt.xlabel("Time (s)")
        plt.ylabel("Velocity (m/s)")
        if self.env.algorithm == "dqn":
            plt.savefig("Plots/DQN/VelocityProfile_New1.png")
        elif self.env.algorithm == "ppo":
            plt.savefig("Plots/PPO/VelocityProfileSmoothed_New1.png")
        elif self.env.algorithm == "fs":
            plt.savefig("Plots/FollowerStopper/VelocityProfileSmoothed_New1.png")
        elif self.env.algorithm == "pi":
            plt.savefig("Plots/PISaturation/VelocityProfileSmoothed_New1.png")

    def plot_avg_vel(self):
        plt.figure(figsize=(15, 5))
        plt.plot(self.convert_action_steps_to_time(self.running_mean), self.running_mean, color='#1B2ACC')

        plt.fill_between(self.convert_action_steps_to_time(self.running_mean),
                         np.array(self.running_mean) - np.array(self.running_deviation),
                         np.array(self.running_mean) + np.array(self.running_deviation), antialiased=True, alpha=0.2,
                         edgecolor='#1B2ACC', facecolor='#089FFF')

        plt.xlabel("Time (s)")
        plt.ylabel("Spatially-Averaged Velocity (m/s)")
        if self.env.algorithm == "dqn":
            plt.savefig("Plots/DQN/AverageVelocity_New1.png")
        elif self.env.algorithm == "ppo":
            plt.savefig("Plots/PPO/AverageVelocitySmoothed_New1.png")
        elif self.env.algorithm == "fs":
            plt.savefig("Plots/FollowerStopper/AverageVelocitySmoothed_New1.png")
        elif self.env.algorithm == "pi":
            plt.savefig("Plots/PISaturation/AverageVelocitySmoothed_New1.png")

    def smooth(self, scalars: List[float], weight: float) -> List[float]:  # Weight between 0 and 1
        last = scalars[0]  # First value in the plot (first timestep)
        smoothed = list()
        for point in scalars:
            smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
            smoothed.append(smoothed_val)  # Save it
            last = smoothed_val  # Anchor the last smoothed value

        return smoothed
