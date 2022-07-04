import math
from typing import List

import numpy as np
from ray.rllib import MultiAgentEnv

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
        # if t == 0:
        #     return
        # veh_v = []
        # for id, vel in self.velocity.items():
        #     x, y = zip(*vel)
        #     veh_v.append(sum(y))
        # mean = sum(veh_v) / (len(veh_v) * t)
        mean = 0
        num_veh = len(self.velocity)

        for id, vel in self.velocity.items():
            mean += vel[-1][1]
        mean /= num_veh
        self.running_mean.append(mean)

        deviation = 0
        for id, vel in self.velocity.items():
            deviation += (mean - vel[-1][1]) ** 2
        deviation /= num_veh
        deviation = deviation ** 0.5

        self.running_deviation.append(deviation)
        # dev = 0
        # for id, vel in self.velocity.items():
        #     x, y = zip(*vel)
        #     for v in y:
        #         dev += (mean - v) ** 2
        #
        # if (len(veh_v) * t) - 1 != 0:
        #     dev /= ((len(veh_v) * t) - 1)
        #     self.running_deviation.append(dev ** 0.5)

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

    def save_fig(self, name, config):
        if self.env.algorithm == "dqn":
            plt.savefig("Plots/DQN/" + name + ".png")
        elif self.env.algorithm == "ppo":
            if isinstance(self.env, MultiAgentEnv):
                if bool(config["multiagent"]):
                    if config["model"]["use_lstm"]:
                        plt.savefig("Plots/PPO/MultiAgent/NonSharedPolicy/" + name + ".png")
                    else:
                        plt.savefig("Plots/PPO/MultiAgent/NonSharedPolicy/" + name + ".png")
                else:
                    if config["model"]["use_lstm"]:
                        plt.savefig("Plots/PPO/MultiAgent/SharedPolicy/" + name + ".png")
                    else:
                        plt.savefig("Plots/PPO/MultiAgent/SharedPolicy/" + name + ".png")
            else:
                if config["model"]["use_lstm"]:
                    plt.savefig("Plots/PPO/SingleAgent/LSTM/" + name + ".png")
                else:
                    plt.savefig("Plots/PPO/SingleAgent/" + name + ".png")
        elif self.env.algorithm == "fs":
            plt.savefig("Plots/FollowerStopper/" + name + ".png")
        elif self.env.algorithm == "pi":
            plt.savefig("Plots/PISaturation/" + name + ".png")
        elif self.env.algorithm == "idm":
            plt.savefig("Plots/IDM/" + name + ".png")

    def plot(self, config):
        self.plot_positions(config)
        self.plot_velocities(config)
        self.plot_avg_vel(config)

    def plot_positions(self, config):
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
        self.save_fig("SpaceTime", config)

    def plot_velocities(self, config):
        plt.figure(figsize=(15, 5))
        for veh in self.env.env_veh:
            x, y = zip(*self.velocity[veh.id])
            plt.plot(self.convert_action_steps_to_time(x), self.smooth(y, 0.9), color='gray')
        for ag in self.env.agents:
            x, y = zip(*self.velocity[ag.id])
            plt.plot(self.convert_action_steps_to_time(x), self.smooth(y, 0.9), color='r')
        plt.xlabel("Time (s)")
        plt.ylabel("Velocity (m/s)")
        plt.grid()
        self.save_fig("VelocityProfile", config)

    def plot_avg_vel(self, config):
        plt.figure(figsize=(15, 5))
        plt.plot(self.convert_action_steps_to_time(self.running_mean), self.smooth(self.running_mean, 0.9), color='#1B2ACC')

        plt.fill_between(self.convert_action_steps_to_time(self.running_mean),
                         self.smooth(np.array(self.running_mean) - np.array(self.running_deviation), 0.9),
                         self.smooth(np.array(self.running_mean) + np.array(self.running_deviation), 0.9), antialiased=True, alpha=0.2,
                         edgecolor='#1B2ACC', facecolor='#089FFF')

        plt.xlabel("Time (s)")
        plt.ylabel("System Average Velocity (m/s)")
        plt.grid()
        self.save_fig("AverageVelocity", config)

    def smooth(self, scalars: List[float], weight: float) -> List[float]:  # Weight between 0 and 1
        last = scalars[0]  # First value in the plot (first timestep)
        smoothed = list()
        for point in scalars:
            smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
            smoothed.append(smoothed_val)  # Save it
            last = smoothed_val  # Anchor the last smoothed value

        return smoothed
