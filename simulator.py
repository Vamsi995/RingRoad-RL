import math

import pygame
import numpy as np
from constants import DISPLAY_WIDTH, DISPLAY_HEIGHT, white, black, ring_radius, road_width, velocity, radius, \
    acceleration, FPS, DISCOUNT_FACTOR, MAX_EPISODE_LENGTH, car_length, ENV_VEHICLES, AGENTS, up_arrow, down_arrow, \
    CONTROL_XPOS, CONTROL_YPOS, car_width
from vehicle import Car, EnvVehicle, Agent
from matplotlib import pyplot as plt
from gym import Env, spaces


class BaseEnv(Env):
    def __init__(self):
        """Constructor Method
        """
        # self.agents = pygame.sprite.Group()
        # self.env_vehicles = pygame.sprite.Group()
        self.agents = []
        self.env_vehicles = []
        # vlead, vlag, vav, hlead, hlag
        features_low = np.array([0, 0, 0, 0, 0], dtype=np.float64)
        features_high = np.array([20, 20, 20, 2500, 2500], dtype=np.float64)

        self.observation_space = spaces.Box(low=features_low, high=features_high, dtype=np.float64)
        self.action_space = spaces.Discrete(2)
        # self.action_space = spaces.Box(low=np.array([-7]), high=np.array([2]), dtype=np.float64)

        self.state = None
        self.reward = None
        self.step_number = 0
        self.discount_factor = DISCOUNT_FACTOR

    def initialize_state(self, envs, agents):
        self.agents.clear()
        self.env_vehicles.clear()
        total_no = envs + agents
        degree_spacing = 360 / total_no
        positions = np.arange(total_no) * degree_spacing
        vehicle_list = []
        for i in range(len(positions)):
            if i < envs:
                vehicle_list.append(EnvVehicle(positions[i], np.random.randint(low=0, high=3), acceleration, i, car_width, car_length))
                # vehicle_list.append(EnvVehicle(positions[i], 0, acceleration, i, car_width, car_length))

            else:
                vehicle_list.append(Agent(positions[i], np.random.randint(low=0, high=3), acceleration, i, car_width, car_length))
                # vehicle_list.append(Agent(positions[i], 0, acceleration, i, car_width, car_length))

        for i in range(len(vehicle_list)):
            cur_veh = vehicle_list[i]
            front_vehicle = vehicle_list[(i + 1) % len(vehicle_list)]
            if (i - 1 < 0):
                back_vehicle = vehicle_list[len(vehicle_list) - 1]
            else:
                back_vehicle = vehicle_list[i - 1]
            cur_veh.front_vehicle = front_vehicle
            cur_veh.back_vehicle = back_vehicle
            if isinstance(cur_veh, EnvVehicle):
                self.env_vehicles.append(cur_veh)
            else:
                self.agents.append(cur_veh)

    def gap_front(self, veh):
        if veh.front_vehicle.rad - veh.rad < 0:
            sf = (2 * math.pi - veh.rad + veh.front_vehicle.rad) * radius - car_length
        else:
            sf = (veh.front_vehicle.rad - veh.rad) * radius - car_length
        return sf

    def calculate_headways(self, ag):
        if ag.front_vehicle.rad - ag.rad < 0:
            sf = (2 * math.pi - ag.rad + ag.front_vehicle.rad) * radius - car_length - 10
        else:
            sf = (ag.front_vehicle.rad - ag.rad) * radius - car_length - 10

        if ag.rad - ag.back_vehicle.rad < 0:
            sb = (2 * math.pi + ag.rad - ag.back_vehicle.rad) * radius - car_length - 10
        else:
            sb = (ag.rad - ag.back_vehicle.rad) * radius - car_length - 10

        if sf <= 0:
            sf = 0.00001
        if sb <= 0:
            sb = 0.00001

        return sf, sb
        # if ag.v != 0:
        #     hlead = sf/ag.v
        #     hback = sb/ag.v
        #     return hlead, hback
        # else:
        #     return 0, 0

    def extract_state_features(self):
        feature_list = []
        for ag in self.agents:
            feature_list.append(ag.front_vehicle.v)
            feature_list.append(ag.back_vehicle.v)
            feature_list.append(ag.v)
            hlead, hback = self.calculate_headways(ag)
            feature_list.append(hlead)
            feature_list.append(hback)
        return feature_list

    def reward_fn(self):
        vav = self.state[2]
        vlead = self.state[0]
        vlag = self.state[1]
        sf = self.state[3]
        sb = self.state[4]

        reward = 0
        if vlead > 0 and vlag > 0:
            reward += 5
        else:
            reward -= 450

        if sf < 40:
            reward -= 450

        return reward
        # return vav


class RenderEnv(BaseEnv):

    def __init__(self):
        super().__init__()
        self.screen = pygame.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT))
        pygame.display.set_caption("Ring Road Simulator")
        self.clock = pygame.time.Clock()

        pygame.init()
        self.create_background()

    def create_background(self):
        self.screen.fill(white)
        pygame.draw.circle(self.screen, black, (DISPLAY_WIDTH / 2, DISPLAY_HEIGHT / 2), ring_radius, road_width)

    def rotate_image_display(self, image, angle, x, y):
        rotated_image = pygame.transform.rotate(image, angle)
        new_rect = rotated_image.get_rect(center=image.get_rect(center=(x, y)).center)
        self.screen.blit(rotated_image, new_rect)

    def reset(self):
        self.create_background()
        self.initialize_state(ENV_VEHICLES, AGENTS)
        self.done = False
        self.step_number = 0
        self.state = self.extract_state_features()
        self.render()
        return np.array(self.state)

    def display_controls(self, action):
        if action is None:
            return

        # if action[0] > 0:
        #     self.screen.blit(up_arrow, [CONTROL_XPOS, CONTROL_YPOS])
        #
        # if action[0] < 0:
        #     self.screen.blit(down_arrow, [CONTROL_XPOS, CONTROL_YPOS])

    def step(self, action):
        self.create_background()
        self.step_number += 1

        for sprite in self.agents:
            sprite.idm_control()
            # sprite.a2c(action)
            # sprite.follower_stopper(4, [2.5, 2.0, 1.5], [14.5, 15.25, 16])
            # sprite.dqn(action)

        for sprite in self.env_vehicles:
            sprite.idm_control()

        for ag in self.agents:
            sf = self.gap_front(ag)
            if sf <= 0:
                self.done = True
                info = {}
                self.render()
                self.reward = -10
                # print("hitting this")
                return np.array(self.state), self.reward, self.done, info

        self.state = self.extract_state_features()
        self.reward = self.reward_fn()

        if self.step_number == MAX_EPISODE_LENGTH:
            self.done = True
        info = {}

        self.render(action)
        return np.array(self.state), self.reward, self.done, info

    def render(self, action=None):
        self.check_quit()
        for sprite in self.env_vehicles:
            self.rotate_image_display(sprite, sprite.rotation, sprite.xpos, sprite.ypos)
        for sprite in self.agents:
            self.rotate_image_display(sprite, sprite.rotation, sprite.xpos, sprite.ypos)

        self.display_controls(action)

        pygame.display.update()
        self.clock.tick(FPS)

    def check_quit(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()


class NoRenderEnv(BaseEnv):

    def __init__(self):
        super(NoRenderEnv, self).__init__()

    def reset(self):

        self.initialize_state(ENV_VEHICLES, AGENTS)
        self.done = False
        self.step_number = 0
        self.state = self.extract_state_features()

        return np.array(self.state)

    def step(self, action):
        self.step_number += 1
        for sprite in self.agents:
            sprite.dqn(action)
            # sprite.a2c(action)
            # sprite.follower_stopper(4, [2.5, 2.0, 1.5], [14.5, 15.25, 16])

        for sprite in self.env_vehicles:
            sprite.idm_control()

        for ag in self.agents:
            sf = self.gap_front(ag)
            if sf <= 0:
                self.done = True
                info = {}
                self.reward = -10000
                return np.array(self.state), self.reward, self.done, info

        self.state = self.extract_state_features()
        self.reward = self.reward_fn()

        if self.step_number == MAX_EPISODE_LENGTH:
            self.done = True
        info = {}

        return np.array(self.state), self.reward, self.done, info
