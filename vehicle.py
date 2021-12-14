import math
import pygame

from constants import agent_car_image, radius, DISPLAY_WIDTH, DISPLAY_HEIGHT, env_car_image, s0, T, a, b, IDM_DELTA, v0, \
    DELTA_T, car_length, car_width, AGENT_MAX_VELOCITY
import numpy as np


class Car(pygame.sprite.Sprite):

    def __init__(self, rad, velocity, acceleration, id):
        pygame.sprite.Sprite.__init__(self)
        self.image = agent_car_image
        self.rect = self.image.get_rect()
        self.length = self.rect.height
        self.width = self.rect.width
        self.id = id
        self.initial_xpos = DISPLAY_WIDTH / 2
        self.initial_ypos = DISPLAY_HEIGHT / 2
        self.v, self.acc = velocity, acceleration
        self.rad = math.radians(rad)
        self.rotation = 90 - math.degrees(self.rad)
        self.xpos = self.initial_xpos + math.cos(self.rad) * radius
        self.ypos = self.initial_ypos + math.sin(self.rad) * radius
        # self.rect.x = self.xpos
        # self.rect.y = self.ypos
        # self.rect.width = self.image.get_width()
        # self.rect.height = self.image.get_height()
        self.distance_covered = 0
        self.front_vehicle = {}
        self.back_vehicle = {}

    def update(self):
        self.v += self.acc
        self.rad += (self.v / radius)
        self.rad = self.rad % (2 * math.pi)
        self.distance_covered = radius * self.rad
        if self.v < 0:
            self.v = 0
            return
        self.xpos = self.initial_xpos + math.cos(self.rad) * radius
        self.ypos = self.initial_ypos + math.sin(self.rad) * radius
        self.rotation = 90 - math.degrees(self.rad)

    def update_position(self):
        self.rad += (self.v / radius)
        self.rad = self.rad % (2 * math.pi)
        self.xpos = self.initial_xpos + math.cos(self.rad) * radius
        self.ypos = self.initial_ypos + math.sin(self.rad) * radius
        ext = (car_length / 2) / radius
        ext += self.rad
        sprite_x = self.initial_xpos + math.cos(ext) * radius
        sprite_y = self.initial_xpos + math.sin(ext) * radius
        self.rect.x = sprite_x
        self.rect.y = sprite_y
        self.rect.width = self.image.get_width()
        self.rect.height = self.image.get_height()
        self.rotation = 90 - math.degrees(self.rad)

class Agent(Car):
    def __init__(self, rad, velocity, acceleration, id):
        super(Agent, self).__init__(rad, velocity, acceleration, id)
        self.image = agent_car_image
        self.rect = self.image.get_rect()
        self.length = self.rect.height
        self.width = self.rect.width
        self.rect.x = self.xpos
        self.rect.y = self.ypos

    def follower_stopper(self, desired_velocity, curvatures, initial_x):
        v_lead = self.front_vehicle.v
        v = min(max(v_lead, 0), desired_velocity)
        delta_v = v_lead - self.v
        delta_v = min(delta_v, 0)

        delta_x1 = initial_x[0] + (1 / (2 * curvatures[0])) * (delta_v ** 2)
        delta_x2 = initial_x[1] + (1 / (2 * curvatures[1])) * (delta_v ** 2)
        delta_x3 = initial_x[2] + (1 / (2 * curvatures[2])) * (delta_v ** 2)

        if self.front_vehicle.rad - self.rad < 0:
            s = (2 * math.pi - self.rad + self.front_vehicle.rad) * radius - car_length
        else:
            s = (self.front_vehicle.rad - self.rad) * radius - car_length

        if s <= delta_x1:
            self.v = 0
        elif delta_x2 >= s > delta_x1:
            self.v = v * ((s - delta_x1) / (delta_x2 - delta_x1))
        elif delta_x3 >= s > delta_x2:
            self.v = v + (desired_velocity - v) * ((s - delta_x2) / (delta_x3 - delta_x2))
        elif s > delta_x3:
            self.v = desired_velocity

        self.rad += (self.v / radius)
        self.rad = self.rad % (2 * math.pi)

        self.xpos = self.initial_xpos + math.cos(self.rad) * radius
        self.ypos = self.initial_ypos + math.sin(self.rad) * radius
        self.rotation = 90 - math.degrees(self.rad)

    def a2c(self, action):
        self.acc = action[0]
        # print(action)
        prev_vel = self.v
        self.v = max(0, min(self.v + (self.acc * DELTA_T), AGENT_MAX_VELOCITY))
        self.acc = (self.v - prev_vel) / DELTA_T
        # print(self.acc)
        self.update_position()

    def dqn(self, action):
        if action == 0:
            self.acc += 0.5
        elif action == 1:
            self.acc -= 2

        prev_vel = self.v
        self.v = max(0, min(self.v + (self.acc * DELTA_T), AGENT_MAX_VELOCITY))
        self.acc = (self.v - prev_vel) / DELTA_T

        self.update_position()

    def idm_control(self):
        delta_v = self.v - self.front_vehicle.v
        if self.front_vehicle.rad - self.rad < 0:
            s = (2 * math.pi - self.rad + self.front_vehicle.rad) * radius - car_length - 10
        else:
            s = (self.front_vehicle.rad - self.rad) * radius - car_length - 10

        if s <= 0:
            s = 0.00001

        s_star = s0 + max(0, (self.v * T) + ((self.v * delta_v) / (2 * np.power(a * b, 0.5))))
        self.acc = a * (1 - np.power(self.v / v0, IDM_DELTA) - np.power(s_star / s, 2))
        old_velocity = self.v
        self.v = max(0, min((old_velocity) + (self.acc * DELTA_T), v0))
        self.acc = (self.v - old_velocity) / DELTA_T  # adjustment to acc due to clamping of velocity

        self.update_position()

    def update(self):


        if self.v < 0:
            self.v = 0
            self.update_position()
            return

        prev_vel = self.v
        self.v = max(0, min(self.v + (self.acc * DELTA_T), AGENT_MAX_VELOCITY))
        self.acc = (self.v - prev_vel) / DELTA_T

        self.update_position()


class EnvVehicle(Car):
    def __init__(self, rad, velocity, acceleration, id):
        super(EnvVehicle, self).__init__(rad, velocity, acceleration, id)
        self.image = env_car_image
        self.rect = self.image.get_rect()
        self.length = self.rect.height
        self.width = self.rect.width
        self.rect.x = self.xpos
        self.rect.y = self.ypos

    def idm_control(self):
        delta_v = self.v - self.front_vehicle.v
        if self.front_vehicle.rad - self.rad < 0:
            s = (2 * math.pi - self.rad + self.front_vehicle.rad) * radius - car_length - 10
        else:
            s = (self.front_vehicle.rad - self.rad) * radius - car_length - 10

        if s <= 0:
            s = 0.00001

        s_star = s0 + max(0, (self.v * T) + ((self.v * delta_v) / (2 * np.power(a * b, 0.5))))
        self.acc = a * (1 - np.power(self.v / v0, IDM_DELTA) - np.power(s_star / s, 2))
        old_velocity = self.v
        self.v = max(0, min((old_velocity) + (self.acc * DELTA_T), v0))
        self.acc = (self.v - old_velocity) / DELTA_T  # adjustment to acc due to clamping of velocity

        self.update_position()