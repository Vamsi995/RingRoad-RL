import math
import pygame

from constants import agent_car_image, radius, DISPLAY_WIDTH, DISPLAY_HEIGHT, env_car_image, s0, T, a, b, IDM_DELTA, v0, \
    DELTA_T, car_length, car_width
import numpy as np


class Car(pygame.sprite.Sprite):

    def __init__(self, rad, velocity, acceleration):
        pygame.sprite.Sprite.__init__(self)
        self.initial_xpos = DISPLAY_WIDTH / 2
        self.initial_ypos = DISPLAY_HEIGHT / 2
        self.v, self.acc = velocity, acceleration
        self.rad = math.radians(rad)
        self.rotation = 90 - math.degrees(self.rad)
        self.xpos = self.initial_xpos + math.cos(self.rad) * radius
        self.ypos = self.initial_ypos + math.sin(self.rad) * radius
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


class Agent(Car):
    def __init__(self, rad, velocity, acceleration):
        super().__init__(rad, velocity, acceleration)
        self.image = agent_car_image
        self.rect = self.image.get_rect()
        self.length = self.rect.height
        self.width = self.rect.width

class EnvVehicle(Car):
    def __init__(self, rad, velocity, acceleration):
        super().__init__(rad, velocity, acceleration)
        self.image = env_car_image
        self.rect = self.image.get_rect()
        self.length = self.rect.height
        self.width = self.rect.width

    def idm_control(self):
        delta_v = self.v - self.front_vehicle.v
        if(self.front_vehicle.rad - self.rad < 0):
            s = (2*math.pi - self.rad + self.front_vehicle.rad) * radius - 100
        else:
            s = (self.front_vehicle.rad - self.rad) * radius - 100

        if s <= 0:
            s = 0.00001

        s_star = s0 + max(0, (self.v * T) + ((self.v * delta_v) / (2 * np.power(a * b, 0.5))))
        self.acc = a * (1 - np.power(self.v / v0, IDM_DELTA) - np.power(s_star / s, 2))
        old_velocity = self.v
        self.v = max(0, min((old_velocity) + (self.acc * DELTA_T), v0))
        self.acc = (self.v - old_velocity) / DELTA_T  # adjustment to acc due to clamping of velocity

        self.rad += (self.v / radius)
        self.rad = self.rad % (2 * math.pi)

        self.xpos = self.initial_xpos + math.cos(self.rad) * radius
        self.ypos = self.initial_ypos + math.sin(self.rad) * radius
        self.rotation = 90 - math.degrees(self.rad)