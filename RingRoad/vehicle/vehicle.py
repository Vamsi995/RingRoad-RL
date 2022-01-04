import math
import numpy as np
import pygame

from RingRoad.constants import DISPLAY_WIDTH, DISPLAY_HEIGHT, radius, IDM_DELTA, v0, s0, a, DELTA_T, b, T, LENGTH


class Car(pygame.Surface):

    def __init__(self, central_angle, velocity, acceleration, id, width, height):
        pygame.Surface.__init__(self, (width, height), pygame.SRCALPHA)
        self.id = id
        self.initial_xpos = DISPLAY_WIDTH / 2
        self.initial_ypos = DISPLAY_HEIGHT / 2
        self.v, self.acc = velocity, acceleration
        self.central_angle = math.radians(central_angle)
        self.rotation = 180 - math.degrees(self.central_angle)
        self.xpos = self.initial_xpos + math.cos(self.central_angle) * radius
        self.ypos = self.initial_ypos + math.sin(self.central_angle) * radius

        self.distance_covered = 0
        self.front_vehicle = {}
        self.back_vehicle = {}

    def update_positions(self):
        self.central_angle += (self.v / radius)
        self.central_angle = self.central_angle % (2 * math.pi)
        self.xpos = self.initial_xpos + math.cos(self.central_angle) * radius
        self.ypos = self.initial_ypos + math.sin(self.central_angle) * radius
        self.rotation = 180 - math.degrees(self.central_angle)


class Agent(Car):
    def __init__(self, central_angle, velocity, acceleration, id, width, height):
        super(Agent, self).__init__(central_angle, velocity, acceleration, id, width, height)
        self.fill((255, 0, 0))

    def step(self):
        self.update_positions()


class EnvVehicle(Car):
    def __init__(self, central_angle, velocity, acceleration, id, width, height):
        super(EnvVehicle, self).__init__(central_angle, velocity, acceleration, id, width, height)
        self.fill((0, 0, 255))

    def idm_control(self):
        delta_v = self.v - self.front_vehicle.v
        if self.front_vehicle.central_angle - self.central_angle < 0:
            s = (2 * math.pi - self.central_angle + self.front_vehicle.central_angle) * radius - LENGTH / 2
        else:
            s = (self.front_vehicle.central_angle - self.central_angle) * radius - LENGTH / 2

        if s <= 0:
            s = 0.00001

        s_star = s0 + max(0, (self.v * T) + ((self.v * delta_v) / (2 * np.power(a * b, 0.5))))
        self.acc = a * (1 - np.power(self.v / v0, IDM_DELTA) - np.power(s_star / s, 2))
        old_velocity = self.v
        self.v = max(0, min((old_velocity) + (self.acc * DELTA_T), v0))
        self.acc = (self.v - old_velocity) / DELTA_T  # adjustment to acc due to clamping of velocity

    def step(self):
        self.idm_control()
        self.update_positions()
