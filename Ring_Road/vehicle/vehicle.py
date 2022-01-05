import math
import numpy as np
import pygame

from Ring_Road.constants import DISPLAY_WIDTH, DISPLAY_HEIGHT, IDM_DELTA, v0, s0, a, DELTA_T, b, T, LENGTH, \
    RADIUS


class Car():

    def __init__(self, central_angle, velocity, acceleration, id, width, length):
        self.width = width
        self.length = length
        self.id = id
        self.initial_xpos = DISPLAY_WIDTH / 2
        self.initial_ypos = DISPLAY_HEIGHT / 2
        self.v, self.acc = velocity, acceleration
        self.central_angle = math.radians(central_angle)
        self.rotation = 180 - math.degrees(self.central_angle)
        self.xpos = self.initial_xpos + math.cos(self.central_angle) * RADIUS
        self.ypos = self.initial_ypos + math.sin(self.central_angle) * RADIUS

        self.distance_covered = 0
        self.front_vehicle = {}
        self.back_vehicle = {}

    def render(self, color):
        self.vehicle_surface = pygame.Surface((self.width, self.length),
                                         flags=pygame.SRCALPHA)  # per-pixel alpha

        rect = (0, 0, self.width, self.length)
        pygame.draw.rect(self.vehicle_surface, color, rect, 0, 4)

        rotated_image = pygame.transform.rotate(self.vehicle_surface, self.rotation)
        new_rect = rotated_image.get_rect(center=self.vehicle_surface.get_rect(center=(self.xpos, self.ypos)).center)

        return rotated_image, new_rect

    def update_positions(self):
        self.central_angle += (self.v / RADIUS)
        self.central_angle = self.central_angle % (2 * math.pi)
        self.xpos = self.initial_xpos + math.cos(self.central_angle) * RADIUS
        self.ypos = self.initial_ypos + math.sin(self.central_angle) * RADIUS
        self.rotation = - math.degrees(self.central_angle)


class Agent(Car):
    def __init__(self, central_angle, velocity, acceleration, id, width, height):
        super(Agent, self).__init__(central_angle, velocity, acceleration, id, width, height)
        # self.fill((255, 0, 0))
        self.crashed = False
        self.stored_action = None

    def step(self):
        self.update_positions()


class EnvVehicle(Car):
    def __init__(self, central_angle, velocity, acceleration, id, width, height):
        super(EnvVehicle, self).__init__(central_angle, velocity, acceleration, id, width, height)
        # self.fill((0, 0, 255))

    def idm_control(self):
        delta_v = self.v - self.front_vehicle.v
        if self.front_vehicle.central_angle - self.central_angle < 0:
            s = (2 * math.pi - self.central_angle + self.front_vehicle.central_angle) * RADIUS - LENGTH
        else:
            s = (self.front_vehicle.central_angle - self.central_angle) * RADIUS - LENGTH

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
