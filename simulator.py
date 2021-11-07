import pygame
import numpy as np
from constants import DISPLAY_WIDTH, DISPLAY_HEIGHT, white, black, ring_radius, road_width, velocity, radius, \
    acceleration
from vehicle import Car, EnvVehicle, Agent
from matplotlib import pyplot as plt

class Env:
    def __init__(self):
        """Constructor Method
        """
        self.screen = pygame.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT))
        pygame.display.set_caption("Ring Road Simulator")
        self.clock = pygame.time.Clock()
        self.agents = pygame.sprite.Group()
        self.env_vehicles = pygame.sprite.Group()
        pygame.init()
        self.create_background()

    def create_background(self):
        self.screen.fill(white)
        pygame.draw.circle(self.screen, black, (DISPLAY_WIDTH / 2, DISPLAY_HEIGHT / 2), ring_radius, road_width)

    def rotate_image_display(self, image, angle, x, y):
        rotated_image = pygame.transform.rotate(image, angle)
        new_rect = rotated_image.get_rect(center=image.get_rect(center=(x, y)).center)
        self.screen.blit(rotated_image, new_rect)

    def reset(self, envs, agents):
        total_no = envs + agents
        degree_spacing = 360/total_no
        positions = np.arange(total_no) * degree_spacing
        vehicle_list = []
        for i in range(len(positions)):
            if i < envs:
                vehicle_list.append(EnvVehicle(positions[i], velocity, acceleration, i))
            else:
                vehicle_list.append(Agent(positions[i], velocity, acceleration, i))

        for i in range(len(vehicle_list)):
            cur_veh = vehicle_list[i]
            front_vehicle = vehicle_list[(i+1) % len(vehicle_list)]
            if(i - 1 < 0):
                back_vehicle = vehicle_list[len(vehicle_list) - 1]
            else:
                back_vehicle = vehicle_list[i-1]
            cur_veh.front_vehicle = front_vehicle
            cur_veh.back_vehicle = back_vehicle
            if isinstance(cur_veh, EnvVehicle):
                self.env_vehicles.add(cur_veh)
            else:
                self.agents.add(cur_veh)

    def step(self):
        self.create_background()
        for sprite in self.agents:
            sprite.update()
        for sprite in self.env_vehicles:
            sprite.idm_control()


        self.render()

    def render(self):
        self.check_quit()
        for sprite in self.env_vehicles:
            self.rotate_image_display(sprite.image, sprite.rotation, sprite.xpos, sprite.ypos)
        for sprite in self.agents:
            self.rotate_image_display(sprite.image, sprite.rotation, sprite.xpos, sprite.ypos)
        pygame.display.update()
        self.clock.tick(20)

    def check_quit(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()


