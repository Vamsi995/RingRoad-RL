import pygame

DISPLAY_WIDTH = 1200
DISPLAY_HEIGHT = 1000
white = (255, 255, 255)
black = (0, 0, 0)
ring_radius = 450
road_width = 50
car_width = 40
car_length = 80
radius = 425
velocity = 0
acceleration = 0


def clean_image(mycar_image):
    road_color = (0, 0, 0)
    threshold_white = 220  # checking for border colours close to WHITE=(255,255,255)
    threshold_black = 20  # checking for black border colours close to BLACK=(0,0,0)
    for x in range(mycar_image.get_width()):
        for y in range(mycar_image.get_height()):  # scanning through the all pixels of my_car.image
            color = mycar_image.get_at((x, y))
            if color.r > threshold_white and color.g > threshold_white and color.b > threshold_white:
                mycar_image.set_at((x, y), (0, 0, 0, 0))


env_car_image = pygame.image.load("Sprites/env_vehicle.png")
env_car_image = pygame.transform.scale(env_car_image, (car_length, car_width))

agent_car_image = pygame.image.load("Sprites/mycar.png")
agent_car_image = pygame.transform.scale(agent_car_image, (car_length, car_width))

up_arrow = pygame.image.load("Sprites/all_arrows_up.png")
down_arrow = pygame.image.load("Sprites/all_arrows_down.png")

clean_image(agent_car_image)
clean_image(env_car_image)

FPS = 30  # Frames per second
DELTA_T = 1 / FPS

"""
IDM Parameters
"""
a = 3
b = 3
T = 1.5
s0 = 2
v0 = 20
IDM_DELTA = 4

DISCOUNT_FACTOR = 0.9
MAX_EPISODE_LENGTH = 2000
AGENT_MAX_VELOCITY = 20
ENV_VEHICLES = 19
AGENTS = 1
CONTROL_XPOS = 0
CONTROL_YPOS = 700
