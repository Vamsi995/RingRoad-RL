import pygame

DISPLAY_WIDTH = 1200
DISPLAY_HEIGHT = 1000
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RING_RADIUS = 22.5
ROAD_WIDTH = 2.5
CAR_WIDTH = 2
CAR_LENGTH = 4
RADIUS = 21.25
velocity = 0
INITIAL_ACCELERATION = 0
PIXEL_CONVERSION = 20

CAR_PIX_LENGTH = CAR_LENGTH * PIXEL_CONVERSION
CAR_PIX_WIDTH = CAR_WIDTH * PIXEL_CONVERSION
RING_PIX_RADIUS = int(RING_RADIUS * PIXEL_CONVERSION)
ROAD_PIX_WIDTH = int(ROAD_WIDTH * PIXEL_CONVERSION)
RADIUS_PIX = RADIUS * PIXEL_CONVERSION


def clean_image(mycar_image):
    road_color = (0, 0, 0)
    threshold_white = 220  # checking for border colours close to WHITE=(255,255,255)
    threshold_black = 20  # checking for black border colours close to BLACK=(0,0,0)
    for x in range(mycar_image.get_width()):
        for y in range(mycar_image.get_height()):  # scanning through the all pixels of my_car.image
            color = mycar_image.get_at((x, y))
            if color.r > threshold_white and color.g > threshold_white and color.b > threshold_white:
                mycar_image.set_at((x, y), (0, 0, 0, 0))


env_car_image = pygame.image.load("/home/vamsi/Documents/GitHub/RingRoad-RL/Ring_Road/sprites/env_vehicle.png")
env_car_image = pygame.transform.scale(env_car_image, (CAR_PIX_LENGTH, CAR_PIX_WIDTH))

agent_car_image = pygame.image.load("/home/vamsi/Documents/GitHub/RingRoad-RL/Ring_Road/sprites/mycar.png")
agent_car_image = pygame.transform.scale(agent_car_image, (CAR_PIX_LENGTH, CAR_PIX_WIDTH))

up_arrow = pygame.image.load("/home/vamsi/Documents/GitHub/RingRoad-RL/Ring_Road/sprites/all_arrows_up.png")
down_arrow = pygame.image.load("/home/vamsi/Documents/GitHub/RingRoad-RL/Ring_Road/sprites/all_arrows_down.png")

clean_image(agent_car_image)
clean_image(env_car_image)

FPS = 30  # Frames per second
ACTION_FREQ = 6
DELTA_T = 1 / FPS

"""
IDM Parameters
"""
a = 2
b = 3
T = 1
s0 = 2
v0 = 30
IDM_DELTA = 4

DISCOUNT_FACTOR = 0.9
MAX_EPISODE_LENGTH = 600
AGENT_MAX_VELOCITY = 20
ENV_VEHICLES = 19
AGENTS = 1
CONTROL_XPOS = 0
CONTROL_YPOS = 700
