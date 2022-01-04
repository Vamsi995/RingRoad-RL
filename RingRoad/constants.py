import pygame

DISPLAY_WIDTH = 1200
DISPLAY_HEIGHT = 1000
white = (255, 255, 255)
black = (0, 0, 0)
ring_radius = 400
road_width = 20
WIDTH = 2
LENGTH = 5
radius = 410
velocity = 0
acceleration = 0


FPS = 5  # Frames per second
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
MAX_EPISODE_LENGTH = 1000
AGENT_MAX_VELOCITY = 20
ENV_VEHICLES = 10
AGENTS = 1