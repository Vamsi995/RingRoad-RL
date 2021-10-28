import math
import pygame
from constants import mycar_image, radius, DISPLAY_WIDTH, DISPLAY_HEIGHT


class Car(pygame.sprite.Sprite):

    def __init__(self, xpos, ypos, velocity, acceleration):
        pygame.sprite.Sprite.__init__(self)

        self.image = mycar_image
        self.rect = self.image.get_rect()
        self.length = self.rect.height  # length and width are in pixels
        self.width = self.rect.width

        self.initial_xpos = DISPLAY_WIDTH / 2
        self.initial_ypos = DISPLAY_HEIGHT / 2
        self.xpos = 0
        self.ypos = 0
        self.rect.x = xpos
        self.rect.y = ypos
        self.v, self.acc = velocity, acceleration
        self.rotation = 0
        self.rad = 0

    def update(self):
        self.v += self.acc
        self.rad += (self.v / radius)
        self.rad = self.rad % (2 * math.pi)

        if self.v < 0:
            self.v = 0
            return

        self.xpos = self.initial_xpos + math.cos(self.rad) * radius
        self.ypos = self.initial_ypos + math.sin(self.rad) * radius
        self.rotation = 90 - math.degrees(self.rad)