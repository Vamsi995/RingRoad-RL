import pygame
from constants import DISPLAY_WIDTH, DISPLAY_HEIGHT, white, black, ring_radius, road_width, velocity, radius, \
    acceleration
from vehicle import Car


class Env:
    def __init__(self):
        """Constructor Method
        """
        self.screen = pygame.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT))
        pygame.display.set_caption("Ring Road Simulator")
        self.clock = pygame.time.Clock()
        self.car = Car(500 - 40, 725 - 40, velocity, acceleration)
        pygame.init()
        self.create_background()

    def create_background(self):
        self.screen.fill(white)
        pygame.draw.circle(self.screen, black, (DISPLAY_WIDTH / 2, DISPLAY_HEIGHT / 2), ring_radius, road_width)

    def rotate_image(self, image, angle, x, y):
        rotated_image = pygame.transform.rotate(image, angle)
        new_rect = rotated_image.get_rect(center=self.car.image.get_rect(center=(x, y)).center)
        return rotated_image, new_rect

    def reset(self):
        pass

    def step(self):
        self.create_background()
        self.car.update()
        self.render()

    def render(self):

        self.check_quit()
        rotated_image, new_rect = self.rotate_image(self.car.image, self.car.rotation, self.car.xpos, self.car.ypos)
        self.screen.blit(rotated_image, new_rect)

        pygame.display.update()
        self.clock.tick(30)

    def check_quit(self):

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()


env = Env()

while True:
    env.step()
    # env.render()
