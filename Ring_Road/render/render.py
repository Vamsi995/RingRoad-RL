import pygame
from pygame import gfxdraw
from Ring_Road.constants import DISPLAY_WIDTH, DISPLAY_HEIGHT, WHITE, BLACK, RING_RADIUS, ROAD_WIDTH, FPS


class Render():

    def __init__(self, env):
        self.env = env
        pygame.init()
        pygame.display.set_caption("RingRoad Env")
        self.screen = pygame.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT))
        self.clock = pygame.time.Clock()

    def _create_background(self):
        self.screen.fill(WHITE)
        gfxdraw.aacircle(self.screen, int(DISPLAY_WIDTH / 2), int(DISPLAY_HEIGHT / 2), RING_RADIUS, BLACK)
        gfxdraw.aacircle(self.screen, int(DISPLAY_WIDTH/2), int(DISPLAY_HEIGHT/2), RING_RADIUS-ROAD_WIDTH, BLACK)

    def _rotate_image_display(self, image, angle, x, y):
        rotated_image = pygame.transform.rotate(image, angle)
        new_rect = rotated_image.get_rect(center=image.get_rect(center=(x, y)).center)
        self.screen.blit(rotated_image, new_rect)

    def render(self):
        self._check_quit()
        self._create_background()

        for agent in self.env.agents:
            # self._rotate_image_display(agent, agent.rotation, agent.xpos, agent.ypos)
            rotated_image, new_rect = agent.render(color=(255,0,0))
            self.screen.blit(rotated_image, new_rect)
        for env_veh in self.env.env_veh:
            # self._rotate_image_display(env_veh, env_veh.rotation, env_veh.xpos, env_veh.ypos)
            rotated_image, new_rect = env_veh.render(color=(0, 0, 255))
            self.screen.blit(rotated_image, new_rect)

        pygame.display.update()
        self.clock.tick(FPS)

    def _check_quit(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
