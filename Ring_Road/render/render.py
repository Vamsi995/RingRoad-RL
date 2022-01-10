import pygame
from Ring_Road.constants import DISPLAY_WIDTH, DISPLAY_HEIGHT, WHITE, BLACK, FPS, \
    RING_PIX_RADIUS, ROAD_PIX_WIDTH, up_arrow, down_arrow


class Render:

    def __init__(self, env):
        self.env = env
        pygame.init()
        pygame.display.set_caption("RingRoad Env")
        self.screen = pygame.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT))
        self.clock = pygame.time.Clock()

    def _create_background(self):
        self.screen.fill(WHITE)
        pygame.draw.circle(self.screen, BLACK, (DISPLAY_WIDTH // 2, DISPLAY_HEIGHT // 2), RING_PIX_RADIUS,
                           ROAD_PIX_WIDTH)

    def _rotate_image_display(self, image, angle, x, y):
        rotated_image = pygame.transform.rotate(image, angle)
        new_rect = rotated_image.get_rect(center=image.get_rect(center=(x, y)).center)
        self.screen.blit(rotated_image, new_rect)

    def _display_action(self):

        for agents in self.env.agents:
            if agents.stored_action == 0:
                self.screen.blit(up_arrow, [0, DISPLAY_HEIGHT - 80])
            else:
                self.screen.blit(down_arrow, [0, DISPLAY_HEIGHT - 80])

    def render(self):
        self._check_quit()
        self._create_background()

        for agent in self.env.agents:
            self._rotate_image_display(agent.image, agent.rotation, agent.xpos, agent.ypos)
        for env_veh in self.env.env_veh:
            self._rotate_image_display(env_veh.image, env_veh.rotation, env_veh.xpos, env_veh.ypos)

        self._display_action()

        pygame.display.update()
        self.clock.tick(FPS)

    def _check_quit(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
