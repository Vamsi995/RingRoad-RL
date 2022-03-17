import math
import pygame

import numpy as np

from Ring_Road.constants import agent_car_image, DISPLAY_HEIGHT, RADIUS_PIX, CAR_PIX_LENGTH, DELTA_T, \
    AGENT_MAX_VELOCITY, \
    IDM_DELTA, v0, \
    T, s0, a, b, env_car_image, DISPLAY_WIDTH, PIXEL_CONVERSION, RADIUS, CAR_LENGTH, ENV_VEHICLES, AGENTS


class Car(pygame.sprite.Sprite):

    def __init__(self, rad, velocity, acceleration, id):
        pygame.sprite.Sprite.__init__(self)
        self.image = agent_car_image
        self.rect = self.image.get_rect()
        self.length = self.rect.height
        self.width = self.rect.width
        self.id = id
        self.initial_xpos = DISPLAY_WIDTH / 2
        self.initial_ypos = DISPLAY_HEIGHT / 2
        self.v, self.acc = velocity, acceleration
        self.central_angle = math.radians(rad)
        self.rotation = 90 - math.degrees(self.central_angle)
        self.xpos = self.initial_xpos + math.cos(self.central_angle) * RADIUS_PIX
        self.ypos = self.initial_ypos + math.sin(self.central_angle) * RADIUS_PIX
        self.distance_covered = 0
        self.front_vehicle = {}
        self.back_vehicle = {}

    def update_positions(self):
        self.central_angle += (self.v * DELTA_T / RADIUS)
        self.central_angle = self.central_angle % (2 * math.pi)
        self.xpos = self.initial_xpos + math.cos(self.central_angle) * RADIUS_PIX
        self.ypos = self.initial_ypos + math.sin(self.central_angle) * RADIUS_PIX
        self.rect.width = self.image.get_width()
        self.rect.height = self.image.get_height()
        self.rotation = 90 - math.degrees(self.central_angle)

    def gap_front(self):
        if self.front_vehicle.central_angle - self.central_angle < 0:
            s = (2 * math.pi - self.central_angle + self.front_vehicle.central_angle) * RADIUS - CAR_LENGTH
        else:
            s = (self.front_vehicle.central_angle - self.central_angle) * RADIUS - CAR_LENGTH

        return s

    def idm_control(self):
        delta_v = self.v - self.front_vehicle.v

        s = self.gap_front()
        if s <= 0:
            s = 0.0001

        s_star = s0 + max(0, (self.v * T) + ((self.v * delta_v) / (2 * np.power(a * b, 0.5))))
        self.acc = a * (1 - np.power(self.v / v0, IDM_DELTA) - np.power(s_star / s, 2)) + np.random.normal(0, 0.2)
        old_velocity = self.v
        self.v = max(0, min((old_velocity) + (self.acc * DELTA_T), v0))
        self.acc = (self.v - old_velocity) / DELTA_T  # adjustment to acc due to clamping of velocity

        return None


class Agent(Car):
    def __init__(self, rad, velocity, acceleration, id, agent_type):
        super(Agent, self).__init__(rad, velocity, acceleration, id)
        self.image = agent_car_image
        self.rect = self.image.get_rect()
        self.length = self.rect.height
        self.width = self.rect.width
        self.rect.x = self.xpos
        self.rect.y = self.ypos
        self.stored_action = None
        self.crashed = False
        self.agent_type = agent_type

        self.agent_vel_history = []
        self.history_len = 380
        self.desired_vel = 0
        self.v_cmd = 0

    def _follower_stopper(self, desired_velocity, curvatures, initial_x):
        v_lead = self.front_vehicle.v
        delta_v = min(v_lead - self.v, 0)

        self.desired_vel = desired_velocity

        delta_x1 = initial_x[0] + (1 / (2 * curvatures[0])) * (delta_v ** 2)
        delta_x2 = initial_x[1] + (1 / (2 * curvatures[1])) * (delta_v ** 2)
        delta_x3 = initial_x[2] + (1 / (2 * curvatures[2])) * (delta_v ** 2)
        v = min(max(v_lead, 0), self.desired_vel)
        s = self.gap_front()
        v = v_lead


        v_cmd = 0
        if s <= delta_x1:
            v_cmd = 0
            print("delta1: {}".format(v_cmd))
        elif s <= delta_x2:
            v_cmd = v * ((s - delta_x1) / (delta_x2 - delta_x1))
            print("delta2: {}".format(v_cmd))
        elif s <= delta_x3:
            v_cmd = v + ((self.desired_vel - v) * ((s - delta_x2) / (delta_x3 - delta_x2)))
            print("delta3: {}".format(v_cmd))
        elif s > delta_x3:
            v_cmd = self.desired_vel
        print("Gap Front: {}, Delta_x1: {}, Delta_x2: {}, Delta_x3: {}, v_cmd: {}, v: {}".format(s, delta_x1, delta_x2, delta_x3, v_cmd, v))

        self.acc = (v_cmd - self.v) / DELTA_T
        return self.acc

    def _pi_controller(self, v_catch=1, gl=7, gu=30):

        self.agent_vel_history.append(self.v)
        if len(self.agent_vel_history) == self.history_len:
            self.agent_vel_history.pop(0)

        self.desired_vel = sum(self.agent_vel_history) / len(self.agent_vel_history)

        v_lead = self.front_vehicle.v
        s = self.gap_front()
        delta_v = v_lead - self.v
        safety = max(2 * delta_v, 4)

        v_target = self.desired_vel + v_catch * min(max((s - gl) / (gu - gl), 0), 1)
        alpha = min(max((s - safety) / 2, 0), 1)
        beta = 1 - (alpha / 2)

        self.v_cmd = beta * (alpha * v_target + (1 - alpha) * v_lead) + (1 - beta) * self.v_cmd
        self.acc = (self.v_cmd - self.v) / DELTA_T

        self.acc = min(self.acc, 1)
        return self.acc

    def _continuous(self):
        self.acc = self.stored_action[0]

        return self.acc

    def _discrete(self):
        if self.stored_action == 0:
            self.acc = 1
        elif self.stored_action == 1:
            self.acc = -2

        return self.acc

    def _manual_control(self):
        if self.stored_action == 0:
            self.acc = 1
        else:
            self.acc = -1

        return self.acc

    def _run_control(self, avg_vel=None):
        accel_action = None
        if self.agent_type == "idm":
            accel_action = self.idm_control()
        elif self.agent_type == "discrete":
            accel_action = self._discrete()
        elif self.agent_type == "continuous":
            accel_action = self._continuous()
        elif self.agent_type == "fs":
            accel_action = self._follower_stopper(avg_vel, [1.5, 1, 0.5], [4.5, 5.25, 6])
        elif self.agent_type == "pi":
            accel_action = self._pi_controller()
        elif self.agent_type == "man":
            accel_action = self._manual_control()

        return accel_action

    def _update_vel(self, accel_action):
        if accel_action == None:
            return
        self.v = max(0, min(self.v + (accel_action * DELTA_T), AGENT_MAX_VELOCITY))
        # print("Updated Velocity: {}".format(self.v))

    def step(self, eval_mode, action_steps, agent_type, state_extractor, avg_vel):
        if eval_mode:
            if action_steps > 3000:
                self.agent_type = agent_type
            else:
                self.agent_type = "idm"

        accel_action = self._run_control(avg_vel)
        accel_action = state_extractor.failsafe_action(accel_action)
        self._update_vel(accel_action)
        self.update_positions()


class EnvVehicle(Car):
    def __init__(self, rad, velocity, acceleration, id):
        super(EnvVehicle, self).__init__(rad, velocity, acceleration, id)
        self.image = env_car_image
        self.rect = self.image.get_rect()
        self.length = self.rect.height
        self.width = self.rect.width
        self.rect.x = self.xpos
        self.rect.y = self.ypos

    def step(self):
        self.idm_control()
        self.update_positions()
