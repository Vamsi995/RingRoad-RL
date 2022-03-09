from typing import Optional

import numpy as np
import gym
from gym import spaces
from gym.utils import seeding

from Ring_Road.constants import DISCOUNT_FACTOR, ENV_VEHICLES, AGENTS, FPS, MAX_EPISODE_LENGTH, ACTION_FREQ, \
    INITIAL_ACCELERATION, AGENT_MAX_VELOCITY, REWARD_ALPHA, WARMUP_STEPS, EVAL_EPISODE_LENGTH
from Ring_Road.render.render import Render
from Ring_Road.vehicle.state import StateExtractor
from Ring_Road.vehicle.vehicle import EnvVehicle, Agent


class RingRoad(gym.Env):

    def __init__(self, env_config):

        self.enable_render = env_config["enable_render"]
        self.agent_type = env_config["agent_type"]
        self.eval_mode = env_config["eval_mode"]
        self.algorithm = env_config["algorithm"]

        self.agents = []
        self.env_veh = []

        features_low = np.array([0, -1, 0], dtype=np.float64)
        features_high = np.array([1, 1, 1], dtype=np.float64)

        if self.agent_type == "continuous":
            self.action_space = spaces.Box(low=np.array([-1]), high=np.array([1]), dtype=np.float64)
        else:
            self.action_space = spaces.Discrete(2)

        self.observation_space = spaces.Box(low=features_low, high=features_high, dtype=np.float64)

        self.simulation_time = 0  # Simulation time
        self.action_steps = 0  # Actions performed
        self.done = False

        self.state = None
        self.reward = None
        self.discount_factor = DISCOUNT_FACTOR

        self.crashed_state = None
        self.state_extractor = StateExtractor(self)

        if self.enable_render:
            self.viewer = Render(self)
        self.collision = False
        np.random.seed(42)

    def _initialize_state(self, env_vehicles=ENV_VEHICLES):

        total_no = env_vehicles + AGENTS
        degree_spacing = 360 / total_no
        positions = np.arange(total_no) * degree_spacing
        vehicle_list = []

        if AGENTS > 0:
            agent_pos = np.random.randint(0, total_no, AGENTS)
        else:
            agent_pos = []

        for i in range(len(positions)):
            if i not in agent_pos:
                vehicle_list.append(EnvVehicle(positions[i], 0, INITIAL_ACCELERATION, i))
            else:
                vehicle_list.append(Agent(positions[i], 0, INITIAL_ACCELERATION, i, self.agent_type))

        for i in range(len(vehicle_list)):
            cur_veh = vehicle_list[i]
            front_vehicle = vehicle_list[(i + 1) % len(vehicle_list)]
            if i - 1 < 0:
                back_vehicle = vehicle_list[len(vehicle_list) - 1]
            else:
                back_vehicle = vehicle_list[i - 1]
            cur_veh.front_vehicle = front_vehicle
            cur_veh.back_vehicle = back_vehicle
            if isinstance(cur_veh, EnvVehicle):
                self.env_veh.append(cur_veh)
            else:
                self.agents.append(cur_veh)

    def _handle_collisions(self):
        for agent in self.agents:
            if self.state_extractor.gap_front(agent) <= 0:
                agent.crashed = True
                self.collision = True
                self.crashed_state = self.state_extractor.neighbour_states()
            else:
                self.collision = False
                agent.crashed = False

    def _linear_map(self, value, leftMax, leftMin, rightMax, rightMin):
        leftSpan = leftMax - leftMin
        rightSpan = rightMax - rightMin

        valueScaled = float(value - leftMin) / float(leftSpan)
        return rightMin + (valueScaled * rightSpan)

    def _simulate(self, action):
        frames = int(FPS // ACTION_FREQ)
        for frame in range(frames):
            if action is not None and self.simulation_time % frames == 0:
                for agent in self.agents:
                    agent.stored_action = action

            for agents in self.agents:
                agents.step(self.eval_mode, self.action_steps, self.agent_type)
            for env_veh in self.env_veh:
                env_veh.step()
            self._handle_collisions()
            self.simulation_time += 1

            if frame < frames - 1:
                self.render()

            if self.collision:
                break

    def _get_average_vel(self):

        vel = []
        for ag in self.env_veh:
            vel.append(ag.v)
        for ag in self.agents:
            vel.append(ag.v)
        return sum(vel) / len(vel)

    def _reward(self, action):

        if action is None:
            return 0

        if self.collision:
            return 0.

        # reward average velocity
        eta_2 = 4.
        reward = eta_2 * self._get_average_vel() / 20

        # punish accelerations (should lead to reduced stop-and-go waves)
        eta = 5  # 0.25
        mean_actions = np.mean(np.abs(np.array(action)))
        accel_threshold = 0

        if mean_actions > accel_threshold:
            reward += eta * (accel_threshold - mean_actions)

        return float(reward)

    def _is_done(self):
        if self.eval_mode:
            if self.action_steps >= EVAL_EPISODE_LENGTH or self.collision:
                return True
            else:
                return False
        else:
            if self.action_steps >= MAX_EPISODE_LENGTH or self.collision:
                return True
            else:
                return False

    def _warmup_steps(self):
        self._set_agent_type("idm")
        for i in range(WARMUP_STEPS):
            self.step(None)

    def _destroy(self):
        self.agents.clear()
        self.env_veh.clear()

    def _set_agent_type(self, agent_type):
        for ag in self.agents:
            ag.agent_type = agent_type

    def reset(self, *, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None):

        # super().reset(seed=seed)
        self._destroy()
        self.done = False
        self.simulation_time = 0
        self.action_steps = 0
        self.collision = False

        if self.eval_mode:
            self._initialize_state()
        else:
            env_vehicles = np.random.randint(8, 22)
            self._initialize_state(env_vehicles)
            self._warmup_steps()
            self._set_agent_type(self.agent_type)

        self.state = self.state_extractor.neighbour_states()
        return self.state

    def step(self, action=None):

        self.action_steps += 1
        self._simulate(action)

        self.state = self.state_extractor.neighbour_states()
        reward = self._reward(action)
        terminal = self._is_done()
        info = {}
        return self.state, reward, terminal, info

    def render(self, mode='human'):

        if not self.enable_render:
            return

        if self.viewer is None:
            self.viewer = Render(self)

        self.viewer.render()
