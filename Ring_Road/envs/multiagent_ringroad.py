from typing import Optional

import numpy as np
from gym import spaces
from ray.rllib import MultiAgentEnv

from Ring_Road.constants import DISCOUNT_FACTOR, INITIAL_ACCELERATION, AGENTS, ENV_VEHICLES
from Ring_Road.render.render import Render
from Ring_Road.vehicle.state import StateExtractor
from Ring_Road.vehicle.vehicle import EnvVehicle, Agent


class MultiAgentRingRoad(MultiAgentEnv):

    def __init__(self, env_config):
        self.enable_render = env_config["enable_render"]
        self.agent_type = env_config["agent_type"]
        self.eval_mode = env_config["eval_mode"]

        self.agents = {}
        self.env_veh = []

        features_low = np.array([0, -1, 0], dtype=np.float64)
        features_high = np.array([1, 1, 1], dtype=np.float64)

        self.observation_space = spaces.Box(low=features_low, high=features_high, dtype=np.float64)

        if self.agent_type == "continuous":
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float64)
        else:
            self.action_space = spaces.Discrete(2)

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

        agent_counter = 0

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
                self.agents[agent_counter] = cur_veh
                agent_counter += 1

    def clip_actions(self, action):

        for ag_id, action in action.items():

            if isinstance(self.action_space, spaces.Box) and action is not None:
                lb, ub = self.action_space.low, self.action_space.high
                scaled_action = lb + (action[0] + 1.) * 0.5 * (ub - lb)
                scaled_action = np.array(np.clip(scaled_action, lb, ub))
            else:
                scaled_action = action

        return scaled_action

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
            env_vehicles = np.random.randint(15, 24)
            self._initialize_state(env_vehicles)
            self._warmup_steps()
            self._set_agent_type(self.agent_type)

        self.state = self.state_extractor.neighbour_states()
        return self.state

    def step(self, action=None):

        self.action_steps += 1
        scaled_action = self.clip_actions(action)
        self._simulate(scaled_action)
        self.state = self.state_extractor.neighbour_states()
        reward = self._reward(scaled_action)
        terminal = self._is_done()
        info = {"action": scaled_action}
        return self.state, reward, terminal, info

    def render(self, mode='human'):

        if not self.enable_render:
            return

        if self.viewer is None:
            self.viewer = Render(self)

        self.viewer.render()
