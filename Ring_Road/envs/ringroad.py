import pygame
import gym


class RingRoad(gym.Env):

    def __init__(self):

        self.agents = pygame.sprite.Group()
        self.env_veh = pygame.sprite.Group()

        features_low = np.array([0, 0, 0, 0, 0], dtype=np.float64)
        features_high = np.array([20, 20, 20, 2500, 2500], dtype=np.float64)

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=features_low, high=features_high, dtype=np.float64)

        self.simulation_time = 0  # Simulation time
        self.action_steps = 0  # Actions performed
        self.done = False

        self.state = None
        self.reward = None
        self.discount_factor = DISCOUNT_FACTOR

        self.state_extractor = StateExtractor(self)
        self.viewer = Render(self)
        self.collision = False

    def _initialize_state(self):
        self.agents.empty()
        self.env_veh.empty()
        total_no = ENV_VEHICLES + AGENTS
        degree_spacing = 360 / total_no
        positions = np.arange(total_no) * degree_spacing
        vehicle_list = []

        if AGENTS > 0:
            agent_pos = np.random.randint(0, total_no, AGENTS)
        else:
            agent_pos = []

        for i in range(len(positions)):
            if i not in agent_pos:
                vehicle_list.append(
                    EnvVehicle(positions[i], np.random.randint(low=0, high=3), INITIAL_ACCELERATION, i, WIDTH, LENGTH))
            else:
                vehicle_list.append(
                    Agent(positions[i], np.random.randint(low=0, high=3), INITIAL_ACCELERATION, i, WIDTH, LENGTH))

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
                self.env_veh.add(cur_veh)
            else:
                self.agents.add(cur_veh)

    def _handle_collisions(self):
        for agent in self.agents:
            if self.state_extractor.gap_front(agent) <= 0:
                agent.crashed = True
                self.collision = True

    def reset(self):
        self.done = False
        self.simulation_time = 0
        self.action_steps = 0
        self._initialize_state()
        self.state = self.state_extractor.neighbour_states()
        return self.state

    def _simulate(self, action):
        frames = int(FPS // ACTION_FREQ)
        for frame in range(frames):
            if action is not None and self.simulation_time % frames == 0:
                for agent in self.agents:
                    agent.stored_action = action

            for agents in self.agents:
                agents.step()
            for env_veh in self.env_veh:
                env_veh.step()
            self._handle_collisions()
            self.simulation_time += 1

            if frame < frames - 1:
                self.render()

    def _reward(self):
        vav = self.state[2]
        vlead = self.state[0]
        vlag = self.state[1]
        sf = self.state[3]
        sb = self.state[4]

        reward = 0
        if vlead > 0 and vlag > 0:
            reward += 10
        else:
            reward -= 450

        if sf < 40:
            reward -= 450

        for agents in self.agents:
            if agents.crashed:
                reward -= 1000

        return reward

    def _is_done(self):
        if self.action_steps >= MAX_EPISODE_LENGTH or self.collision:
            return True

    def step(self, action):

        self.action_steps += 1
        self._simulate(action)

        obs = self.state_extractor.neighbour_states()
        reward = self._reward()
        terminal = self._is_done()
        # info = self._info(obs, action)
        info = {}
        return obs, reward, terminal, info

    def render(self, mode='human'):

        if self.viewer is None:
            self.viewer = Render(self)

        self.viewer.render()
