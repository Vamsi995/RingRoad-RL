import math

import numpy as np

from Ring_Road.constants import RADIUS, CAR_LENGTH, RADIUS, AGENT_MAX_VELOCITY, TRACK_LENGTH


class StateExtractor:
    def __init__(self, env):
        self.env = env

    def gap_front(self, veh):
        if veh.front_vehicle.central_angle - veh.central_angle < 0:
            sf = (2 * math.pi - veh.central_angle + veh.front_vehicle.central_angle) * RADIUS - CAR_LENGTH
        else:
            sf = (veh.front_vehicle.central_angle - veh.central_angle) * RADIUS - CAR_LENGTH

        return sf

    def _calculate_distances(self, agent):
        if agent.front_vehicle.central_angle - agent.central_angle < 0:
            sf = (2 * math.pi - agent.central_angle + agent.front_vehicle.central_angle) * RADIUS
        else:
            sf = (agent.front_vehicle.central_angle - agent.central_angle) * RADIUS - CAR_LENGTH

        if agent.central_angle - agent.back_vehicle.central_angle < 0:
            sb = (2 * math.pi + agent.central_angle - agent.back_vehicle.central_angle) * RADIUS - CAR_LENGTH
        else:
            sb = (agent.central_angle - agent.back_vehicle.central_angle) * RADIUS - CAR_LENGTH

        if sf <= 0:
            sf = 0.00001
        if sb <= 0:
            sb = 0.00001

        return sf, sb

    def neighbour_states(self):
        feature_list = []
        for ag in self.env.agents:
            feature_list.append(ag.v / AGENT_MAX_VELOCITY)
            feature_list.append((ag.front_vehicle.v - ag.v) / AGENT_MAX_VELOCITY)
            slead, sback = self._calculate_distances(ag)
            feature_list.append(slead / TRACK_LENGTH)

        return np.array(feature_list)
