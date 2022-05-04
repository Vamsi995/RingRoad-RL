import math

import numpy as np

from Ring_Road.constants import RADIUS, CAR_LENGTH, RADIUS, AGENT_MAX_VELOCITY, TRACK_LENGTH, DELTA_T


class StateExtractor:
    def __init__(self, env):
        self.env = env
        self.delay = 0

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
        feature_dict = {}
        max_speed = 15
        for ag_id, agent in self.env.agents.items():
            feat_list = [agent.v / max_speed, (agent.front_vehicle.v - agent.v) / max_speed]
            s_lead, s_back = self._calculate_distances(agent)
            feat_list.append(s_lead / TRACK_LENGTH)
            feature_dict[ag_id] = np.array(feat_list)
        return feature_dict

    def get_safe_action_instantaneous(self, action, front_veh, agent):

        this_vel = front_veh.v
        next_vel = this_vel + action * DELTA_T
        h = agent.gap_front()

        if next_vel > 0:
            # the second and third terms cover (conservatively) the extra
            # distance the vehicle will cover before it fully decelerates
            if h < DELTA_T * next_vel + this_vel * 1e-3 + \
                    0.5 * this_vel * DELTA_T:
                # if the vehicle will crash into the vehicle ahead of it in the
                # next time step (assuming the vehicle ahead of it is not
                # moving), then stop immediately
                # if self.display_warnings:
                #     print(
                #         "=====================================\n"
                #         "Vehicle {} is about to crash. Instantaneous acceleration "
                #         "clipping applied.\n"
                #         "=====================================".format(self.veh_id))

                return -this_vel / DELTA_T
            else:
                # if the vehicle is not in danger of crashing, continue with
                # the requested action
                return action
        else:
            return action

    def get_safe_velocity_action(self, action, front_veh, agent):

        h = agent.gap_front()
        dv = front_veh.v - agent.v

        safe_velocity = 2 * h / DELTA_T + dv - agent.v * (2 * self.delay)

        this_vel = agent.v
        sim_step = DELTA_T

        if this_vel + action * sim_step > safe_velocity:
            if safe_velocity > 0:
                return (safe_velocity - this_vel) / sim_step
            else:
                return -this_vel / sim_step
        else:
            return action

    def failsafe_action(self, accel, front_veh, agent):
        if accel == None:
            return None

        accel = self.get_safe_action_instantaneous(accel, front_veh, agent)
        accel = self.get_safe_velocity_action(accel, front_veh, agent)
        accel = np.clip(accel, -100, 1)
        return accel

    def get_average_vel(self):

        vel = []
        for ag in self.env.env_veh:
            vel.append(ag.v)
        for ag_id, agent in self.env.agents.items():
            vel.append(agent.v)
        return sum(vel) / len(vel)
