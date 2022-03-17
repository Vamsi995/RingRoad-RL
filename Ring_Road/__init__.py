from gym.envs.registration import register

from Ring_Road.envs.multiagent_ringroad import MultiAgentRingRoad
from Ring_Road.envs.ringroad import RingRoad
from ray.tune.registry import register_env

#
# register(
#     id='ringroad-v1',
#     entry_point='Ring_Road.envs.ringroad:RingRoad'
# )


def env_creator(env_config):
    return RingRoad(env_config)  # return an env instance

def multiagent_env_creator(env_config):
    return MultiAgentRingRoad(env_config)  # return an env instance

register_env("ringroad-v1", env_creator)
register_env("multiagent_ringroad-v1", multiagent_env_creator)
