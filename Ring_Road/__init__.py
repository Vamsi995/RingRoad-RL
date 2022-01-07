from gym.envs.registration import register

register(
    id='ringroad-v1',
    entry_point='Ring_Road.envs.ringroad:RingRoad',
)