from gym.envs.registration import register

register(
    id='ringroad-v0',
    entry_point='Ring_Road.envs.ringroad:RingRoad',
)
