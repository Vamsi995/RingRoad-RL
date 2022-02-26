import gym
from Ring_Road import RingRoad


def evaluate():
    env_config = {
        "enable_render": False,
        "agent_type": "idm",
        "eval_mode": True
    }
    env = RingRoad(env_config)

    obs = env.reset()
    done = False

    while not done:
        obs, rew, done, info = env.step()
        print(env.action_steps)
    env.close()
