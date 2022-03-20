import gym
from Ring_Road import RingRoad
from Ring_Road.metrics import Metrics


def evaluate():
    env_config = {
        "enable_render": False,
        "agent_type": "pi",
        "eval_mode": True,
        "algorithm": "pi"
    }

    env = RingRoad(env_config)
    obs = env.reset()

    met = Metrics(env)
    done = False

    while not done:
        obs, rew, done, info = env.step()
        met.step()
        print(env.action_steps)

    met.plot()
    env.close()
