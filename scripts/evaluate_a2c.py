from stable_baselines import A2C

from Ring_Road.metrics import Metrics
import gym


def main():
    env = gym.make("ringroad-v1", enable_render=True, agent_type="a2c")
    model = A2C.load("Models/A2C/ActorCritic2.zip")

    obs = env.reset()
    done = False
    met = Metrics(env)

    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        met.step()
        env.render()
    met.plot()
    env.close()
