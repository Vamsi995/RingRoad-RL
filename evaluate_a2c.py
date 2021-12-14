import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import A2C

from metrics import Metrics
from simulator import RenderEnv

env = RenderEnv()
model = A2C.load("Models/ActorCritic2.zip")

obs = env.reset()
done = False
met = Metrics(env)
met.register_cars()
t = 0
while not done:
    action, _states = model.predict(obs)
    met.store_v(t)
    met.store_xy(t)
    met.running_mean_vel(t)
    obs, rewards, done, info = env.step(action)
    t += 1


met.plot_positions()
met.plot_velocities()