import gym
from stable_baselines.common import make_vec_env
from stable_baselines.common.policies import FeedForwardPolicy
from stable_baselines import A2C

from Ring_Road.constants import DISCOUNT_FACTOR
import tensorflow as tf


class CustomPolicy(FeedForwardPolicy):

    def __init__(self, *args, **kwargs, ):
        super(CustomPolicy, self).__init__(*args, **kwargs,
                                           act_fun=tf.nn.relu,
                                           net_arch=[dict(pi=[64, 64, 64], vf=[64, 64, 64])],
                                           feature_extraction='mlp')


def train():
    env = make_vec_env("ringroad-v1", n_envs=4)
    # env = gym.make("ringroad-v1", enable_render=False)
    model = A2C(CustomPolicy, env, gamma=DISCOUNT_FACTOR, lr_schedule="linear", verbose=1, n_steps=32,
                tensorboard_log="logs/A2C", learning_rate=0.0001)
    model.learn(total_timesteps=200000)
    model.save("Models/ActorCritic2")
