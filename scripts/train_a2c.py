import gym
from stable_baselines.common import make_vec_env
from stable_baselines.common.policies import FeedForwardPolicy
from stable_baselines import A2C

from Ring_Road.constants import DISCOUNT_FACTOR
import tensorflow as tf

from Ring_Road.envs.ringroad import RingRoad


class CustomPolicy(FeedForwardPolicy):

    def __init__(self, *args, **kwargs, ):
        super(CustomPolicy, self).__init__(*args, **kwargs,
                                           act_fun=tf.nn.relu,
                                           net_arch=[dict(pi=[256, 256], vf=[256, 256])],
                                           feature_extraction='mlp')


def train():
    # env = make_vec_env(RingRoad, n_envs=4, env_kwargs={"agent_type": "a2c"})
    env = gym.make("ringroad-v1", agent_type="a2c", enable_render=False)
    model = A2C(CustomPolicy, env, gamma=DISCOUNT_FACTOR, lr_schedule="linear", verbose=1, n_steps=2,
                tensorboard_log="logs/A2C", learning_rate=0.0001)
    model.learn(total_timesteps=600000)
    model.save("Models/A2C/ActorCritic5")
