from stable_baselines.common.policies import FeedForwardPolicy
from stable_baselines import A2C

from Ring_Road.constants import DISCOUNT_FACTOR
from simulator import NoRenderEnv
import tensorflow as tf

class CustomPolicy(FeedForwardPolicy):

    def __init__(self, *args, **kwargs,):
        super(CustomPolicy, self).__init__(*args, **kwargs,
                                           act_fun=tf.nn.relu,
                                           net_arch=[dict(pi=[64, 64, 64], vf=[64, 64, 64])],
                                           feature_extraction='mlp')

def train():
    """
    Train A2C model for atari environment, for testing purposes
    :param env_id: (str) Environment ID
    :param num_timesteps: (int) The total number of samples
    :param seed: (int) The initial seed for training
    :param policy: (A2CPolicy) The policy model to use (MLP, CNN, LSTM, ...)
    :param lr_schedule: (str) The type of scheduler for the learning rate update ('linear', 'constant',
                                 'double_linear_con', 'middle_drop' or 'double_middle_drop')
    :param num_env: (int) The number of environments
    """

    env = NoRenderEnv()
    # env = VecFrameStack(env, 4)
    # policy_fn = MlpPolicy(net_arch=[dict(pi=[128, 128, 128], vf=[128, 128, 128])])
    model = A2C(CustomPolicy, env, gamma=DISCOUNT_FACTOR, lr_schedule="linear", verbose=1, n_steps=128, learning_rate=0.0001)
    model.learn(total_timesteps=200000)
    model.save("Models/ActorCritic2")


if __name__ == '__main__':
    train()
