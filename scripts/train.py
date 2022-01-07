import gym
from stable_baselines import DQN
from stable_baselines.deepq.policies import FeedForwardPolicy

from Ring_Road.control.DQN import learn
from baselines.common import models
from Ring_Road.constants import DISCOUNT_FACTOR


class CustomDQNPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomDQNPolicy, self).__init__(*args, **kwargs,
                                              layers=[64, 64, 64],
                                              layer_norm=False,
                                              feature_extraction="mlp")


def train():
    env = gym.make("ringroad-v1", enable_render=False)

    model = DQN(
        env=env,
        policy=CustomDQNPolicy,
        learning_rate=1e-4,
        buffer_size=100000,
        exploration_fraction=0.9,
        exploration_final_eps=0.01,
        train_freq=6,
        batch_size=64,
        double_q=True,
        learning_starts=10000,
        prioritized_replay=True,
        target_network_update_freq=1000,
        param_noise=True,
        verbose=1,
        policy_kwargs=dict(dueling=True),
        tensorboard_log="/home/vamsi/Documents/GitHub/RingRoad-RL/logs/DQN/"
    )

    model.learn(total_timesteps=100000)
    model.save("Models/DQN")

    # print("Saving model to Model.pkl")
    # act.save("Models/Model2.pkl")
    env.close()


if __name__ == '__main__':
    train()
