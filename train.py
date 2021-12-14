import gym
from DQN import learn
from baselines.common import models
from constants import DISCOUNT_FACTOR
from simulator import NoRenderEnv


def callback(lcl, _glb):
    # stop training if reward maintains a desired value
    is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 56700
    return is_solved


def train():
    env = NoRenderEnv()
    act = learn(
        env,
        seed=None,  # TODO: what?
        lr=1e-4,
        network=models.mlp(num_hidden=64, num_layers=3),
        total_timesteps=100000,
        buffer_size=100000,
        exploration_fraction=0.9,  # Percentage of time in which exploration has to be done
        exploration_final_eps=0.1,
        train_freq=4,
        batch_size=32,
        print_freq=10,
        checkpoint_freq=1000,
        checkpoint_path='/home/vamsi/Desktop/RingRoad',  # TODO: put in
        learning_starts=100,
        gamma=DISCOUNT_FACTOR,
        double_q=True,
        target_network_update_freq=1000,
        # callback=callback,  # TODO: put in
        load_path=None,  # initial checkpoint directory to be used
        inbuild_network=True,
        param_noise=True,
        plot_name='reward.png',
        prioritized_replay=True,
        policy_kwargs=dict(dueling=True)
    )
    print("Saving model to Model.pkl")
    act.save("Models/Model2.pkl")


if __name__ == '__main__':
    train()
