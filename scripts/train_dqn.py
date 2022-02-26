import gym
import ray
from ray import tune
from ray.rllib.agents import dqn


class Experiment:

    def __init__(self):
        ray.shutdown()
        ray.init()
        self.agent = None
        self.env = gym.make("CartPole-v0")
        self.config = dqn.DEFAULT_CONFIG.copy()
        self.config["env"] = "CartPole-v0"
        self.config["num_gpus"] = 1
        self.config["num_workers"] = 1
        self.config["lr"] = 0.0001
        # self.config["horizon"] = 3000
        self.config['replay_buffer_config']['capacity'] = 50000
        self.config['learning_starts'] = 1000
        self.config['target_network_update_freq'] = 500
        self.config['train_batch_size'] = 32
        self.config['model']['fcnet_hiddens'] = [256, 256, 256]
        self.config["evaluation_interval"] = 2
        self.config["evaluation_duration"] = 20
        self.config["framework"] = "torch"

    def train(self):
        results = tune.run(dqn.DQNTrainer,
                           verbose=1,
                           config=self.config,
                           stop={"timesteps_total": 200000},
                           local_dir="Models/DQN/",
                           checkpoint_freq=2
                           )
        checkpoint_path = results.get_last_checkpoint()
        print("Checkpoint path:", checkpoint_path)
        return checkpoint_path, results

    def load(self, path):
        self.agent = dqn.DQNTrainer(config=self.config)
        self.agent.restore(path)

    def evaluate(self):
        """Test trained agent for a single episode. Return the episode reward"""
        # instantiate env class
        env = self.env

        # run until episode ends
        episode_reward = 0
        done = False
        obs = env.reset()
        while not done:
            action = self.agent.compute_single_action(obs)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            env.render()

        return episode_reward


def train():
    exp = Experiment()
    checkpoint_path, results = exp.train()
    exp.load(checkpoint_path)
    print(exp.evaluate())