from ray.rllib.agents import dqn

from experiment import Experiment

env_config = {
    "enable_render": False,
    "agent_type": "discrete",
    "eval_mode": True,
    "algorithm": "dqn"
}

config = dqn.DEFAULT_CONFIG.copy()
config["env_config"]["enable_render"] = False
config["env_config"]["agent_type"] = "discrete"
config["env_config"]["eval_mode"] = False
config["env"] = "ringroad-v1"
config["num_gpus"] = 1
config["num_workers"] = 1
config["lr"] = 0.0001
config["horizon"] = 3000
config['replay_buffer_config']['capacity'] = 100000
config['learning_starts'] = 1000
config['target_network_update_freq'] = 500
config['train_batch_size'] = 64
config['model']['fcnet_hiddens'] = [256, 256, 256]
config["evaluation_interval"] = 2
config["evaluation_duration"] = 20
config["framework"] = "torch"


def train():
    exp = Experiment(env_config, config)
    checkpoint_path, results = exp.train()
    print(exp.evaluate(checkpoint_path))


def evaluate(path):
    exp = Experiment(env_config, config)
    episode_reward = exp.evaluate(path)
    print(episode_reward)
