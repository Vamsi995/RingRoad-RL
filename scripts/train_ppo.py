from ray.rllib.agents import ppo

from experiment import Experiment

env_config = {
    "enable_render": False,
    "agent_type": "discrete",
    "eval_mode": True,
    "algorithm": "ppo"
}

config = ppo.DEFAULT_CONFIG.copy()
config["env_config"]["enable_render"] = False
config["env_config"]["agent_type"] = "discrete"
config["env_config"]["eval_mode"] = False
config["env"] = "ringroad-v1"
config["num_gpus"] = 1
config["num_workers"] = 1
config["lr"] = 0.0001
config["horizon"] = 3000
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
