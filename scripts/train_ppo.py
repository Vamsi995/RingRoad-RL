from ray.rllib.agents import ppo

from Ring_Road.constants import MAX_EPISODE_LENGTH, WARMUP_STEPS
from experiment import Experiment

env_config = {
    "enable_render": False,
    "agent_type": "continuous",
    "eval_mode": True,
    "algorithm": "ppo",
    "time_steps": 1000000
}

config = ppo.DEFAULT_CONFIG.copy()
config["env_config"]["enable_render"] = False
config["env_config"]["agent_type"] = "continuous"
config["env_config"]["eval_mode"] = False
config["env_config"]["algorithm"] = env_config["algorithm"]
config["env"] = "ringroad-v1"
# gpu_count = 1
# num_gpus = 0.0001 # Driver GPU
# num_gpus_per_worker = (gpu_count - num_gpus) / num_workers
config["gamma"] = 0.999
config["use_gae"] = True
config["lambda"] = 0.97
config["kl_target"] = 0.02
config["num_sgd_iter"] = 10
config["num_gpus"] = 1
config["num_workers"] = 8
config["lr"] = 1e-6
config["horizon"] = MAX_EPISODE_LENGTH
config["model"]["fcnet_hiddens"] = [32, 32, 32]
# config["model"]["use_lstm"] = True
# config["model"]["lstm_cell_size"] = 32
# config["model"]["lstm_use_prev_action_reward"] = True
# config["model"]["lstm_use_prev_action"] = True
# config["model"]["max_seq_len"] = 32
# config["clip_actions"] = True
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
