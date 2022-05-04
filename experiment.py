import collections
import random

import gym
import numpy as np
import ray
from gym.spaces import Tuple, Dict
from ray import tune
from ray.rllib.agents import dqn, ppo, qmix
from ray.rllib.evaluate import DefaultMapping

from ray.rllib.examples.models.centralized_critic_models import TorchCentralizedCriticModel, CentralizedCriticModel, \
    YetAnotherTorchCentralizedCriticModel, YetAnotherCentralizedCriticModel
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.utils.spaces.space_utils import flatten_to_single_ndarray
from ray.tune import register_env

from Ring_Road import RingRoad, MultiAgentRingRoad
from Ring_Road.constants import AGENTS
from Ring_Road.metrics import Metrics
from scripts.centralized_critic import FillInActions, central_critic_observer

class Experiment:

    def __init__(self, env_config, config):
        ray.shutdown()
        ray.init()
        self.algorithm = env_config["algorithm"]
        self.time_steps = env_config["time_steps"]
        self.agent = None
        self.config = config
        self.env = MultiAgentRingRoad(env_config)

    def update_multiagent_config(self):
        def gen_policy(i):
            config = {"gamma": 0.99}
            return PolicySpec(config=config)

        policies = {"policy_{}".format(i): gen_policy(i) for i in range(AGENTS)}
        policy_ids = list(policies.keys())

        def policy_mapping_fn(agent_id, **kwargs):
            pol_id = random.choice(policy_ids)
            return pol_id

        self.config["multiagent"]["policies"] = policies
        self.config["multiagent"]["policy_mapping_fn"] = policy_mapping_fn

    def train_multiple_policy(self):
        self.update_multiagent_config()
        global results

        save_path = ""
        if self.config["model"]["use_lstm"]:
            save_path = "Models/PPO/MultiAgent/NonSharedPolicy/LSTM/"
        else:
            save_path = "Models/PPO/MultiAgent/NonSharedPolicy/"

        if self.algorithm == "ppo":
            results = tune.run(ppo.PPOTrainer,
                               verbose=1,
                               config=self.config,
                               stop={"timesteps_total": self.time_steps},
                               local_dir=save_path,
                               checkpoint_at_end=True
                               )
            checkpoint_path = results.get_last_checkpoint()
            print("Checkpoint path:", checkpoint_path)
            return checkpoint_path, results

    def evaluate_multiple_policy(self, path):
        self.config["env_config"]["eval_mode"] = True
        self.config["env_config"]["enable_render"] = True
        self.config["num_workers"] = 0

        self.update_multiagent_config()
        if self.algorithm == "dqn":
            self.agent = dqn.DQNTrainer(config=self.config)
        elif self.algorithm == "ppo":
            self.agent = ppo.PPOTrainer(config=self.config)


        self.agent.restore(path)

        env = self.env

        # run until episode ends
        episode_reward = 0
        done = {"__all__": False}
        obs = env.reset()

        met = Metrics(env)

        mapping_cache = {}  # in case policy_agent_mapping is stochastic
        agent_states = DefaultMapping(
            lambda agent_id: [np.zeros([self.config["model"]["lstm_cell_size"]], np.float32) for _ in range(2)]
        )
        prev_actions = DefaultMapping(
            lambda agent_id: flatten_to_single_ndarray(self.env.action_space.sample())
        )

        use_lstm = {"policy_{}".format(p): len(s) > 0 for p, s in obs.items()}
        policy_agent_mapping = self.agent.config["multiagent"]["policy_mapping_fn"]

        prev_rewards = collections.defaultdict(lambda: 0.0)
        reward_total = 0.0
        while not done["__all__"]:

            multi_obs = obs
            action_dict = {}
            action = None
            for agent_id, a_obs in multi_obs.items():
                if a_obs is not None:
                    policy_id = mapping_cache.setdefault(
                        agent_id, policy_agent_mapping(agent_id)
                    )
                    p_use_lstm = use_lstm[policy_id]
                    if p_use_lstm:
                        a_action, p_state, _ = self.agent.compute_single_action(
                            a_obs,
                            state=agent_states[agent_id],
                            prev_action=prev_actions[agent_id],
                            prev_reward=prev_rewards[agent_id],
                            policy_id=policy_id,
                        )
                        agent_states[agent_id] = p_state
                    else:
                        a_action = self.agent.compute_single_action(
                            a_obs,
                            prev_action=prev_actions[agent_id],
                            prev_reward=prev_rewards[agent_id],
                            policy_id=policy_id,
                        )
                    a_action = flatten_to_single_ndarray(a_action)
                    action_dict[agent_id] = a_action
                    prev_actions[agent_id] = a_action
                action = action_dict

            next_obs, reward, done, info = env.step(action)

            for agent_id, r in reward.items():
                prev_rewards[agent_id] = r

            reward_total += sum(r for r in reward.values() if r is not None)
            obs = next_obs
            met.step()

            print(env.action_steps, action)
            # env.render()
            episode_reward += reward_total
        met.plot(self.config)
        return episode_reward

    def train(self):

        save_path = ""
        if self.config["model"]["use_lstm"]:
            save_path = "Models/PPO/MultiAgent/SharedPolicy/LSTM/"
        else:
            save_path = "Models/PPO/MultiAgent/SharedPolicy/"

        global results
        if self.algorithm == "dqn":
            results = tune.run(dqn.DQNTrainer,
                               verbose=1,
                               config=self.config,
                               stop={"timesteps_total": self.time_steps},
                               # restore="/home/vamsi/Documents/GitHub/RingRoad-RL/Models/DQN/DQNTrainer_2022-02-26_18-33-46/DQNTrainer_ringroad-v1_88476_00000_0_2022-02-26_18-33-46/checkpoint_000096/checkpoint-96",
                               local_dir="Models/DQN/",
                               checkpoint_at_end=True,
                               checkpoint_freq=20
                               )
            checkpoint_path = results.get_last_checkpoint()
            print("Checkpoint path:", checkpoint_path)
            return checkpoint_path, results

        elif self.algorithm == "ppo":
            results = tune.run(ppo.PPOTrainer,
                               verbose=1,
                               config=self.config,
                               stop={"timesteps_total": self.time_steps},
                               local_dir=save_path,
                               checkpoint_at_end=True
                               )
        checkpoint_path = results.get_last_checkpoint()
        print("Checkpoint path:", checkpoint_path)
        return checkpoint_path, results

    def evaluate(self, path):
        """Test trained agent for a single episode. Return the episode reward"""
        # instantiate env class
        self.config["env_config"]["eval_mode"] = True
        self.config["env_config"]["enable_render"] = True
        self.config["num_workers"] = 0

        if self.algorithm == "dqn":
            self.agent = dqn.DQNTrainer(config=self.config)
        elif self.algorithm == "ppo":
            self.agent = ppo.PPOTrainer(config=self.config)

        self.agent.restore(path)

        env = self.env

        # run until episode ends
        episode_reward = 0
        done = {"__all__": False}
        obs = env.reset()

        met = Metrics(env)
        while not done["__all__"]:
            action = self.agent.compute_actions(obs)
            obs, reward, done, info = env.step(action)
            met.step()
            print(env.action_steps, reward)
            env.render()
        met.plot(self.config)
        return episode_reward

    def train_centralized_critic(self):
        """Train centralized critic"""

        ModelCatalog.register_custom_model(
            "cc_model",
            YetAnotherTorchCentralizedCriticModel
            if self.config["framework"] == "torch"
            else YetAnotherCentralizedCriticModel,
        )

        action_space = self.env.action_space
        observer_space = Dict(
            {
                "own_obs": self.env.observation_space,
                # These two fields are filled in by the CentralCriticObserver, and are
                # not used for inference, only for training.
                "opponent_obs": self.env.observation_space,
                "opponent_action": self.env.action_space,
            }
        )

        save_path = "Models/PPO/CentralizedCritic/"
        self.config["batch_mode"] = "complete_episodes"
        self.update_multiagent_config()
        self.config["multiagent"]["observation_fn"] = central_critic_observer
        self.config["model"]["custom_model"] = "cc_model"
        self.config["callbacks"] = FillInActions

        result = tune.run(ppo.PPOTrainer,
                          verbose=1,
                          config=self.config,
                          stop={"timesteps_total": self.time_steps},
                          local_dir=save_path,
                          checkpoint_freq=1,
                          checkpoint_at_end=True
                          )
        checkpoint_path = result.get_last_checkpoint()
        print("Checkpoint path:", checkpoint_path)
        return checkpoint_path, results

    def train_qmix(self, mixer="qmix"):

        grouping = {
            "group_1": [i for i in range(AGENTS)],
        }

        self.env.action_space = gym.spaces.Discrete(2)
        self.env.agent_type = "discrete"

        obs_space = Tuple(
            [
                self.env.observation_space for _ in range(AGENTS)
            ]
        )
        act_space = Tuple(
            [
                self.env.action_space for _ in range(AGENTS)
            ]
        )

        register_env(
            "grouped_ringroad",
            lambda config: self.env.with_agent_groups(
                grouping, obs_space=obs_space, act_space=act_space
            ),
        )

        config = qmix.DEFAULT_CONFIG.copy()
        config["env"] = "grouped_ringroad"
        config["train_batch_size"] = 32
        config["evaluation_interval"] = 2
        config["mixer"] = mixer
        config["num_workers"] = 2
        config["num_gpus"] = 1
        config["buffer_size"] = 1000
        config["lr"] = 1e-3
        config["framework"] = "torch"
        config["model"]["lstm_cell_size"] = 32
        config["model"]["max_seq_len"] = 10

        save_path = None
        if mixer == "vdn":
            save_path = "Models/VDN/"
        else:
            save_path = "Models/QMIX/"

        results = tune.run("QMIX",
                           verbose=1,
                           config=config,
                           stop={"timesteps_total": self.time_steps},
                           local_dir=save_path,
                           checkpoint_freq=10,
                           checkpoint_at_end=True
                           )
        checkpoint_path = results.get_last_checkpoint()
        print("Checkpoint path:", checkpoint_path)
        return checkpoint_path, results

    def eval_qmix(self, path, mixer="qmix"):
        self.config["env_config"]["eval_mode"] = True
        self.config["env_config"]["enable_render"] = True
        self.config["num_workers"] = 0

        grouping = {
            "group_1": [i for i in range(AGENTS)],
        }

        self.env.action_space = gym.spaces.Discrete(2)
        self.env.agent_type = "discrete"

        obs_space = Tuple(
            [
                self.env.observation_space for _ in range(AGENTS)
            ]
        )
        act_space = Tuple(
            [
                self.env.action_space for _ in range(AGENTS)
            ]
        )

        register_env(
            "grouped_ringroad",
            lambda config: self.env.with_agent_groups(
                grouping, obs_space=obs_space, act_space=act_space
            ),
        )

        config = qmix.DEFAULT_CONFIG.copy()
        config["env"] = "grouped_ringroad"
        config["train_batch_size"] = 32
        config["evaluation_interval"] = 2
        config["mixer"] = mixer
        config["num_workers"] = 0
        config["num_gpus"] = 1
        config["buffer_size"] = 1000
        config["lr"] = 1e-6
        config["framework"] = "torch"
        config["model"]["lstm_cell_size"] = 32
        config["model"]["max_seq_len"] = 10

        config["explore"] = False

        self.agent = qmix.QMixTrainer(config=config)

        self.agent.restore(path)

        env = self.env.with_agent_groups(
                grouping, obs_space=obs_space, act_space=act_space
            )

        # run until episode ends
        episode_reward = 0
        done = {"__all__": False}
        obs = env.reset()

        state = [np.zeros((AGENTS,  config["model"]["lstm_cell_size"]), np.float32) for _ in range(2)]
        prev_a = 0
        prev_r = 0
        met = Metrics(env.env)

        def convert_action_tupletodict(action_tuple):
            action_dict = {}
            for i in range(len(action_tuple)):
                action_dict[i] = action_tuple[i]
            return action_dict

        while not done["__all__"]:
            action, state_out, _ = self.agent.compute_action(obs['group_1'], state=state, prev_a=prev_a, prev_r=prev_r)
            action_dict = convert_action_tupletodict(action)
            obs, reward, done, info = env.step(action_dict)

            met.step()
            prev_a = action
            prev_r = reward
            state = state_out

            print(env.env.action_steps, reward)
        met.plot(self.config)
        return episode_reward
