import gym
import ray
from ray import tune
from ray.rllib.agents import dqn, ppo
from Ring_Road import RingRoad, MultiAgentRingRoad
from Ring_Road.metrics import Metrics


class Experiment:

    def __init__(self, env_config, config):
        ray.shutdown()
        ray.init()
        self.algorithm = env_config["algorithm"]
        self.time_steps = env_config["time_steps"]
        self.agent = None
        self.config = config
        self.env = MultiAgentRingRoad(env_config)

    def train(self):
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
                               local_dir="Models/PPO/",
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

        policy = self.agent.get_policy()

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

            #
            # if info["action"] <= 1.0 and info["action"] >= -1.0:
            #     print("Action:", info["action"])
            #     print(env.action_steps, reward)
            # else:
            #     # print("Action:", action)
            #     print("Action invalid:", action)
            #     break
            met.step()
            # episode_reward += reward
            # env.render()
            print(env.action_steps, reward)
        met.plot()
        return episode_reward
