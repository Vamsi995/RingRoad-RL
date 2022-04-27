import gym
import numpy as np
import ray
from ray import tune
from ray.rllib.agents import dqn, ppo
from Ring_Road import RingRoad
from Ring_Road.metrics import Metrics


class Experiment:

    def __init__(self, env_config, config):
        ray.shutdown()
        ray.init()
        self.algorithm = env_config["algorithm"]
        self.time_steps = env_config["time_steps"]
        self.agent = None
        self.config = config
        self.env = RingRoad(env_config)

    def train(self):

        save_path = ""
        if self.config["model"]["use_lstm"]:
            save_path = "Models/PPO/SingleAgent/LSTM/"
        else:
            save_path = "Models/PPO/SingleAgent/"

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
                               checkpoint_at_end=True,
                               checkpoint_freq=20
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
        done = False
        obs = env.reset()

        met = Metrics(env)

        if self.config["model"]["use_lstm"]:
            state = [np.zeros(self.config["model"]["lstm_cell_size"], np.float32) for _ in range(2)]
            prev_a = 0
            prev_r = 0

            while not done:
                action, state_out, _ = self.agent.compute_single_action(obs, state=state, prev_action=prev_a, prev_reward=prev_r)
                obs, reward, done, info = env.step(action)
                episode_reward += reward

                met.step()
                prev_a = action[0]
                prev_r = reward
                state = state_out

            met.plot(self.config)
            return episode_reward

        else:
            while not done:
                action = self.agent.compute_single_action(obs)

                obs, reward, done, info = env.step(action)
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
                # print(env.action_steps, reward)
            met.plot(self.config)
            return episode_reward
