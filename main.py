import scripts.train_dqn, scripts.evaluate_idm, scripts.train_ppo, scripts.multiagent_train_ppo
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, required=True)
    args = parser.parse_args()

    if args.type == "train_dqn":
        scripts.train_dqn.train()

    elif args.type == "train_ppo":
        scripts.train_ppo.train()

    elif args.type == "multiagent_ppo":
        scripts.multiagent_train_ppo.train()

    elif args.type == "multiagent_ppo_ind":
        scripts.multiagent_train_ppo.train_multiagent()

    elif args.type == "multiagent_evalppo":
        scripts.multiagent_train_ppo.evaluate("/home/vamsi/Documents/GitHub/RingRoad-RL/Models/PPO/PPOTrainer_2022-04-06_10-23-24/PPOTrainer_ringroad-v1_7d822_00000_0_2022-04-06_10-23-24/checkpoint_000250/checkpoint-250")

    elif args.type == "multiagent_evalppo_ind":
        scripts.multiagent_train_ppo.evaluate_multiagent("/home/vamsi/Documents/GitHub/RingRoad-RL/Models/PPO/PPOTrainer_2022-04-26_16-51-18/PPOTrainer_multiagent_ringroad-v1_fe68c_00000_0_2022-04-26_16-51-18/checkpoint_000003/checkpoint-3")


    elif args.type == "eval_dqn":
        scripts.train_dqn.evaluate("/home/vamsi/Documents/GitHub/RingRoad-RL/Models/DQN/DQNTrainer_2022-03-03_19-28-07/DQNTrainer_ringroad-v1_f4621_00000_0_2022-03-03_19-28-08/checkpoint_000488/checkpoint-488")

    elif args.type == "eval_ppo":
        scripts.train_ppo.evaluate("/home/vamsi/Documents/GitHub/RingRoad-RL/Models/PPO/PPOTrainer_2022-03-16_19-00-00/PPOTrainer_ringroad-v1_2dc8d_00000_0_2022-03-16_19-00-00/checkpoint_000250/checkpoint-250")

    elif args.type == "eval_idm":
        scripts.evaluate_idm.evaluate()



if __name__ == "__main__":
    main()
