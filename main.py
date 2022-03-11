import scripts.train_dqn, scripts.evaluate_idm, scripts.train_ppo
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, required=True)
    args = parser.parse_args()

    if args.type == "train_dqn":
        scripts.train_dqn.train()

    elif args.type == "train_ppo":
        scripts.train_ppo.train()

    elif args.type == "eval_dqn":
        scripts.train_dqn.evaluate("/home/vamsi/Documents/GitHub/RingRoad-RL/Models/DQN/DQNTrainer_2022-03-03_19-28-07/DQNTrainer_ringroad-v1_f4621_00000_0_2022-03-03_19-28-08/checkpoint_000488/checkpoint-488")

    elif args.type == "eval_ppo":
        scripts.train_ppo.evaluate("/home/vamsi/Documents/GitHub/RingRoad-RL/Models/PPO/PPOTrainer_2022-03-04_10-41-17/PPOTrainer_ringroad-v1_85549_00000_0_2022-03-04_10-41-17/checkpoint_000124/checkpoint-124")

    elif args.type == "eval_idm":
        scripts.evaluate_idm.evaluate()



if __name__ == "__main__":
    main()
