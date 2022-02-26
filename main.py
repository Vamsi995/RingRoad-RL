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
        scripts.train_dqn.evaluate("/home/vamsi/Documents/GitHub/RingRoad-RL/Models/DQN/DQNTrainer_2022-02-26_20-18-27/DQNTrainer_ringroad-v1_2849a_00000_0_2022-02-26_20-18-27/checkpoint_000200/checkpoint-200")

    elif args.type == "eval_ppo":
        scripts.train_ppo.evaluate("")

    elif args.type == "eval_idm":
        scripts.evaluate_idm.evaluate()



if __name__ == "__main__":
    main()
