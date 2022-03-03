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
        scripts.train_dqn.evaluate("")

    elif args.type == "eval_ppo":
        scripts.train_ppo.evaluate("/home/vamsi/Documents/GitHub/RingRoad-RL/Models/PPO/PPOTrainer_2022-03-02_17-00-29/PPOTrainer_ringroad-v1_29c23_00000_0_2022-03-02_17-00-29/checkpoint_000100/checkpoint-100")

    elif args.type == "eval_idm":
        scripts.evaluate_idm.evaluate()



if __name__ == "__main__":
    main()
