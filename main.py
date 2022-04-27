import scripts.train_dqn, scripts.evaluate_idm, scripts.train_ppo, scripts.multiagent_train_ppo
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, required=True)
    args = parser.parse_args()

    if args.type == "multiagent_shared_ppo":
        scripts.multiagent_train_ppo.train()

    elif args.type == "multiagent_shared_evalppo":
        scripts.multiagent_train_ppo.evaluate(
            "/home/vamsi/Documents/GitHub/RingRoad-RL/Models/PPO/PPOTrainer_2022-04-26_11-15-06/PPOTrainer_multiagent_ringroad-v1_07147_00000_0_2022-04-26_11-15-07/checkpoint_000221/checkpoint-221")

    elif args.type == "multiagent_nonshared_ppo":
        scripts.multiagent_train_ppo.train_multiagent()

    elif args.type == "multiagent_nonshared_evalppo":
        scripts.multiagent_train_ppo.evaluate_multiagent(
            "/home/vamsi/Documents/GitHub/RingRoad-RL/Models/PPO/PPOTrainer_2022-04-26_11-15-06/PPOTrainer_multiagent_ringroad-v1_07147_00000_0_2022-04-26_11-15-07/checkpoint_000221/checkpoint-221")

    elif args.type == "eval_idm":
        scripts.evaluate_idm.evaluate()


if __name__ == "__main__":
    main()
