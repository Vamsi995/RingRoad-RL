import scripts.train_dqn, scripts.evaluate_idm, scripts.train_ppo, scripts.multiagent_train_ppo
import argparse

mixer = "vdn"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, required=True)
    args = parser.parse_args()

    if args.type == "multiagent_shared_ppo":
        scripts.multiagent_train_ppo.train()

    elif args.type == "multiagent_shared_evalppo":
        scripts.multiagent_train_ppo.evaluate(
            "/home/vamsi/Documents/GitHub/RingRoad-RL/Models/PPO/MultiAgent/SharedPolicy/PPOTrainer_2022-05-01_01-34-57/PPOTrainer_multiagent_ringroad-v1_cef30_00000_0_2022-05-01_01-34-57/checkpoint_000221/checkpoint-221")

    elif args.type == "multiagent_nonshared_ppo":
        scripts.multiagent_train_ppo.train_multiagent()

    elif args.type == "multiagent_nonshared_evalppo":
        scripts.multiagent_train_ppo.evaluate_multiagent(
            "/home/vamsi/Documents/GitHub/RingRoad-RL/Models/PPO/MultiAgent/NonSharedPolicy/PPOTrainer_2022-05-01_11-47-03/PPOTrainer_multiagent_ringroad-v1_5186a_00000_0_2022-05-01_11-47-03/checkpoint_000250/checkpoint-250")


    elif args.type == "multiagent_centralized_critic":
        scripts.multiagent_train_ppo.train_multiagent_centralized_critic()

    elif args.type == "multiagent_qmix":
        scripts.multiagent_train_ppo.train_qmix(mixer)

    elif args.type == "multiagent_qmix_eval":
        scripts.multiagent_train_ppo.evaluate_qmix(
            "/home/vamsi/Documents/GitHub/RingRoad-RL/Models/VDN/QMIX/QMIX_grouped_ringroad_af722_00000_0_2022-05-03_19-25-26/checkpoint_000020/checkpoint-20", mixer)
    elif args.type == "eval_idm":
        scripts.evaluate_idm.evaluate()



if __name__ == "__main__":
    main()
