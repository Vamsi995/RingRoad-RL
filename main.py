import scripts.evaluate_dqn, scripts.evaluate_a2c, scripts.train_dqn, scripts.train_a2c, scripts.evaluate_idm
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, required=True)
    args = parser.parse_args()

    if args.type == "train_dqn":
        scripts.train_dqn.train()
    elif args.type == "train_a2c":
        scripts.train_a2c.train()
    elif args.type == "eval_dqn":
        scripts.evaluate_dqn.main()
    elif args.type == "eval_a2c":
        scripts.evaluate_a2c.main()
    elif args.type == "eval_idm":
        scripts.evaluate_idm.main()
    # elif args.type == "train_trpo":
    #     scripts.train_trpo.train()
    # elif args.type == "eval_trpo":
    #     scripts.evaluate_trpo.main()


if __name__ == "__main__":
    main()