import scripts.evaluate, scripts.evaluate_a2c, scripts.train, scripts.a2c, scripts.evaluate_idm
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, required=True)
    args = parser.parse_args()

    if args.type == "train_dqn":
        scripts.train.train()
    elif args.type == "eval_dqn":
        scripts.evaluate.main()
    elif args.type == "train_a2c":
        scripts.a2c.train()
    elif args.type == "eval_a2c":
        scripts.evaluate_a2c.main()
    elif args.type == "eval_idm":
        scripts.evaluate_idm.main()

if __name__ == "__main__":
    main()
