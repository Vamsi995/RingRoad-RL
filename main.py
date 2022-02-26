import scripts.train_dqn, scripts.evaluate_dqn
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, required=True)
    args = parser.parse_args()

    if args.type == "train_dqn":
        scripts.train_dqn.train()

    if args.type == "eval_dqn":
        scripts.evaluate_dqn.main()



if __name__ == "__main__":
    main()
