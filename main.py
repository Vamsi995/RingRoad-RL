import scripts.train_dqn, scripts.evaluate
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, required=True)
    args = parser.parse_args()

    if args.type == "train_dqn":
        scripts.train_dqn.train()
    if args.type == "eval":
        scripts.evaluate.evaluate()



if __name__ == "__main__":
    main()
