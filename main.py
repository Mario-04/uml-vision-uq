import argparse

from src.data.get_data import download_all

def parse_args():
    parser = argparse.ArgumentParser(description="UML Vision UQ")
    parser.add_argument("--download_data",
                        action="store_true",
                        help="Download CIFAR-10 and CIFAR-100 datasets")

    parser.add_argument("--train_cifar10CNN",
                        action="store_true",
                        help="Train the baseline CNN on CIFAR-10")

    parser.add_argument("--seed", type=int, default=42,
                        help="Global random seed for reproducibility")

    return parser.parse_args()

def main():
    args = parse_args()
    if args.download_data:
        download_all()

    if args.train_cifar10CNN:
        from src.models.cnn import baseline_CNN
        from src.train import train_baseline_cifar10CNN
        train_baseline_cifar10CNN(seed=args.seed)


if __name__ == "__main__":
    main()