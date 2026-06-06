import argparse

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
    
    parser.add_argument("--train_dropout",
                        action="store_true",
                        help="Train the MC Dropout CNN on CIFAR-10")

    return parser.parse_args()

def main():
    args = parse_args()
    if args.download_data:
        from src.data import download_all
        download_all()

    if args.train_cifar10CNN:
        from src.models.cnn import baseline_CNN
        from src.train import train_baseline_cifar10CNN
        train_baseline_cifar10CNN(seed=args.seed)

    if args.train_dropout:
        from src.models.cnn import MCDropout_CNN
        from src.train import train_mc_dropout_cifar10CNN
        train_mc_dropout_cifar10CNN(seed=args.seed)

if __name__ == "__main__":
    main()