import argparse
from pathlib import Path

import torch
from src.data.get_data import download_all, load_cifar10
from src.train import get_default_device
from src.evaluate import evaluate_full
from src.artifacts import load_run

def parse_args():
    parser = argparse.ArgumentParser(description="UML Vision UQ")
    parser.add_argument("--download_data",
                        action="store_true",
                        help="Download CIFAR-10 and SVHN datasets")

    parser.add_argument("--train_cifar10CNN",
                        action="store_true",
                        help="Train the baseline CNN on CIFAR-10")

    parser.add_argument("--train_cifar10MCDropoutCNN",
                        action="store_true",
                        help="Train the MC-Dropout CNN on CIFAR-10")
    
    parser.add_argument("--evaluate",
                        action="store_true",
                        help="Evaluate a trained model on test set")

    parser.add_argument("--dropout_p", type=float, default=0.2,
                        help="P for dropout layers in MC-Dropout CNN")

    parser.add_argument("--seed", type=int, default=42,
                        help="Global random seed for reproducibility")
    
    parser.add_argument("--run_dir", type=str, default=None,
                        help="Directory of model to be evaluated")
    
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of epochs to train for (if training)")

    return parser.parse_args()

def main():
    args = parse_args()
    if args.download_data:
        download_all()

    if args.train_cifar10CNN:
        from src.train import train_baseline_cifar10CNN
        train_baseline_cifar10CNN(seed=args.seed, epochs=args.epochs)

    if args.train_cifar10MCDropoutCNN:
        from src.train import train_mc_dropout_cifar10CNN
        train_mc_dropout_cifar10CNN(dropout_p=args.dropout_p, seed=args.seed, epochs=args.epochs)

    if args.evaluate:
        if args.run_dir is None:
            raise ValueError("Please ensure --run_dir exists and is a valid model.")
        
        model, history, config = load_run(args.run_dir)
        _, _, test_loader = load_cifar10(batch_size=256, seed=args.seed)

        device = get_default_device()
        model.to(device)

        reliability_path = Path(args.run_dir) / "reliability.png"
        results = evaluate_full(
            model, test_loader, device, reliability_path=reliability_path
        )

        print(results)
        print(f"Reliability diagram saved to {reliability_path}")


if __name__ == "__main__":
    main()