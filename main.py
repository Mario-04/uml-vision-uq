import argparse

from src.data.get_data import download_all

def parse_args():
    parser = argparse.ArgumentParser(description="UML Vision UQ")
    parser.add_argument("--download_data", 
                        action="store_true",
                        help="Download CIFAR-10 and CIFAR-100 datasets")

    return parser.parse_args()

def main():
    args = parse_args()
    if args.download_data:
        download_all()



if __name__ == "__main__":
    main()