import argparse
from pathlib import Path


def find_distinct(input_file: Path) -> None:
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_file', type=Path, help='Input similarity.csv file')
    args = parser.parse_args()

    find_distinct(args.input_file)
