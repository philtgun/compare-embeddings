import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
# from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


def compute_neighbors(embeddings_list: pd.DataFrame, input_dir: Path, n_neighbors: int):
    neighbors_all = {}
    for _, (name, embeddings_file) in tqdm(embeddings_list.iterrows()):
        embeddings = np.load(input_dir / embeddings_file)
        model = NearestNeighbors(n_neighbors=n_neighbors)
        model.fit(embeddings)
        _, neighbors_all[name] = model.kneighbors(embeddings)

    return neighbors_all


def analyze(input_dir: Path, output_dir: Path, list_file: Path, n_neighbors: int = 10, n_seeds: int = 100) -> None:
    embeddings_list = pd.read_csv(list_file)
    output_dir.mkdir(exist_ok=True)

    neighbors_file = output_dir / 'neighbors.npz'
    if not neighbors_file.exists():
        neighbors_all = compute_neighbors(embeddings_list, input_dir, n_neighbors)
        np.savez(neighbors_file, **neighbors_all)
    else:
        neighbors_all = dict(np.load(neighbors_file).items())
        print('Loading neighbor data')

    for (name_src, neighbors_src), (name_cmp, neighbors_cmp) in itertools.combinations(neighbors_all.items(), 2):
        print(name_src, name_cmp)
        # TODO implement
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_dir', type=Path, help='Input directory that contains .npy files')
    parser.add_argument('output_dir', type=Path,
                        help='Output directory that will contain the intermediate and output files')
    parser.add_argument('list_file', type=Path,
                        help='List .csv file that contains list of spaces to compare, needs to contain columns NAME, '
                             'FILE')
    args = parser.parse_args()

    analyze(args.input_dir, args.output_dir, args.list_file)
