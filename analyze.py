import argparse
import itertools
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


def compute_neighbors(embeddings_file: Path, n_neighbors: int, metric: str, indices: Optional[np.ndarray]):
    embeddings = np.load(str(embeddings_file))
    if indices is not None:
        embeddings = embeddings[indices]
    model = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
    model.fit(embeddings)
    _, neighbors = model.kneighbors(embeddings)
    return neighbors


def intersect(src, dst):
    return len(set(src) & set(dst)) / len(src)


def spearman(src, dst):
    ids = sorted(set(src) | set(dst))
    src_idx = [np.where(src == i)[0][0] if i in src else len(src) for i in ids]
    dst_idx = [np.where(dst == i)[0][0] if i in dst else len(dst) for i in ids]
    return spearmanr(src_idx, dst_idx).correlation


def intersect_rbo(src, dst):
    import rbo
    return rbo.RankingSimilarity(src, dst).rbo()


def analyze(input_dir: Path, list_file: Path, output_file: Path, n_neighbors_list: list[int],
            metric: str, indices_file: Path = None) -> None:
    embeddings_list = pd.read_csv(list_file, comment='#')
    n_embeddings = len(embeddings_list)

    indices = np.loadtxt(str(indices_file), dtype=int) if indices_file is not None else None

    output_file.parent.mkdir(exist_ok=True)
    neighbors_all = {}
    for name, embeddings_file in tqdm(embeddings_list.values, total=n_embeddings):
        neighbors_all[name] = compute_neighbors(input_dir / embeddings_file, max(n_neighbors_list)+1, metric, indices)

    results: dict[str, list] = {'src': [], 'dst': [], 'similarity': [], 'at': []}
    for name_src, name_dst in tqdm(itertools.combinations(embeddings_list.name, 2),
                                   total=n_embeddings * (n_embeddings - 1) // 2):
        for n_neighbors in n_neighbors_list:
            n_common_neighbors = []
            for (row_src, row_dst) in zip(neighbors_all[name_src], neighbors_all[name_dst]):
                # naive implementation
                n_common_neighbors.append(intersect_rbo(row_src[1:n_neighbors+1], row_dst[1:n_neighbors+1]))
            results['src'].append(name_src)
            results['dst'].append(name_dst)
            results['similarity'].append(np.mean(n_common_neighbors))
            results['at'].append(n_neighbors)

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_dir', type=Path, help='Input directory that contains .npy files')
    parser.add_argument('list_file', type=Path,
                        help='List .csv file that contains list of spaces to compare, needs to contain columns NAME, '
                             'FILE')
    parser.add_argument('output_file', type=Path,
                        help='Output .csv file that will contain the computed results')
    parser.add_argument('--at', nargs='+', type=int, default=[5, 10, 20, 50, 100, 200],
                        help='Number of neighbors retrieved')
    parser.add_argument('--metric', type=str, default='minkowski', help='Distance used to compute nearest neighbors')
    parser.add_argument('--indices-file', type=Path,
                        help='.txt file that contains indices that subset the data for analysis')
    args = parser.parse_args()

    analyze(args.input_dir, args.list_file, args.output_file, args.at, args.metric, args.indices_file)
