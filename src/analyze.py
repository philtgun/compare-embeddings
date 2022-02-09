# Copyright 2022 Philip Tovstogan, Music Technology Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import itertools
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


def compute_neighbors(embeddings_file: Path, n_neighbors: int, metric: str,
                      indices: Optional[np.ndarray]) -> np.ndarray:
    embeddings = np.load(str(embeddings_file))
    if indices is not None:
        embeddings = embeddings[indices]
    model = NearestNeighbors(n_neighbors=n_neighbors + 1, metric=metric)
    model.fit(embeddings)
    _, neighbors = model.kneighbors(embeddings)
    return neighbors[:, 1:]


def intersect(src: list, dst: list) -> float:
    return len(set(src) & set(dst)) / len(src)


def spearman(src: list, dst: list) -> float:
    ids = sorted(set(src) | set(dst))
    src_idx = [np.where(src == i)[0][0] if i in src else len(src) for i in ids]
    dst_idx = [np.where(dst == i)[0][0] if i in dst else len(dst) for i in ids]
    return spearmanr(src_idx, dst_idx).correlation


def intersect_rbo(src: list, dst: list) -> float:
    import rbo
    return rbo.RankingSimilarity(src, dst).rbo()


def load_and_compute_neighbors(input_dir: Path, list_file: Path, n_neighbors: int, metric: str,
                               indices_file: Path = None) -> dict[str, np.ndarray]:
    embeddings_list = pd.read_csv(list_file, comment='#')
    indices = np.loadtxt(str(indices_file), dtype=int) if indices_file is not None else None

    neighbors_all = {}
    for name, embeddings_file in tqdm(embeddings_list.values, total=len(embeddings_list)):
        neighbors_all[name] = compute_neighbors(input_dir / embeddings_file, n_neighbors, metric, indices)

    return neighbors_all


def analyze(input_dir: Path, list_file: Path, output_file: Path, n_neighbors_list: list[int],
            metric: str, indices_file: Path = None) -> None:

    neighbors_all = load_and_compute_neighbors(input_dir, list_file, max(n_neighbors_list), metric, indices_file)
    n_embeddings = len(neighbors_all)

    results: dict[str, list] = {'src': [], 'dst': [], 'similarity': [], 'at': []}
    for name_src, name_dst in tqdm(itertools.combinations(neighbors_all.keys(), 2),
                                   total=n_embeddings * (n_embeddings - 1) // 2):
        for n_neighbors in n_neighbors_list:
            similarity = []
            for (row_src, row_dst) in zip(neighbors_all[name_src], neighbors_all[name_dst]):
                similarity.append(intersect(row_src[:n_neighbors], row_dst[:n_neighbors]))
            results['src'].append(name_src)
            results['dst'].append(name_dst)
            results['similarity'].append(np.mean(similarity))
            results['at'].append(n_neighbors)

    results_df = pd.DataFrame(results)
    output_file.parent.mkdir(exist_ok=True)
    results_df.to_csv(output_file, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Compute NN-similarity at multiple cutoffs for spaces that are specified in the list file and '
                    'save the results in .csv file')
    parser.add_argument('input_dir', type=Path, help='input directory that contains .npy files')
    parser.add_argument('list_file', type=Path,
                        help='list .csv file that contains list of spaces to compare, needs to contain columns NAME, '
                             'FILE')
    parser.add_argument('output_file', type=Path,
                        help='output .csv file that will contain the computed results')
    parser.add_argument('--at', nargs='+', type=int, default=[5, 10, 100, 200],
                        help='number of neighbors retrieved')
    parser.add_argument('--metric', type=str, default='minkowski', help='distance used to compute nearest neighbors')
    parser.add_argument('--indices-file', type=Path,
                        help='.txt file that contains indices that subset the data for analysis')
    args = parser.parse_args()

    analyze(args.input_dir, args.list_file, args.output_file, args.at, args.metric, args.indices_file)
