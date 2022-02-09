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
import logging
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP


def project(input_file: Path, output_dir: Path, projections: list[str], seed: int) -> None:
    embeddings = np.load(str(input_file))

    pca = PCA(random_state=seed, copy=False)
    embeddings_pca = pca.fit_transform(embeddings)

    for projection_name in projections:
        logging.info(f'Performing {projection_name}')
        if projection_name == 'pca':
            embeddings_projected = embeddings_pca[:, :2]
        elif projection_name == 'tsne':
            tsne = TSNE(n_components=2, random_state=seed, verbose=True)
            embeddings_projected = tsne.fit_transform(embeddings)
        elif projection_name == 'umap':
            umap = UMAP(n_components=2, init='random', random_state=seed, verbose=True)
            embeddings_projected = umap.fit_transform(embeddings)
        else:
            raise ValueError(f'Invalid projection name: {projection_name}')

        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / (input_file.stem + '-' + projection_name + '-' + str(seed) + '.npy')
        np.save(str(output_file), embeddings_projected)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='Compute projections of one space')
    parser.add_argument('input_file', type=Path, help='input .npy file')
    parser.add_argument('output_dir', type=Path, help='output directory to contain projected embeddings')
    parser.add_argument('--projections', type=str, nargs='+', default=['pca', 'tsne', 'umap'],
                        help='Projections to apply')
    parser.add_argument('--seed', type=int, default=0, help='randomization seed for reproducibility')
    args = parser.parse_args()

    project(args.input_file, args.output_dir, args.projections, args.seed)
