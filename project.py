import argparse
import logging
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def project(input_file: Path, output_dir: Path, projections: list[str], n_pcs: int, seed: int) -> None:
    embeddings = np.load(str(input_file))

    pca = PCA(random_state=seed, copy=False)
    embeddings_pca = pca.fit_transform(embeddings)

    for projection_name in projections:
        logging.info(f'Performing {projection_name}')
        if projection_name == 'pca':
            embeddings_projected = embeddings_pca[:, :2]
        elif projection_name == 'tsne':
            tsne = TSNE(n_components=2, random_state=seed, verbose=True)
            embeddings_projected = tsne.fit_transform(embeddings_pca[:, :n_pcs])
        elif projection_name == 'umap':
            from umap import UMAP
            umap = UMAP(n_components=2, init='random', random_state=seed)
            embeddings_projected = umap.fit_transform(embeddings_pca[:, :n_pcs])
        else:
            raise ValueError(f'Invalid projection name: {projection_name}')

        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / (input_file.stem + '-' + projection_name + '-' + str(seed) + '.npy')
        np.save(str(output_file), embeddings_projected)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_file', type=Path, help='Input .npy file')
    parser.add_argument('output_dir', type=Path, help='Output directory to contain projected embeddings')
    parser.add_argument('--projections', type=str, nargs='+', default=['pca', 'tsne', 'umap'],
                        help='Projections to apply')
    parser.add_argument('--n-pcs', type=int, default=20,
                        help='Number of principal components to be used for tsne/umap')
    parser.add_argument('--seed', type=int, default=0, help='randomization seed for reproducibility')
    args = parser.parse_args()

    project(args.input_file, args.output_dir, args.projections, args.n_pcs, args.seed)
