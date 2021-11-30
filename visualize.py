import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def visualize(input_file: Path, output_file: Path, scale: int) -> None:
    df_all = pd.read_csv(input_file)
    n_at_list = sorted(df_all['at'].unique())
    n_rows = int(np.ceil(len(n_at_list) / 2))
    fig, axes = plt.subplots(n_rows, 2, figsize=(scale * 2, scale * n_rows), sharex='all', sharey='all')

    for n_at, ax in zip(n_at_list, axes.flat):
        df = df_all[df_all['at'] == n_at]
        names = sorted(set(df['src']) | set(df['dst']))
        names_indices = {name: i for i, name in enumerate(names)}
        n = len(names)
        matrix = np.zeros((n, n))
        for _, row in df.iterrows():
            i, j = names_indices[row['src']], names_indices[row['dst']]
            matrix[i, j] = row['similarity'] / (n_at - 1)
            matrix[j, i] = row['similarity'] / (n_at - 1)

        np.fill_diagonal(matrix, 1)
        ax.set_title(f'@{n_at}')
        sns.heatmap(matrix, annot=True, fmt='.2f', ax=ax, cmap='mako_r', xticklabels=names, yticklabels=names,
                    square=True, vmin=0, vmax=1, cbar=False)

    if output_file is not None:
        plt.savefig(output_file, bbox_inches='tight')
    else:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_file', type=Path, help='Input similarity.csv file')
    parser.add_argument('output_file', type=Path, help='Output .png or .pdf plot file')
    parser.add_argument('--scale', type=float, default=6, help='Increase this number if the figure is too cluttered')
    args = parser.parse_args()

    visualize(args.input_file, args.output_file, args.scale)
