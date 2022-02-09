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

    # string conversion function: don't print 1.0 and remove leading 0s
    to_str = np.vectorize(lambda x: f'{x:.2f}'.removeprefix('0') if x < 1.0 else ' ')

    for n_at, ax in zip(n_at_list, axes.flat):
        df = df_all[df_all['at'] == n_at]
        names = sorted(set(df['src']) | set(df['dst']))
        names_indices = {name: i for i, name in enumerate(names)}
        n = len(names)
        matrix = np.zeros((n, n))
        for _, row in df.iterrows():
            i, j = names_indices[row['src']], names_indices[row['dst']]
            matrix[i, j] = row['similarity']
            matrix[j, i] = row['similarity']

        np.fill_diagonal(matrix, 1)
        annot = to_str(matrix)

        ax.set_title(f'@{n_at}')
        sns.heatmap(matrix, annot=annot, fmt='s', ax=ax, cmap='mako_r', xticklabels=names, yticklabels=names,
                    square=True, vmin=0, vmax=1, cbar=False)
        ax.tick_params(left=False, bottom=False)

    if output_file is not None:
        plt.savefig(output_file, bbox_inches='tight')
    else:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='Create figure to visualize the data in NN-similarity .csv file')
    parser.add_argument('input_file', type=Path, help='input similarity.csv file')
    parser.add_argument('output_file', type=Path, help='output .png or .pdf plot file')
    parser.add_argument('--scale', type=float, default=6, help='increase this number if the figure is too cluttered')
    args = parser.parse_args()

    visualize(args.input_file, args.output_file, args.scale)
