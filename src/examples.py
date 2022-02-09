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
import json
from pathlib import Path

import numpy as np

from analyze import load_and_compute_neighbors


def generate_examples(input_dir: Path, list_file: Path, output_file: Path, indices_file: Path, ids_file: Path,
                      n_neighbors: int, n_references: int, strategy: str, custom_ids: list[int]) -> None:
    neighbors_all = load_and_compute_neighbors(input_dir, list_file, n_neighbors, 'cosine', indices_file)
    track_ids = np.loadtxt(str(ids_file), dtype=int)

    if strategy == 'dissimilar':
        for i, track_id in enumerate(track_ids):
            for name, neighbors in neighbors_all.items():
                pass  # TODO: implement
        reference_ids = []
    elif strategy == 'random':
        reference_ids = np.random.choice(track_ids, n_references)
    elif strategy == 'custom':
        if len(custom_ids) != n_references:
            raise ValueError(f'Custom-ids should have exactly -n ({n_references}) items for "custom" strategy')
        reference_ids = custom_ids
    else:
        raise ValueError(f'Invalid strategy: {strategy}')

    def neighbors_for_track(reference_id: int) -> dict[str, np.ndarray]:
        reference_idx = np.searchsorted(track_ids, reference_id)
        return {_name: [int(track_ids[neighbor_idx]) for neighbor_idx in _neighbors[reference_idx]]
                for _name, _neighbors in neighbors_all.items()}

    data = [{'reference': int(track_id), 'options': neighbors_for_track(track_id)} for track_id in reference_ids]
    with output_file.open('w') as fp:
        json.dump(data, fp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_dir', type=Path, help='input directory that contains npy files')
    parser.add_argument('list_file', type=Path,
                        help='list csv file that contains list of spaces to compare, needs to contain columns NAME, '
                             'FILE')
    parser.add_argument('output_file', type=Path, help='output json file with data for experiment')
    parser.add_argument('--indices-file', type=Path,
                        help='txt file that contains indices that subset the data for analysis')
    parser.add_argument('--ids-file', type=Path,
                        help='txt file that contains Jamendo ids that correspond to the indices')

    parser.add_argument('--at', type=int, default=4, help='number of neighbors retrieved')
    parser.add_argument('-n', type=int, default=4, help='number of seed tracks')
    parser.add_argument('--strategy', type=str, choices=['random', 'dissimilar', 'custom'], default='random',
                        help='how to pick the reference tracks')
    parser.add_argument('--seed', type=int, help='optional seed for reproducibility')
    parser.add_argument('--custom-ids', nargs='+', type=int,
                        help='track ids for the custom strategy, expected exactly -n ids')

    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    generate_examples(args.input_dir, args.list_file, args.output_file, args.indices_file, args.ids_file,
                      args.at, args.n, args.strategy, args.custom_ids)
