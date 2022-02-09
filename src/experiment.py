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
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd


def separate(df: pd.DataFrame, x: str) -> list[list]:
    return [group['rating'].values for _, group in df.groupby(x)]


def analyze_variable(df: pd.DataFrame, x: str) -> None:
    separated = separate(df, x)
    print(x, stats.kruskal(*separated))
    print(x, stats.f_oneway(*separated))


def plot(df: pd.DataFrame, x: str, output_file: Path) -> None:
    plt.figure(figsize=[4, 4])
    sns.barplot(data=df, x=x, y='rating')
    plt.xticks(rotation=90)
    plt.tick_params(left=False, bottom=False)
    sns.despine(left=True, bottom=True)
    plt.xlabel('')
    plt.savefig(output_file, bbox_inches='tight')


def main(input_dir: Path, output_dir: Path) -> None:
    data = load_data(input_dir)

    df_all_list = []
    for participant, df in data.items():
        df = pd.DataFrame(df.stack()).reset_index()
        df.columns = ['space', 'ref_track', 'rating']
        df['participant'] = participant
        df_all_list.append(df)

    df_all = pd.concat(df_all_list)

    print(stats.shapiro(df_all['rating'].values))

    analyze_variable(df_all, 'space')
    analyze_variable(df_all, 'ref_track')
    analyze_variable(df_all, 'participant')

    print(pairwise_tukeyhsd(endog=df_all['rating'], groups=df_all['space']))
    print(pairwise_tukeyhsd(endog=df_all['rating'], groups=df_all['ref_track']))

    output_dir.parent.mkdir(exist_ok=True)
    plot(df_all, 'space', output_dir)


def load_data(input_dir: Path) -> dict[str, pd.DataFrame]:
    data = {}
    for csv_file in input_dir.glob('*.csv'):
        data[csv_file.stem] = pd.read_csv(csv_file, index_col=0)
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='Analyze responses from online experiment')
    parser.add_argument('input_dir', type=Path, help='input directory with csv files')
    parser.add_argument('output_file', type=Path, help='output figure')
    args = parser.parse_args()
    main(args.input_dir, args.output_file)
