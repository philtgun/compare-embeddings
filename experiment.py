import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd


def separate(df: pd.DataFrame, x: str) -> list[list]:
    return [group['rating'].values for _, group in df.groupby(x)]


def plot_variable(df: pd.DataFrame, x: str, output_dir: Path):
    plt.figure(figsize=[8, 6])
    sns.boxplot(data=df, x=x, y='rating')
    plt.savefig(output_dir / f'{x}.png', bbox_inches='tight')

    separated = separate(df, x)
    print(x, stats.kruskal(*separated))
    print(x, stats.f_oneway(*separated))


def plot_bars(df: pd.DataFrame, x: str, output_dir: Path) -> None:
    plt.figure(figsize=[4, 4])
    sns.barplot(data=df, x=x, y='rating')
    plt.xticks(rotation=90)
    plt.tick_params(left=False, bottom=False)
    sns.despine(left=True, bottom=True)
    plt.xlabel('')
    plt.savefig(output_dir / f'{x}-bar.png', bbox_inches='tight')
    plt.savefig(output_dir / f'{x}-bar.pdf', bbox_inches='tight')


def main(input_dir: Path, output_dir: Path) -> None:
    data = load_data(input_dir)

    df_all = []
    for participant, df in data.items():
        df = pd.DataFrame(df.stack()).reset_index()
        df.columns = ['space', 'ref_track', 'rating']
        df['participant'] = participant
        df_all.append(df)

    df_all = pd.concat(df_all)

    print(stats.shapiro(df_all['rating'].values))

    output_dir.mkdir(exist_ok=True)
    plot_variable(df_all, 'space', output_dir)
    plot_variable(df_all, 'ref_track', output_dir)
    plot_variable(df_all, 'participant', output_dir)

    print(pairwise_tukeyhsd(endog=df_all['rating'], groups=df_all['space']))
    print(pairwise_tukeyhsd(endog=df_all['rating'], groups=df_all['ref_track']))

    plot_bars(df_all, 'space', output_dir)
    plot_bars(df_all, 'ref_track', output_dir)


def load_data(input_dir: Path) -> dict[str, pd.DataFrame]:
    data = {}
    for csv_file in input_dir.glob('*.csv'):
        data[csv_file.stem] = pd.read_csv(csv_file, index_col=0)
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_dir', type=Path, help='input directory with csv files')
    parser.add_argument('output_dir', type=Path, help='output directory with figures')
    args = parser.parse_args()
    main(**vars(args))
