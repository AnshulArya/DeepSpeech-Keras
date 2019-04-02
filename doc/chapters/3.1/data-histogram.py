import pickle
from typing import Any
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# cmap = plt.get_cmap('YlGnBu')
# linear = sns.color_palette("YlGnBu")
# sns.palplot(linear)
# plt.show()


def dump_session(data: Any):
    with open(f'{__file__}.session', 'wb') as file:
        pickle.dump(data, file)


def load_session():
    with open(f'{__file__}.session', 'rb') as file:
        return pickle.load(file)


def save_histogram(times: tuple, word_counts: tuple):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3))
    clarin_times, jurisdic_times = times
    clarin_wc, jurisdic_wc = word_counts

    sns.distplot(clarin_times, ax=ax1, label='Clarin-PL')
    sns.distplot(jurisdic_times, ax=ax1, label='Inne')
    ax1.set_xlabel(r'\textbf{Czas [s]}')
    ax1.set_xlim(right=8)
    ax1.legend(loc='upper right')

    ax2.hist([clarin_wc, jurisdic_wc], bins=np.arange(2.5, 8.5, 1),
             color=[(0.02, 0.36, 0.60, 0.4),
                    (0.9, 0.39, 0.03, 0.4)], density=True,
             label=['Clarin-PL', 'Inne'])
    ax2.set_xlabel(r'\textbf{Liczba słów}')
    ax2.legend(loc='upper left')
    plt.tight_layout()
    fig.savefig('doc/chapters/3.1/data-histogram.png')


def read_times_and_wc(store_path: str) -> tuple:
    with pd.HDFStore(store_path, mode='r') as store:
        references = store['references']
        sizes = references['size'].astype(int)
        times = pd.Series(sizes / 16e3, name='time')    # Divide by sample rate
        transcripts = references['transcript']
        word_counts = transcripts.str.split().map(len)
    return times, word_counts


if __name__ == '__main__':
    clarin_times, clarin_wc = read_times_and_wc('data/train-clarin-1-normalized.hdf5')
    jurisdic_times, jurisdic_wc = read_times_and_wc('data/train-jurisdic-1-normalized.hdf5')
    dump_session([clarin_times, clarin_wc, jurisdic_times, jurisdic_wc])
    clarin_times, clarin_wc, jurisdic_times, jurisdic_wc = load_session()
    save_histogram(times=(clarin_times, jurisdic_times),
                   word_counts=(clarin_wc, jurisdic_wc))
