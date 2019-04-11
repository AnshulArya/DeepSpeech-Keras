import os
import pickle
import operator
from collections import defaultdict
from functools import reduce

import h5py
import pandas as pd
import numpy as np
import seaborn as sns
from joblib import Memory
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib import pyplot as plt

from text import Alphabet

cache_dir = os.path.join(os.path.dirname(__file__), 'cachedir')
memory = Memory(cache_dir, verbose=0)


def dump_session(data):
    with open(f'{__file__}.session', 'wb') as file:
        pickle.dump(data, file)


def add_vectors(softmax, lstm, clusters):
    letter_indices = np.argmax(softmax.value, axis=1)
    for i, letter_index in enumerate(letter_indices):
        prob = softmax[i, letter_index]
        if prob > 0.95:
            vector = lstm.value[i, :]
            clusters[letter_index].append(vector)


@memory.cache
def calculate_clusters():
    """
    Define LSTM vectors properties.
    """
    with pd.HDFStore('models/2019-03-17/07/evaluation-jurisdic.hdf5', mode='r') as store:
        references = store['references']
        correct_predicted = references[references.cer == 0]

    clusters = defaultdict(list)
    cluster_means = {}
    with h5py.File('models/2019-03-17/07/evaluation-jurisdic.hdf5', mode='r') as store:
        lstm_index, softmax_index = 13, 14
        for sample_id in tqdm(correct_predicted.index):
            lstm = store[f'outputs/{lstm_index}/{sample_id}']

            softmax = store[f'outputs/{softmax_index}/{sample_id}']
            add_vectors(softmax, lstm, clusters)

    for letter in clusters.keys():
        mean_vector = np.array(clusters[letter]).mean(axis=0)
        cluster_means[letter] = mean_vector
    return clusters, cluster_means


def main():
    alphabet = Alphabet('tests/models/test/alphabet.txt')
    clusters, cluster_means = calculate_clusters()
    cluster_activations = reduce(operator.add, (v for k, v in clusters.items()))
    dump_session(cluster_activations)

    df = pd.DataFrame(cluster_means)
    df = df.reindex(sorted(df.columns), axis=1)
    ticks = [alphabet.string_from_label(label) for label in df.columns]
    ticks[0], ticks[-1] = '_', '$'

    centers = df.values.T
    distances = 1 - cosine_similarity(centers)

    fig, ax = plt.subplots(figsize=(15, 15))
    ax = sns.heatmap(distances, square=True, xticklabels=ticks, yticklabels=ticks,
                     annot=False, linewidths=.1, ax=ax, fmt='d', cmap="YlGnBu")
    plt.tight_layout()
    file_name = os.path.join(os.path.dirname(__file__), 'lstm-activation-cluster-distances.svg')
    fig.savefig(file_name, format='svg', dpi=1200)


if __name__ == '__main__':
    main()
