import os
import numpy as np
import seaborn as sns
import pandas as pd
import pickle
from source.utils import load
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

from text import Alphabet


def load_session(name):
    with open(name, 'rb') as file:
        return pickle.load(file)


def calculate_variances(matrix):
    pca = PCA()
    pca.fit(matrix)
    return pca.explained_variance_ratio_


if __name__ == '__main__':
    # deepspeech = load('models/2019-03-17/07')
    # softmax_layer = deepspeech.model.layers[-1]
    # weights, bias = softmax_layer.get_weights()
    # vector_size, letter_num = weights.shape
    # softmax_pca_var = calculate_variances(weights.T)

    cluster_activations = load_session('doc/chapters/3.5/lstm-activation-cluster-distances.py.session')
    lstm_pca_var = calculate_variances(cluster_activations)

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.plot(np.cumsum(lstm_pca_var))
    plt.show()




    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')






    alphabet = Alphabet('tests/models/test/alphabet.txt')
    ticks = [alphabet.string_from_label(label) for label in range(letter_num)]
    ticks[0], ticks[-1] = '_', '$'

    distances = 1 - cosine_similarity(weights.T)
    fig, ax = plt.subplots(figsize=(15, 15))
    ax = sns.heatmap(distances, square=True, xticklabels=ticks, yticklabels=ticks,
                     annot=False, linewidths=.1, ax=ax, fmt='d', cmap="YlGnBu")
    plt.tight_layout()
    file_name = os.path.join(os.path.dirname(__file__), 'softmax-distances.svg')
    fig.savefig(file_name, format='svg', dpi=1200)
