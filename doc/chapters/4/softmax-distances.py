import os
import seaborn as sns
from source.utils import load
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

from text import Alphabet


if __name__ == '__main__':
    deepspeech = load('models/2019-03-17/07')
    softmax_layer = deepspeech.model.layers[-1]
    weights, bias = softmax_layer.get_weights()
    vector_size, letter_num = weights.shape

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
