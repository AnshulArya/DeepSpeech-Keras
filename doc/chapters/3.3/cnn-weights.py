import h5py
import pandas as pd
import numpy as np
from tqdm import tqdm
from source.utils import load
from matplotlib import pyplot as plt


if __name__ == '__main__':
    deepspeech = load('models/2019-03-17/07')
    cnn_layer = deepspeech.model.get_layer('conv2d_1')
    weights, bias = cnn_layer.get_weights()
    weights = np.squeeze(weights, axis=2)

    with pd.HDFStore('models/2019-03-17/07/evaluation-jurisdic.hdf5', mode='r') as store:
        references = store['references']

    with h5py.File('models/2019-03-17/07/evaluation-jurisdic.hdf5', mode='r') as store:
        output_index = 4
        activations = np.concatenate([store[f'outputs/{output_index}/{sample_id}']
                                      for sample_id in tqdm(references.index)])
        variances = pd.Series(activations.var(axis=0)).sort_values(ascending=False)

    vector = pd.concat([variances[:17], variances[-1:]]).index.values
    fig, axs = plt.subplots(figsize=(18, 6), nrows=3, ncols=6,
                            subplot_kw={'xticks': [], 'yticks': []})
    for i, ax in zip(vector, axs.flat):
        ax.imshow(weights[:, :, i], aspect='auto', origin='lower', interpolation='lanczos')
    plt.tight_layout()
    fig.savefig(f'doc/chapters/3.3/cnn-weights.png')
