import os
import dill
import numpy as np
from matplotlib import pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
# import seaborn as sns
# cmap = plt.get_cmap('YlGnBu')
# linear = sns.color_palette("YlGnBu")
# sns.palplot(linear)
# plt.show()


def load(file_path: str):
    with open(file_path, mode='rb') as file:
        return dill.load(file)


def plot_results(results_path: str):
    fig, ax = plt.subplots(figsize=(9, 3))
    data = load(results_path)
    epochs, loss, val_loss = np.array([epoch[:-1] for epoch in data]).T
    batch_results = np.array([epoch[-1] for epoch in data[1:]]).flatten()
    epochs = epochs.astype(int) + 1

    ax.plot(epochs, loss, label='CTC loss')
    ax.plot(epochs, val_loss, label=f'CTC dev loss')

    min_index = np.argmin(val_loss)
    min_value = val_loss[min_index]
    ax.scatter(min_index+1, min_value, color='red', label=f'Min dev loss ({val_loss[min_index]:.2f})')

    y_range = np.append(loss, val_loss)
    for y in np.arange(5, max(y_range), step=5):
        ax.axhline(y, xmin=0.0, xmax=1.0, alpha=0.1, color='black', linestyle='--', linewidth=1)

    ax.set_ylim(bottom=0, top=max(y_range) + 5)
    ax.set_xlabel('Epoch')
    ax.set_xticks(epochs)
    ax.set_ylabel('CTC Loss')
    ax.legend(loc='upper right')

    new_ax = ax.twiny()
    new_ax.plot(batch_results, alpha=0.2, linewidth=.05)
    new_ax.set_xticks([])

    plt.tight_layout()
    dir_name = os.path.dirname(results_path)
    fig.savefig(os.path.join(dir_name, 'results.png'))


if __name__ == '__main__':
    plot_results(results_path='models/2019-03-17/09/results.bin')
