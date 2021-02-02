import numpy as np
from matplotlib import pyplot as plt
from typing import Optional, Tuple
from pathlib import Path

from abstractfirst import configs


def plot_heatmap(mat: np.ndarray,
                 y_tick_labels: Optional[list] = None,
                 x_tick_labels: Optional[list] = None,
                 label_interval: int = 10,
                 save_path: Optional[Path] = None,
                 title: str = '',
                 vmin: Optional[float] = None,
                 vmax: Optional[float] = None,
                 figsize: Tuple[int, int] = (6, 2)
                 ):

    if y_tick_labels is None:
        y_tick_labels = []
    if x_tick_labels is None:
        x_tick_labels = []

    fig, ax = plt.subplots(figsize=figsize, dpi=configs.Fig.dpi)
    plt.title(title, fontsize=3)

    # heatmap
    # print('Plotting heatmap...')
    ax.imshow(mat,
              aspect='equal',
              cmap=plt.get_cmap('viridis'),
              interpolation='nearest',
              vmin=vmin,
              vmax=vmax,
              )

    # x ticks
    x_tick_labels_spaced = []
    for i, l in enumerate(x_tick_labels):
        x_tick_labels_spaced.append(l if i % label_interval == 0 else '')

    num_cols = len(mat.T)
    ax.set_xticks(np.arange(num_cols))
    ax.xaxis.set_ticklabels(x_tick_labels_spaced, rotation=90, fontsize=1)

    # y ticks
    y_tick_labels_spaced = []
    for i, l in enumerate(y_tick_labels):
        y_tick_labels_spaced.append(l if i % label_interval == 0 else '')

    num_rows = len(mat)
    ax.set_yticks(np.arange(num_rows))
    ax.yaxis.set_ticklabels(y_tick_labels_spaced,  # no need to reverse (because no extent is set)
                            rotation=0, fontsize=1)

    # remove tick lines
    lines = (ax.xaxis.get_ticklines() +
             ax.yaxis.get_ticklines())
    plt.setp(lines, visible=False)
    plt.show()

    # save
    if save_path:
        fig.savefig(save_path, format='png')
