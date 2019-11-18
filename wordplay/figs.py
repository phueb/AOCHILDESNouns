import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from typing import Optional, List, Dict

from wordplay import config


def plot_heatmap(mat,
                 y_tick_labels,
                 x_tick_labels,
                 label_interval: int = 10,
                 save_name=None):
    fig, ax = plt.subplots(figsize=(10, 10), dpi=163 * 2)
    plt.title('', fontsize=5)

    # heatmap
    print('Plotting heatmap...')
    ax.imshow(mat,
              aspect='equal',
              cmap=plt.get_cmap('jet'),
              interpolation='nearest')

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
                            rotation=0, fontsize=2)

    # remove tick lines
    lines = (ax.xaxis.get_ticklines() +
             ax.yaxis.get_ticklines())
    plt.setp(lines, visible=False)
    plt.show()

    # save
    if save_name:
        fig.savefig(f'heatmap_{save_name}.svg', format='svg')  # TODO test


def make_histogram(y1: np.ndarray,
                   y2: np.ndarray,
                   x_label: str,
                   label1: str = 'partition 1',
                   label2: str = 'partition 2',
                   title: str = '',
                   y_max: Optional[float] = None,
                   num_bins: Optional[int] = None,
                   x_range: Optional[List[int]] = None,
                   ) -> plt.figure:
    fig, ax = plt.subplots(figsize=config.Fig.fig_size, dpi=config.Fig.dpi)
    ax.set_title(title, fontsize=config.Fig.ax_fontsize)
    ax.set_ylabel('Probability', fontsize=config.Fig.ax_fontsize)
    ax.set_xlabel(x_label, fontsize=config.Fig.ax_fontsize)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='both', top=False, right=False)
    ax.set_ylim([0, y_max])
    ax.set_yticklabels([])
    # plot
    colors = sns.color_palette("hls", 2)[::-1]
    y1binned, x1, _ = ax.hist(y1, density=True, label=label1, color=colors[0], histtype='step',
                              bins=num_bins, range=x_range, zorder=3)
    y2binned, x2, _ = ax.hist(y2, density=True, label=label2, color=colors[1], histtype='step',
                              bins=num_bins, range=x_range, zorder=3)
    #  fill between the lines (highlighting the difference between the two histograms)
    for i, x1i in enumerate(x1[:-1]):
        y1line = [y1binned[i], y1binned[i]]
        y2line = [y2binned[i], y2binned[i]]
        ax.fill_between(x=[x1i, x1[i + 1]],
                        y1=y1line,
                        y2=y2line,
                        where=y1line > y2line,
                        color=colors[0],
                        alpha=0.5,
                        zorder=2)
    #
    plt.legend(frameon=False, loc='upper right', fontsize=config.Fig.leg_fontsize)
    plt.tight_layout()

    return fig