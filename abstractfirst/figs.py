import numpy as np
from matplotlib import pyplot as plt
from typing import Optional, Tuple, List, Dict
from pathlib import Path

from abstractfirst import configs


def make_line_fig(label2y: Dict[str, List[float]],
                  title: str,
                  x_axis_label: str,
                  y_axis_label: str,
                  x_ticks: List[int],
                  x_tick_labels: Optional[List[str]] = None,
                  y_lims: Optional[List[float]] = None,
                  ):
    fig, ax = plt.subplots(1, figsize=(6, 4), dpi=163)
    plt.title(title, fontsize=configs.Figs.title_font_size)
    ax.set_ylabel(y_axis_label, fontsize=configs.Figs.ax_font_size)
    ax.set_xlabel(x_axis_label, fontsize=configs.Figs.ax_font_size)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_tick_labels or x_ticks, fontsize=configs.Figs.tick_font_size)
    ax.yaxis.grid(True)
    if y_lims:
        ax.set_ylim(y_lims)

    # plot
    for label, y in label2y.items():
        ax.plot(x_ticks, y, linewidth=2, label=label)

    plt.legend()

    return fig, ax


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
