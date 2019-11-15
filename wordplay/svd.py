from typing import Optional, List, Set, Dict
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import pyprind
import seaborn as sns
from sortedcontainers import SortedSet

from wordplay import config


def plot_category_encoding_dimensions(cat2dim_ids: Dict[str, List[int]],
                                      num_dims: int,
                                      title: Optional[str] = '',
                                      ):
    y_offset = 0.02
    categories = SortedSet(cat2dim_ids.keys())
    num_categories = len(categories)

    # scatter plot
    _, ax = plt.subplots(dpi=192, figsize=(6, 6))
    ax.set_title(title, fontsize=config.Fig.ax_fontsize)
    # axis
    ax.set_xlabel('Singular Dimension', fontsize=config.Fig.ax_fontsize)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='both', top=False, right=False, left=False)
    ax.set_yticks([])
    ax.set_xlim(left=0, right=num_dims)
    ax.set_ylim([-y_offset, y_offset * num_categories + y_offset])
    # plot
    x = np.arange(num_dims)
    cat2color = {n: c for n, c in zip(categories, sns.color_palette("hls", num_categories))}
    for n, cat in enumerate(categories):
        cat_dim_ids = cat2dim_ids[cat][::-1]
        y = [n * y_offset if not np.isnan(dim_id) else np.nan for dim_id in cat_dim_ids]
        color = cat2color[cat] if cat != 'random' else 'black'
        color = 'white' if np.all(np.isnan(y)) else color
        ax.scatter(x, y, color=color, label=cat.upper())
        print(f'{np.count_nonzero(~np.isnan(y))} dimensions encode {cat:<12}')

    plt.legend(frameon=True, framealpha=1.0, bbox_to_anchor=(0.5, 1.4), ncol=4, loc='lower center')
    plt.show()


def inspect_loadings(x_words, dimension, category_words, random_words, dim_id):
    num_x = len(x_words)

    _, ax = plt.subplots(dpi=192, figsize=(6, 6))
    ax.set_title(f'Singular dimension {dim_id}')
    x = np.arange(num_x)
    ax.scatter(x, dimension, color='grey')
    loadings = [v if w in category_words else np.nan for v, w in zip(dimension, x_words)]
    ax.scatter(x, loadings, color='red')
    ax.axhline(y=np.nanmean(loadings), color='red', zorder=3)
    plt.show()

    _, ax = plt.subplots(dpi=192, figsize=(6, 6))
    ax.set_title(f'Singular dimension {dim_id}')
    x = np.arange(num_x)
    ax.scatter(x, dimension, color='grey')
    loadings = [v if w in random_words else np.nan for v, w in zip(dimension, x_words)]
    ax.scatter(x, loadings, color='blue')
    ax.axhline(y=np.nanmean(loadings), color='blue', zorder=3)
    plt.show()

    # qq plot
    _, ax2 = plt.subplots(1)
    stats.probplot(dimension, plot=ax2)
    plt.show()


def decode_singular_dimensions(u: np.ndarray,
                               cat2words: Dict[str, Set[str]],
                               x_words: List[str],
                               control_words: Optional[Set[str]] = None,
                               num_dims: int = 256,
                               nominal_alpha: float = 0.01,
                               plot_loadings: bool = False,
                               verbose: bool = False,
                               ):
    """
    Collect singular dimension IDs which have been identified as encoding a category.
    Each dimension is allowed to encode only one category.

    WARNING: the dimension IDs are not ordered by descending amount of variance accounted for.
    They are ordered by increasing amount of variance accounted for.
    This means that larger IDs correspond to more dimensions accounting for more variance.
    """

    adj_alpha = nominal_alpha / num_dims
    categories = cat2words.keys()

    if not verbose:
        pbar = pyprind.ProgBar(num_dims, stream=2, title='Decoding')

    cat2ps = {cat: [] for cat in categories}
    for dim_id in range(num_dims):
        if verbose:
            print()
            print(f'Singular Dimension={num_dims - dim_id}')

        dimension = u[:, dim_id]

        for cat in categories:
            category_words = cat2words[cat]

            # control words are useful when semantic categories are tested, containing only nouns.
            # instead of comparing category_words to all other words, compare to non-category nouns instead
            if control_words is None:
                groups = [[v for v, w in zip(dimension, x_words) if w in category_words],
                          [v for v, w in zip(dimension, x_words) if w not in category_words]]
            else:
                groups = [[v for v, w in zip(dimension, x_words) if w in category_words],
                          [v for v, w in zip(dimension, x_words)
                           if w in control_words and w not in category_words]]

            # non-parametric analysis of variance.
            # is variance between category words and random words different?
            _, p = stats.kruskal(*groups)
            cat2ps[cat].append(p)

            if verbose:
                print(p)
                print(f'Dimension encodes {cat}= {p < adj_alpha}')

            # inspect how category words actually differ in their loadings from other words
            if p < adj_alpha and plot_loadings:
                inspect_loadings(x_words, dimension, category_words, cat2words['random'], dim_id)

        if not verbose:
            pbar.update()

    # a dimension is not allowed to encode multiple categories - so chose best
    cat2dim_ids = {cat: [] for cat in categories}
    for dim_id, ps_at_dim in enumerate(zip(*[cat2ps[cat] for cat in categories])):
        values = np.array(ps_at_dim)  # allows item assignment
        bool_ids = np.where(values < adj_alpha)[0]

        # in case the dimension encodes more than 1 category, only allow 1 winner
        # by setting all but lowest value to np.nan
        if len(bool_ids) > 1:
            min_i = np.argmin(ps_at_dim).item()
            values = np.array([v if i == min_i else np.nan for i, v in enumerate(ps_at_dim)])
            print(f'WARNING: Dimension encodes multiple categories')

        num_significant = len(np.where(values < adj_alpha)[0])
        assert num_significant <= 1  # only one dimension can have p < alpha

        # collect
        for n, cat in enumerate(categories):
            cat2dim_ids[cat].append(dim_id if values[n] < adj_alpha else np.nan)

    return cat2dim_ids