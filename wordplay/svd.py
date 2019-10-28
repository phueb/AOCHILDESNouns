from typing import Optional, List, Set, Dict
from scipy import sparse
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import pyprind

from categoryeval.probestore import ProbeStore

from wordplay.utils import get_sliding_windows


def make_term_by_window_co_occurrence_mat(prep,
                                          tokens: Optional[List[str]] = None,
                                          start: Optional[int] = None,
                                          end: Optional[int] = None,
                                          window_size: Optional[int] = 7,
                                          max_frequency: int = 1000 * 1000,
                                          log: bool = True,
                                          probe_store: Optional[ProbeStore] = None):
    """
    terms are in cols, windows are in rows.
    y_words are windows, x_words are terms.
    y_words always occur after x_words in the input.
    this format matches that used in TreeTransitions
    """

    print(f'Making term-window matrix'
          f' from tokens between {start:,} & {end:,}')

    # tokens
    if tokens is None:
        if start is not None and end is not None:
            tokens = prep.store.tokens[start:end]
        else:
            raise ValueError('Need either "tokens" or "start" and "end".')

    # x_words
    if probe_store is not None:
        print('WARNING: Using only probes as x-words')
        x_words = probe_store.types
    else:
        x_words = sorted(set(tokens))
    num_xws = len(x_words)
    xw2col_id = {t: n for n, t in enumerate(x_words)}

    # windows
    windows_in_order = get_sliding_windows(window_size, tokens)
    unique_windows = sorted(set(windows_in_order))
    num_unique_windows = len(unique_windows)
    window2row_id = {w: n for n, w in enumerate(unique_windows)}

    # make sparse matrix (y_words in rows, windows in cols)
    shape = (num_unique_windows, num_xws)
    print(f'shape={shape}')
    data = []
    row_ids = []
    cold_ids = []
    mat_loc2freq = {}  # needed to keep track of freq
    for n, window in enumerate(windows_in_order[:-window_size]):
        # row_id + col_id
        row_id = window2row_id[window]
        next_window = windows_in_order[n + 1]
        next_word = next_window[-1]  # -1 is correct because windows slide by 1 word
        try:
            col_id = xw2col_id[next_word]
        except KeyError:  # using probe_store
            continue
        # freq
        try:
            freq = min(max_frequency, mat_loc2freq[(row_id, col_id)])
        except KeyError:
            mat_loc2freq[(row_id, col_id)] = 1
            freq = 1
        else:
            mat_loc2freq[(row_id, col_id)] += 1
        # collect
        row_ids.append(row_id)
        cold_ids.append(col_id)
        data.append(freq)

    if log:
        data = np.log(data)

    # make sparse matrix once (updating it is expensive)
    res = sparse.csr_matrix((data, (row_ids, cold_ids)), shape=shape)

    print('term-by-window co-occurrence matrix has sum={:,}'.format(res.sum()))

    y_words = unique_windows
    return res, x_words, y_words


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
                               num_dims: int = 256,
                               nominal_alpha: float = 0.01,
                               plot_loadings: bool = False,
                               verbose: bool = False,
                               ):
    """
    collect p-value for each singular dimension for plotting
    """

    adj_alpha = nominal_alpha / num_dims
    categories = cat2words.keys()

    if not verbose:
        pbar = pyprind.ProgBar(num_dims, stream=2, title='Decoding')

    cat2ps = {cat: [] for cat in categories}
    dim_ids = []
    for dim_id in range(num_dims):
        if verbose:
            print()
            print(f'Singular Dimension={num_dims - dim_id}')

        dimension = u[:, dim_id]

        for cat in categories:
            category_words = cat2words[cat]

            # non-parametric analysis of variance.
            # is variance between category words and random words different?
            groups = [[v for v, w in zip(dimension, x_words) if w in category_words],
                      [v for v, w in zip(dimension, x_words) if w not in category_words]]
            _, p = stats.kruskal(*groups)
            cat2ps[cat].append(p)

            if verbose:
                print(p)
                print(f'Dimension encodes {cat}= {p < adj_alpha}')

            # inspect how category words actually differ in their loadings from other words
            if p < adj_alpha and plot_loadings:
                inspect_loadings(x_words, dimension, category_words, cat2words['random'], dim_id)

            if p < adj_alpha and cat != 'random':
                dim_ids.append(dim_id)

        if not verbose:
            pbar.update()

    # a dimension cannot encode both nouns and verbs - so chose best
    cat2y = {cat: [] for cat in categories}
    for ps_at_sd in zip(*[cat2ps[cat] for cat in categories]):
        values = np.array(ps_at_sd)  # allows item assignment
        bool_ids = np.where(values < adj_alpha)[0]

        # in case the dimension encodes more than 1 category, only allow 1 winner
        # by setting all but lowest value to np.nan
        if len(bool_ids) > 1:
            min_i = np.argmin(ps_at_sd).item()
            values = [v if i == min_i else np.nan for i, v in enumerate(ps_at_sd)]
            print(f'WARNING: Dimension encodes multiple categories')

        # collect
        for n, cat in enumerate(categories):
            cat2y[cat].append(0.02 * n if values[n] < adj_alpha else np.nan)

    return cat2y, dim_ids