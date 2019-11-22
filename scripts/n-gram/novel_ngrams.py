import numpy as np
import pyprind
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import attr

from preppy.legacy import TrainPrep

from wordplay import config
from wordplay.params import PrepParams
from wordplay.docs import load_docs
from wordplay.utils import human_format
from wordplay.utils import get_sliding_windows

# /////////////////////////////////////////////////////////////////

CORPUS_NAME = 'childes-20180319'

docs = load_docs(CORPUS_NAME)

params = PrepParams()
prep = TrainPrep(docs, **attr.asdict(params))

# /////////////////////////////////////////////////////////////////

NUM_BINS = 32
NGRAM_SIZES = [2, 3]


def make_novel_xys(n_grams):
    """
    Return a list of (x, y) coordinates for plotting
    """
    # trajectories
    seen = set()
    num_ngrams = len(n_grams)
    pbar = pyprind.ProgBar(num_ngrams, stream=2, title='Tracking seen and novel n-grams')
    trajectory = []
    for ng in n_grams:
        if ng not in seen:
            trajectory.append(1)
            seen.add(ng)
        else:
            trajectory.append(np.nan)
        pbar.update()
    # res
    ns = np.where(np.array(trajectory) == 1)[0]
    hist, b = np.histogram(ns, bins=NUM_BINS, range=[0, num_ngrams])
    res = (b[:-1], hist)
    return res


# size2novel_xys
num_ngram_sizes = len(NGRAM_SIZES)
size2novel_xys1 = {}
size2novel_xys2 = {}
for ngram_size in NGRAM_SIZES:
    ngram_range = (ngram_size, ngram_size)
    ngrams = get_sliding_windows(ngram_size, prep.store.tokens)
    xys1 = make_novel_xys(ngrams)
    xys2 = make_novel_xys(ngrams[::-1])
    size2novel_xys1[ngram_size] = xys1
    size2novel_xys2[ngram_size] = xys2

# fig
fig, axs = plt.subplots(num_ngram_sizes, 1, sharex='all', dpi=config.Fig.dpi, figsize=None)
if num_ngram_sizes == 1:
    axs = [axs]
for ax, ngram_size in zip(axs, NGRAM_SIZES):
    if ax == axs[-1]:
        ax.tick_params(axis='both', which='both', top=False, right=False)
        ax.set_ylabel('Corpus Location', fontsize=config.Fig.ax_fontsize)
    else:
        ax.tick_params(axis='both', which='both', top=False, right=False, bottom='off')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_major_formatter(FuncFormatter(human_format))
    ax.set_ylabel('Novel {}-grams'.format(ngram_size), fontsize=config.Fig.ax_fontsize)
    ax.yaxis.grid(True)
    # plot
    ax.plot(*size2novel_xys1[ngram_size], linestyle='-', label='age-ordered')
    ax.plot(*size2novel_xys2[ngram_size], linestyle='-', label='reverse age-ordered')
plt.legend(framealpha=1.0)
plt.tight_layout()
plt.show()


