"""
Research questions:
1. Do contexts in partition 1 of AO-CHILDES contain category members more exclusively?
2. Are contexts more "pure" in the sense that they are ess contaminated by non-category words?

Ideally, the measure in question quantifies the probability of a context
re-occurring with a category member given it has occurred with a category member once before
"""

import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import attr
import numpy as np
import pyprind
import seaborn as sns
from typing import Set, List, Dict, Tuple, Any

from preppy.legacy import TrainPrep
from categoryeval.probestore import ProbeStore

from wordplay import config
from wordplay.word_sets import excluded
from wordplay.params import PrepParams
from wordplay.docs import load_docs
from wordplay.memory import set_memory_limit

# /////////////////////////////////////////////////////////////////

CORPUS_NAME = 'childes-20180319'
PROBES_NAME = 'sem-all'
NUM_TYPES = 4096  # a small vocabulary is needed to compute purity, otherwise vocabulary is too large

docs = load_docs(CORPUS_NAME)

params = PrepParams(num_types=NUM_TYPES)
prep = TrainPrep(docs, **attr.asdict(params))

probe_store = ProbeStore(CORPUS_NAME, PROBES_NAME, prep.store.w2id, excluded=excluded)

# /////////////////////////////////////////////////////////////////

CONTEXT_SIZES = [1]
MEASURE_NAME = 'purity'


def make_context2info(probes: Set[str],
                      tokens: List[str],
                      distance: int,
                      ) -> Dict[Tuple[str], Dict[str, Any]]:
    """
    collect information about contexts across entire corpus.
    """
    print('Collecting information about probe context locations...')

    res = {}
    pbar = pyprind.ProgBar(len(tokens))
    for loc, token in enumerate(tokens[:-distance]):
        context = tuple(tokens[loc + d] for d in range(-distance, 0) if d != 0)
        res.setdefault(context, {}).setdefault('locations', []).append(loc)

        if token in probes:
            res.setdefault(context, {}).setdefault('in-category', []).append(True)
        else:
            res.setdefault(context, {}).setdefault('in-category', []).append(False)

        pbar.update()

    return res


def make_tuples(d, expected_prob):
    """
    generate tuples like [(y, locations), (y, locations, ...]

    y is the measure of interest; it is a ratio of two probabilities.
    the numerator is the observed probability of category contexts co-occurring with category member
    the denominator is the expected probability of category contexts co-occurring with category member
    """
    for context in d.keys():

        # context must occur with category member at least once
        is_category_context = np.any(d[context]['in-category'])

        # TODO the expected probability should be conditioned on the fact that the context
        #  has co-occurred with a category member once already

        if is_category_context:
            observed_prob = np.sum(d[context]['in-category']) / len(d[context])
            y = observed_prob / expected_prob
            yield y, d[context]['locations']


set_memory_limit(prop=0.9)


headers = ['Category', 'partition', 'context-size', MEASURE_NAME, 'n']
name2col = {name: [] for name in headers}
cat2context_size2p = {cat: {} for cat in probe_store.cats}
for cat in probe_store.cats:
    cat_probes = probe_store.cat2probes[cat]

    # the probability that a context occurs with a category member
    num_cat_tokens = sum([prep.store.w2f[p] for p in cat_probes])
    in_cat_prob = num_cat_tokens / prep.store.num_tokens
    print(f'Probability of in-category context co-occurrence={in_cat_prob:.4f}')

    for context_size in CONTEXT_SIZES:

        # compute measure for contexts associated with a single category
        try:
            context2info = make_context2info(cat_probes, prep.store.tokens, context_size)
        except MemoryError:
            raise SystemExit('Reached memory limit')

        # separate measure by location
        y1 = []
        y2 = []
        for y, locations in make_tuples(context2info, in_cat_prob):
            num_locations_in_part1 = len(np.where(np.array(locations) < prep.midpoint)[0])
            num_locations_in_part2 = len(np.where(np.array(locations) > prep.midpoint)[0])
            y1 += [y] * num_locations_in_part1
            y2 += [y] * num_locations_in_part2
        y1 = np.array(y1)
        y2 = np.array(y2)

        # t test
        t, prob = ttest_ind(y1, y2, equal_var=True)
        cat2context_size2p[cat][context_size] = prob
        print('t={}'.format(t))
        print('p={:.6f}'.format(prob))
        print()

        # populate tabular data
        for name, di in zip(headers, (cat, 1, context_size, np.mean(y1), len(y1))):
            name2col[name].append(di)
        for name, di in zip(headers, (cat, 2, context_size, np.mean(y2), len(y2))):
            name2col[name].append(di)

        NUM_BINS = None
        X_RANGE = None
        _, ax = plt.subplots(figsize=config.Fig.fig_size, dpi=config.Fig.dpi)
        ax.set_title('context-size={}'.format(context_size), fontsize=config.Fig.ax_fontsize)
        ax.set_ylabel('Probability', fontsize=config.Fig.ax_fontsize)
        ax.set_xlabel(MEASURE_NAME, fontsize=config.Fig.ax_fontsize)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', which='both', top=False, right=False)
        # ax.set_ylim([0, Y_MAX])
        # plot
        colors = sns.color_palette("hls", 2)[::-1]
        y1binned, x1, _ = ax.hist(y1, density=True, label='partition 1', color=colors[0], histtype='step',
                                  bins=NUM_BINS, range=X_RANGE, zorder=3)
        y2binned, x2, _ = ax.hist(y2, density=True, label='partition 2', color=colors[1], histtype='step',
                                  bins=NUM_BINS, range=X_RANGE, zorder=3)
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
        plt.show()