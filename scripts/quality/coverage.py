"""
Research questions:
1. Do contexts in partition 1 of AO-CHILDES cover category members more uniformly?
2. Are same-category members more substitutable in contexts that occur in partition 1?

This measure was previously called within-category substitutability.


How it works:
Each context type is assigned a KL divergence reflecting its "coverage" across the entire corpus.
To obtain a coverage specific to a particular partition,
 the coverage value of each context token which occurs in the partition is obtained, and their values averaged.

Why not compute coverage for each partition separately?
Because this would result in each partition being evaluated on the coverage on a dramatically different set of contexts.
Computing coverage for different sets of contexts would make it difficult to compare coverage between partitions.

"""

import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import attr
import numpy as np
import pyprind
import seaborn as sns

from preppy.legacy import TrainPrep
from categoryeval.probestore import ProbeStore

from wordplay import config
from wordplay.params import PrepParams
from wordplay.docs import load_docs
from wordplay.measures import calc_kl_divergence
from wordplay.memory import set_memory_limit

# /////////////////////////////////////////////////////////////////

CORPUS_NAME = 'childes-20180319'
PROBES_NAME = 'syn-4096'  # TODO what about syntactic categories?

SHUFFLE_DOCS = False
NUM_MID_TEST_DOCS = 0

docs = load_docs(CORPUS_NAME,
                 num_test_take_from_mid=NUM_MID_TEST_DOCS,
                 num_test_take_random=0,
                 shuffle_docs=SHUFFLE_DOCS)

params = PrepParams()
prep = TrainPrep(docs, **attr.asdict(params))

probe_store = ProbeStore(CORPUS_NAME, PROBES_NAME, prep.store.w2id)

# /////////////////////////////////////////////////////////////////


MIN_CONTEXT_FREQ = 10
MIN_CAT_FREQ = 1
CONTEXT_DISTANCES = [1, 2, 3, 4]
POS = 'VERB'  # None to include all categories in probe store - use this to evaluate nouns separately

try:
    types = probe_store.cat2probes[POS]
except KeyError:
    types = probe_store.types


def make_context2info(ps, tokens, distance):
    """
    compute KL divergence for each probe context type across entire corpus.
    keep track of information about each context type
    """
    print('Collecting information about probe context locations...')

    res = {}
    pbar = pyprind.ProgBar(len(tokens))
    for loc, token in enumerate(tokens[:-distance]):
        context = tuple(tokens[loc + d] for d in range(-distance, 0) if d != 0)
        if token in types:
            try:
                cat = ps.probe2cat[token]
                res[context]['freq_by_probe'][token] += 1
                res[context]['freq_by_cat'][cat] += 1
                res[context]['total_freq'] += 1
                res[context]['term_freq'] += 0
                res[context]['probe_freq'] += 1
                res[context]['locations'].append(loc)
            except KeyError:
                res[context] = {'freq_by_probe': {probe: 0.0 for probe in ps.types},
                                'freq_by_cat': {cat: 0.0 for cat in ps.cats},
                                'total_freq': 0,
                                'term_freq': 0,
                                'probe_freq': 0,
                                'locations': [],
                                }
        else:
            try:
                res[context]['term_freq'] += 1
            except KeyError:  # only update contexts which are already tracked
                pass
        pbar.update()

    return res


def make_kld_tuples(d, ps, custom_name):
    """
    return list of tuples like [(kld, locations), (kld, locations, ...] or [(kld, cats), (kld, cats), ...]
    used to filter KL divergences (e.g. by location or category.
    """
    cat2probes_expected_probabilities = {cat: np.array([1 / len(ps.cat2probes[cat])
                                                        if probe in ps.cat2probes[cat] else 0.0
                                                        for probe in ps.types])
                                         for cat in ps.cats}

    print('Calculating KL divergences...')
    for context in d.keys():
        context_freq = d[context]['total_freq']
        if context_freq > MIN_CONTEXT_FREQ:
            # observed
            probes_observed_probabilities = np.array(
                [d[context]['freq_by_probe'][probe] / d[context]['total_freq']
                 for probe in ps.types])
            # compute KL div for each category that the context is associated with (not just most common category)
            for cat, cat_freq in d[context]['freq_by_cat'].items():
                if cat_freq > MIN_CAT_FREQ:
                    probes_expected_probabilities = cat2probes_expected_probabilities[cat]
                    kl = calc_kl_divergence(probes_expected_probabilities, probes_observed_probabilities)  # asymmetric

                    yield (kl, d[context][custom_name])


set_memory_limit(prop=0.9)


for context_dist in CONTEXT_DISTANCES:

    # compute KL values for all probe contexts
    try:
        context2info = make_context2info(probe_store, prep.store.tokens, context_dist)
    except MemoryError:
        raise SystemExit('Reached memory limit')

    # separate kl values by location
    y1 = []
    y2 = []
    for kld, locations in make_kld_tuples(context2info, probe_store, 'locations'):
        num_locations_in_part1 = len(np.where(np.array(locations) < prep.midpoint)[0])
        num_locations_in_part2 = len(np.where(np.array(locations) > prep.midpoint)[0])
        y1 += [kld] * num_locations_in_part1
        y2 += [kld] * num_locations_in_part2
    y1 = np.array(y1)
    y2 = np.array(y2)

    # fig
    Y_MAX = 0.5
    _, ax = plt.subplots(figsize=config.Fig.fig_size, dpi=config.Fig.dpi)
    ax.set_title('context-size={}'.format(context_dist), fontsize=config.Fig.ax_fontsize)
    ax.set_ylabel('Probability', fontsize=config.Fig.ax_fontsize)
    ax.set_xlabel('KL Divergence', fontsize=config.Fig.ax_fontsize)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='both', top=False, right=False)
    ax.set_ylim([0, Y_MAX])
    # plot
    colors = sns.color_palette("hls", 2)[::-1]
    num_bins = 20
    y1binned, x1, _ = ax.hist(y1, density=True, label='partition 1', color=colors[0], histtype='step',
                              bins=num_bins, range=[0, 12], zorder=3)
    y2binned, x2, _ = ax.hist(y2, density=True, label='partition 2', color=colors[1], histtype='step',
                              bins=num_bins, range=[0, 12], zorder=3)
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
    plt.legend(frameon=False, loc='upper left', fontsize=config.Fig.leg_fontsize)
    plt.tight_layout()
    plt.show()

    # separate kl values by category
    cat2klds = {cat: [] for cat in probe_store.cats}
    for kld, cat2f in make_kld_tuples(context2info, probe_store, 'freq_by_cat'):
        for cat, f in cat2f.items():
            cat2klds[cat] += [kld] * int(f)

    for cat, klds in cat2klds.items():
        print(f'{cat:<12}: {np.mean(klds):>9.4f}')

    # console
    print(f'partition 1: mean={np.mean(y1):.2f}+/-{np.std(y1):.1f}\nn={len(y1):,}')
    print(f'partition 2: mean={np.mean(y2):.2f}+/-{np.std(y2):.1f}\nn={len(y2):,}')

    # t test
    t, prob = ttest_ind(y1, y2, equal_var=False)
    print('t={}'.format(t))
    print('p={:.6f}'.format(prob))
    print()

    del context2info

