"""
Research questions:
1. Are same-category probe words more substitutable in the first or second half of the input?
"""

import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import attr
import numpy as np
import pyprind
import seaborn as sns

from preppy.legacy import TrainPrep
from categoryeval.probestore import ProbeStore

from wordplay.params import PrepParams
from wordplay.docs import load_docs

# /////////////////////////////////////////////////////////////////

CORPUS_NAME = 'childes-20180319'
PROBES_NAME = 'sem-4096'

SHUFFLE_DOCS = False
NUM_MID_TEST_DOCS = 100

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
CONTEXT_DISTANCES = [3]
Y_MAX = 0.5


def make_kld2locations(ps, tokens, distance):
    print('Collecting information about probe context locations...')
    cat2probes_expected_probabilities = {cat: np.array([1 / len(ps.cat2probes[cat])
                                                        if probe in ps.cat2probes[cat] else 0.0
                                                        for probe in ps.types])
                                         for cat in ps.cats}
    # context_d
    context_d = {}
    pbar = pyprind.ProgBar(len(tokens))
    for loc, token in enumerate(tokens[:-distance]):
        context = tuple(tokens[loc + d] for d in range(-distance, 0) if d != 0)
        if token in ps.types:
            try:
                cat = ps.probe2cat[token]
                context_d[context]['freq_by_probe'][token] += 1
                context_d[context]['freq_by_cat'][cat] += 1
                context_d[context]['total_freq'] += 1
                context_d[context]['term_freq'] += 0
                context_d[context]['probe_freq'] += 1
                context_d[context]['locations'].append(loc)
            except KeyError:
                context_d[context] = {'freq_by_probe': {probe: 0.0 for probe in ps.types},
                                      'freq_by_cat': {cat: 0.0 for cat in ps.cats},
                                      'total_freq': 0,
                                      'term_freq': 0,
                                      'probe_freq': 0,
                                      'locations': [],
                                      }
        else:
            try:
                context_d[context]['term_freq'] += 1
            except KeyError:  # only update contexts which are already tracked
                pass
        pbar.update()

    # result
    print('Calculating KL divergences...')
    result = {}
    for context in context_d.keys():
        context_freq = context_d[context]['total_freq']
        if context_freq > MIN_CONTEXT_FREQ:
            # observed
            probes_observed_probabilities = np.array(
                [context_d[context]['freq_by_probe'][probe] / context_d[context]['total_freq']
                 for probe in ps.types])
            # compute KL div for each category that the context is associated with (not just most common category)
            for cat, cat_freq in context_d[context]['freq_by_cat'].items():
                if cat_freq > MIN_CAT_FREQ:
                    probes_expected_probabilities = cat2probes_expected_probabilities[cat]
                    kl = calc_kl_divergence(probes_expected_probabilities, probes_observed_probabilities)  # asymmetric
                    # collect
                    result[kl] = context_d[context]['locations']
    return result


def calc_kl_divergence(p, q, epsilon=0.00001):
    pe = p + epsilon
    qe = q + epsilon
    divergence = np.sum(pe * np.log2(pe / qe))
    return divergence


# plot results
for context_dist in CONTEXT_DISTANCES:

    kld2locations = make_kld2locations(probe_store, prep.store.tokens, context_dist)

    # fig
    fontsize = 16
    _, ax = plt.subplots(figsize=(6, 6))
    ax.set_title('context-size={}'.format(context_dist), fontsize=fontsize)
    ax.set_ylabel('Probability', fontsize=fontsize)
    ax.set_xlabel('KL Divergence', fontsize=fontsize)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='both', top=False, right=False)
    ax.set_ylim([0, Y_MAX])
    # plot
    colors = sns.color_palette("hls", 2)[::-1]
    y1 = []
    y2 = []
    for kld, locations in kld2locations.items():
        num_locations_in_part1 = len(np.where(np.array(locations) < prep.midpoint)[0])
        num_locations_in_part2 = len(np.where(np.array(locations) > prep.midpoint)[0])
        y1 += [kld] * num_locations_in_part1
        y2 += [kld] * num_locations_in_part2
    y1 = np.array(y1)
    y2 = np.array(y2)
    num_bins = 20
    y1binned, x1, _ = ax.hist(y1, density=True, label='partition 1', color=colors[0], histtype='step',
                              bins=num_bins, range=[0, 12], zorder=3)
    y2binned, x2, _ = ax.hist(y2, density=True, label='partition 2', color=colors[1], histtype='step',
                              bins=num_bins, range=[0, 12], zorder=3)
    ax.text(0.02, 0.7, 'partition 1:\nmean={:.2f}+/-{:.1f}\nn={:,}'.format(
        np.mean(y1), np.std(y1), len(y1)), transform=ax.transAxes, fontsize=fontsize - 2)
    ax.text(0.02, 0.55, 'partition 2:\nmean={:.2f}+/-{:.1f}\nn={:,}'.format(
        np.mean(y2), np.std(y2), len(y2)), transform=ax.transAxes, fontsize=fontsize - 2)
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
    plt.legend(frameon=False, loc='upper left', fontsize=fontsize)
    plt.tight_layout()
    plt.show()

    # t test
    t, prob = ttest_ind(y1, y2, equal_var=False)
    print('t={}'.format(t))
    print('p={:.6f}'.format(prob))
    print()

