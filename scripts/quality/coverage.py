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

Coverage for a single category = 1 / mean(klds)
where mean(klds) is mean of all kl-divergences, one for each category context.
"""

import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import attr
import numpy as np
import pyprind
import seaborn as sns
from typing import Dict, Any, Set, List, Tuple
from tabulate import tabulate
import pandas as pd
import pingouin as pg

from preppy.legacy import TrainPrep
from categoryeval.probestore import ProbeStore

from wordplay import config
from wordplay.word_sets import excluded
from wordplay.params import PrepParams
from wordplay.docs import load_docs
from wordplay.measures import calc_kl_divergence
from wordplay.memory import set_memory_limit

# /////////////////////////////////////////////////////////////////

CORPUS_NAME = 'childes-20180319'
PROBES_NAME = 'sem-all'

docs = load_docs(CORPUS_NAME)

params = PrepParams()
prep = TrainPrep(docs, **attr.asdict(params))

probe_store = ProbeStore(CORPUS_NAME, PROBES_NAME, prep.store.w2id, excluded=excluded)

# /////////////////////////////////////////////////////////////////


MIN_CONTEXT_FREQ = 1  # using any more than 1 reduces power to detect differences at context-size=3
CONTEXT_SIZES = [1, 2, 3]
SHOW_HISTOGRAM = False


def make_context2info(probes: Set[str],
                      tokens: List[str],
                      distance: int,
                      ) -> Dict[Tuple[str], Dict[str, Any]]:
    """
    compute KL divergence for each probe context type across entire corpus.
    keep track of information about each context type
    """
    print('Collecting information about probe context locations...')

    res = {}
    pbar = pyprind.ProgBar(len(tokens))
    for loc, token in enumerate(tokens[:-distance]):
        context = tuple(tokens[loc + d] for d in range(-distance, 0) if d != 0)
        if token in probes:
            try:
                res[context]['freq_by_probe'][token] += 1
                res[context]['total_freq'] += 1
                res[context]['term_freq'] += 0
                res[context]['probe_freq'] += 1
                res[context]['locations'].append(loc)
            except KeyError:
                res[context] = {'freq_by_probe': {probe: 0.0 for probe in probes},
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


def make_kld_tuples(d, probes):
    """
    generate tuples like [(kld, locations), (kld, locations, ...]
    """
    probes_expected_probabilities = np.array([1 / len(probes) for _ in probes])

    print('Calculating KL divergences...')
    for context in d.keys():
        context_freq = d[context]['total_freq']
        if context_freq > MIN_CONTEXT_FREQ:

            # kld
            probes_observed_probabilities = np.array(
                [d[context]['freq_by_probe'][probe] / d[context]['total_freq']
                 for probe in probes])
            kl = calc_kl_divergence(probes_expected_probabilities, probes_observed_probabilities)  # asymmetric

            yield kl, d[context]['locations'], context


set_memory_limit(prop=0.9)

headers = ['Category', 'partition', 'context-size', 'coverage', 'n']
name2col = {name: [] for name in headers}
cat2context_size2p = {cat: {} for cat in probe_store.cats}
for cat in probe_store.cats:
    cat_probes = probe_store.cat2probes[cat]

    for context_size in CONTEXT_SIZES:

        # compute KL values for contexts associated with a single category
        try:
            context2info = make_context2info(cat_probes, prep.store.tokens, context_size)
        except MemoryError:
            raise SystemExit('Reached memory limit')

        # separate kl values by location
        y1 = []
        y2 = []
        for kld, locations, context in make_kld_tuples(context2info, cat_probes):
            num_locations_in_part1 = len(np.where(np.array(locations) < prep.midpoint)[0])
            num_locations_in_part2 = len(np.where(np.array(locations) > prep.midpoint)[0])
            y1 += [kld] * num_locations_in_part1
            y2 += [kld] * num_locations_in_part2
        y1 = np.array(y1)
        y2 = np.array(y2)

        # print examples
        print(cat)
        st = [(t[0], t[2]) for t in sorted(make_kld_tuples(context2info, cat_probes), key=lambda t: t[0])]
        print(st[:10])
        print(st[-10:])

        # fig
        if SHOW_HISTOGRAM:
            Y_MAX = 0.6
            NUM_BINS = 30
            X_RANGE = [0, 14]
            _, ax = plt.subplots(figsize=config.Fig.fig_size, dpi=config.Fig.dpi)
            ax.set_title('context-size={}'.format(context_size), fontsize=config.Fig.ax_fontsize)
            ax.set_ylabel('Probability', fontsize=config.Fig.ax_fontsize)
            ax.set_xlabel('KL Divergence', fontsize=config.Fig.ax_fontsize)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.tick_params(axis='both', which='both', top=False, right=False)
            ax.set_ylim([0, Y_MAX])
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

        # t test - operates on kl divergences, not coverage
        t, prob = ttest_ind(y1, y2, equal_var=True)
        cat2context_size2p[cat][context_size] = prob
        print('t={}'.format(t))
        print('p={:.6f}'.format(prob))
        print()

        # convert kld to coverage
        coverage1 = 1.0 / np.mean(y1)
        coverage2 = 1.0 / np.mean(y2)

        # populate tabular data
        for name, di in zip(headers, (cat, 1, context_size, coverage1, len(y1))):
            name2col[name].append(di)
        for name, di in zip(headers, (cat, 2, context_size, coverage2, len(y2))):
            name2col[name].append(di)

# data frame
df = pd.DataFrame(name2col)

# plot difference between partitions including all context-sizes
ax = pg.plot_paired(data=df, dv='coverage', within='partition', subject='Category', dpi=config.Fig.dpi)
ax.set_title(f'context-sizes={CONTEXT_SIZES}')
ax.set_ylabel('Coverage')
plt.show()

# aggregate over context_sizes and build easily readable table
dfs = []
for context_size in CONTEXT_SIZES:
    # filter by context size
    df_at_context_size = df[df['context-size'] == context_size]

    # quick comparison
    comparison = df_at_context_size.groupby(['Category', 'partition'])\
        .mean().reset_index().sort_values('coverage', ascending=False)
    print(comparison)
    print()

    # concatenate data from part 1 and 2 horizontally
    df1 = df_at_context_size.set_index('Category').groupby('Category')[['coverage', 'n']].first()
    df2 = df_at_context_size.set_index('Category').groupby('Category')[['coverage', 'n']].last()
    df1 = df1.rename(columns={'coverage': 'mean-coverage-1'})
    df2 = df2.rename(columns={'coverage': 'mean-coverage-2'})

    df_concat = pd.concat((df1, df2), axis=1)
    df_concat['p'] = [cat2context_size2p[cat][context_size] for cat in df_concat.index]
    dfs.append(df_concat)

    # pairwise t-test between means associated with each category - pairwise has more power in this case
    res = pg.pairwise_ttests(data=df_at_context_size, dv='coverage', within='partition', subject='Category')
    print(res)

    # plot difference between partitions
    ax = pg.plot_paired(data=df_at_context_size, dv='coverage', within='partition', subject='Category',
                        dpi=config.Fig.dpi)
    ax.set_title(f'context-size={context_size}')
    ax.set_ylabel('Coverage')
    plt.show()

df_master = pd.concat(dfs, axis=1)
df_master['overall-mean'] = df_master.filter(regex='mean*', axis=1).mean(axis=1)
df_master = df_master.sort_values('overall-mean', ascending=False)
df_master = df_master.drop('overall-mean', axis=1)

print(tabulate(df_master,
               tablefmt='latex',
               floatfmt=".3f"))
