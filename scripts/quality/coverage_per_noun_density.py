"""
Research questions:
1. Do contexts in partition 1 of AO-CHILDES cover category members more uniformly?
2. Are same-category members more substitutable in contexts that occur in partition 1?

This measure was previously called within-category substitutability.


This is an attempt at computing coverage un-confounded by changing token frequency of nouns between partitions.
To do so, coverage is computed for chunks of partitions with equal noun density
"""

import matplotlib.pyplot as plt
import attr
import numpy as np
import pyprind
from typing import Dict, Any, Set, List, Tuple
from tabulate import tabulate
import pandas as pd
import pingouin as pg
from copy import deepcopy

from preppy.legacy import TrainPrep
from categoryeval.probestore import ProbeStore

from wordplay import config
from wordplay.word_sets import excluded
from wordplay.params import PrepParams
from wordplay.docs import load_docs
from wordplay.measures import calc_kl_divergence
from wordplay.memory import set_memory_limit
from wordplay.figs import make_histogram

# /////////////////////////////////////////////////////////////////

CORPUS_NAME = 'childes-20180319'
PROBES_NAME = 'sem-all'

docs = load_docs(CORPUS_NAME)

params = PrepParams()
prep = TrainPrep(docs, **attr.asdict(params))

probe_store = ProbeStore(CORPUS_NAME, PROBES_NAME, prep.store.w2id, excluded=excluded)

# /////////////////////////////////////////////////////////////////

COMPUTE_KLDS_ONCE_ON_WHOLE_CORPUS = False  # otherwise compute klds separately on each partition
CONTEXT_SIZES = [1, 2, 3]
MEASURE_NAME = 'Coverage'
SHOW_HISTOGRAM = False

measure_name1 = MEASURE_NAME + '-p1'
measure_name2 = MEASURE_NAME + '-p2'


def make_klds(probes: Set[str],
              tokens: List[str],
              size: int,
              start: int,
              end: int,
              ) -> Dict[Tuple[str], Dict[str, Any]]:
    """
    compute klds for context types on entire corpus if tokens corresponds to entire corpus,
    or compute klds for context types on a partition, if tokens corresponds to a partition
    """
    info = {'freq_by_probe': {probe: 0.0 for probe in probes},
            'total_freq': 0,
            'locations': [],
            }
    context2info = {}
    pbar = pyprind.ProgBar(len(tokens))
    for loc, token in enumerate(tokens[:-size]):
        context = tuple(tokens[loc + dist] for dist in range(-size, 0) if dist != 0)
        if token in probes:
            context2info.setdefault(context, deepcopy(info))['freq_by_probe'][token] += 1
            context2info.setdefault(context, deepcopy(info))['total_freq'] += 1
            context2info.setdefault(context, deepcopy(info))['locations'].append(loc)
        pbar.update()

    # klds
    e = np.array([1 / len(probes) for _ in probes])
    for context in context2info.keys():
        o = np.array(
            [context2info[context]['freq_by_probe'][p] / context2info[context]['total_freq'] for p in probes])
        kl = calc_kl_divergence(e, o)  # asymmetric: expected probabilities, observed probabilities

        locations_array = np.array(context2info[context]['locations'])
        num_locations_in_partition = np.sum(np.logical_and(start < locations_array, locations_array < end)).item()
        for _ in range(num_locations_in_partition):  # get the same kld value each time the context occurs

            yield kl


set_memory_limit(prop=0.9)

headers = ['category', 'partition', 'context-size', MEASURE_NAME, 'n']
name2col = {name: [] for name in headers}
cat2context_size2p = {cat: {} for cat in probe_store.cats}
for cat in probe_store.cats:
    cat_probes = probe_store.cat2probes[cat]

    for context_size in CONTEXT_SIZES:
        print(cat)
        print(f'context-size={context_size}')

        if COMPUTE_KLDS_ONCE_ON_WHOLE_CORPUS:
            tokens1 = prep.store.tokens
            tokens2 = prep.store.tokens
            start1, end1 = 0, prep.midpoint
            start2, end2 = prep.midpoint, prep.store.num_tokens
        else:
            tokens1 = prep.store.tokens[:prep.midpoint]
            tokens2 = prep.store.tokens[-prep.midpoint:]
            start1, end1 = 0, prep.num_tokens_in_part
            start2, end2 = 0, prep.num_tokens_in_part

        # compute measure for contexts associated with a single category in a single partition
        try:
            klds1 = np.array(list(make_klds(cat_probes, tokens1, context_size,
                                            start=start1, end=end1)))
        except MemoryError:
            raise SystemExit('Reached memory limit')

        try:
            klds2 = np.array(list(make_klds(cat_probes, tokens2, context_size,
                                            start=start2, end=end2)))
        except MemoryError:
            raise SystemExit('Reached memory limit')

        # fig
        if SHOW_HISTOGRAM:
            title = f'context-size={context_size}'
            x_label = 'KL Divergence'
            make_histogram(klds1, klds2, x_label, y_max=0.6, num_bins=30, x_range=[0, 14])
            plt.show()

        # klds are not normally distributed - need non-parametric test
        # statistical test - operates on kl divergences, not coverage
        res = pg.mwu(klds1, klds2, tail='two-sided')
        print()
        print(res)
        prob = res['p-val']

        # convert klds to coverage
        yi1 = 1.0 / np.mean(klds1)
        yi2 = 1.0 / np.mean(klds2)

        # console
        print(f'n1={len(klds1):,}')
        print(f'n2={len(klds2):,}')
        print(f'{MEASURE_NAME}={yi1 :.6f}')
        print(f'{MEASURE_NAME}={yi2 :.6f}')
        print()

        # populate tabular data
        for name, di in zip(headers, (cat, 1, context_size, yi1, len(klds1))):
            name2col[name].append(di)
        for name, di in zip(headers, (cat, 2, context_size, yi2, len(klds2))):
            name2col[name].append(di)

        cat2context_size2p[cat][context_size] = prob

# data frame for statistics
df_stats = pd.DataFrame(name2col)

# plot difference between partitions including all context-sizes
ax = pg.plot_paired(data=df_stats, dv=MEASURE_NAME, within='partition', subject='category', dpi=config.Fig.dpi)
ax.set_title(f'context-sizes={CONTEXT_SIZES}')
ax.set_ylabel(MEASURE_NAME)
plt.show()

# convert df to human readable format
dfs = []
for context_size in CONTEXT_SIZES:
    # filter by context size
    df_at_context_size = df_stats[df_stats['context-size'] == context_size]

    # quick comparison
    comparison = df_at_context_size.groupby(['category', 'partition']) \
        .mean().reset_index().sort_values(MEASURE_NAME, ascending=False)
    print(comparison)
    print()

    # concatenate data from part 1 and 2 horizontally
    df1 = df_at_context_size.set_index('category').groupby('category')[[MEASURE_NAME, 'n']].first()
    df2 = df_at_context_size.set_index('category').groupby('category')[[MEASURE_NAME, 'n']].last()
    df1 = df1.rename(columns={MEASURE_NAME: 'mean' + measure_name1})
    df2 = df2.rename(columns={MEASURE_NAME: 'mean' + measure_name2})

    df_concat = pd.concat((df1, df2), axis=1)
    df_concat['p'] = [cat2context_size2p[cat][context_size] for cat in df_concat.index]
    dfs.append(df_concat)

    # pairwise t-test between means associated with each category - pairwise has more power in this case
    res = pg.pairwise_ttests(data=df_at_context_size, dv=MEASURE_NAME, within='partition', subject='category')
    print(res)

    # plot difference between partitions
    ax = pg.plot_paired(data=df_at_context_size, dv=MEASURE_NAME, within='partition', subject='category',
                        dpi=config.Fig.dpi)
    ax.set_title(f'context-size={context_size}')
    ax.set_ylabel(MEASURE_NAME)
    plt.show()

# convert to human readable format
df_human = pd.concat(dfs, axis=1)
df_human['overall-mean'] = df_human.filter(regex='mean*', axis=1).mean(axis=1)
df_human = df_human.sort_values('overall-mean', ascending=False)
df_human = df_human.drop('overall-mean', axis=1)

print(tabulate(df_human,
               tablefmt='latex',
               floatfmt=".3f"))
