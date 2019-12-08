"""
Research questions:
1. Do contexts in partition 1 of AO-CHILDES contain category members more exclusively?
2. Are contexts more "pure" in the sense that they are ess contaminated by non-category words?

Ideally, the measure in question quantifies the probability of a context
re-occurring with a category member given it has occurred with a category member once before.

COMPUTE_MEASURE_ONCE_ON_WHOLE_CORPUS:
setting this to true is supposed to control for the unequal number of category members between partitions.
for example, nouns are more frequent in partition 1 of AO-CHILDES, but ideally,
prominence should work independently of frequency and if higher for nouns in partition 1 should
reflect some quality of the distributional properties of nouns independent of frequency.
if this is set to true, then each context is assigned a prominence value, based on the entire corpus,
rather than being computed on a single partition.
the difference in prominence between partitions then, stems entirely from the extent to which
high or low prominence contexts are distributed over partition 1 and 2

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
from wordplay.figs import make_histogram
from wordplay.memory import set_memory_limit

# /////////////////////////////////////////////////////////////////

CORPUS_NAME = 'childes-20180319'
PROBES_NAME = 'syn-4096'

docs = load_docs(CORPUS_NAME)

params = PrepParams()
prep = TrainPrep(docs, **attr.asdict(params))

probe_store = ProbeStore(CORPUS_NAME, PROBES_NAME, prep.store.w2id, excluded=excluded)

# /////////////////////////////////////////////////////////////////

COMPUTE_MEASURE_ONCE_ON_WHOLE_CORPUS = False
CONTEXT_SIZES = [1, 2, 3]
MEASURE_NAME = 'Prominence'
SHOW_HISTOGRAM = False

measure_name1 = MEASURE_NAME + '-p1'
measure_name2 = MEASURE_NAME + '-p2'


def make_context_ys(probes: Set[str],
                    tokens: List[str],
                    size: int,
                    start: int,
                    end: int,
                    ) -> Dict[Tuple[str], Dict[str, Any]]:
    """
    compute proportion of times a category context actually co-occurs with cateogy member
    """
    info = {'in-category': [],
            'locations': [],
            }
    context2info = {}
    pbar = pyprind.ProgBar(len(tokens))
    for loc, token in enumerate(tokens[:-size]):
        context = tuple(tokens[loc + dist] for dist in range(-size, 0) if dist != 0)
        if token in probes:
            context2info.setdefault(context, deepcopy(info))['in-category'].append(1)
            context2info.setdefault(context, deepcopy(info))['locations'].append(loc)
        else:
            context2info.setdefault(context, deepcopy(info))['in-category'].append(0)
            context2info.setdefault(context, deepcopy(info))['locations'].append(loc)
        pbar.update()

    # return a measure for each time a context occurs between start and end
    for context in context2info.keys():
        # a context must occur at least once with the category
        # note: this returns a lot of contexts, because lots of generic noun contexts co-occur with semantic probes
        in_category_list = context2info[context]['in-category']
        num_in_category = np.sum(in_category_list)
        if num_in_category == 0:
            continue

        y = num_in_category / len(in_category_list)

        locations_array = np.array(context2info[context]['locations'])
        num_locations_in_partition = np.sum(np.logical_and(start < locations_array, locations_array < end)).item()
        for _ in range(num_locations_in_partition):  # get the same prominence value each time the context occurs

            yield y


set_memory_limit(prop=0.90)

headers = ['category', 'partition', 'context-size', MEASURE_NAME, 'n']
name2col = {name: [] for name in headers}
cat2context_size2p = {cat: {} for cat in probe_store.cats}
for cat in probe_store.cats:
    cat_probes = probe_store.cat2probes[cat]

    for context_size in CONTEXT_SIZES:
        print(cat)
        print(f'context-size={context_size}')

        if COMPUTE_MEASURE_ONCE_ON_WHOLE_CORPUS:
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
            context_ys1 = np.array(list(make_context_ys(cat_probes, tokens1, context_size,
                                                        start=start1, end=end1))).astype(np.float32)
            context_ys2 = np.array(list(make_context_ys(cat_probes, tokens2, context_size,
                                                        start=start2, end=end2))).astype(np.float32)
        except MemoryError:
            print('WANING: Memory Error. Skipping computation')
            continue

        # fig
        if SHOW_HISTOGRAM:
            title = f'context-size={context_size}'
            x_label = MEASURE_NAME
            make_histogram(context_ys1, context_ys2, x_label, y_max=None, num_bins=None, x_range=None)
            plt.show()

        # mann-whitney test
        try:
            res = pg.mwu(context_ys1, context_ys2, tail='two-sided')
        except MemoryError:
            print('Skipping Mann-Whitney test due to memory error')
            prob = np.nan
        else:
            print()
            print(res)
            prob = res['p-val']

        # take mean over prominence values computed on contexts
        # to compute overall prominence for a category
        yi1 = np.mean(context_ys1)
        yi2 = np.mean(context_ys2)

        # console
        print(f'n1={len(context_ys1):,}')
        print(f'n2={len(context_ys2):,}')
        print(f'{MEASURE_NAME}={yi1 :.6f}')
        print(f'{MEASURE_NAME}={yi2 :.6f}')
        print()

        # populate tabular data
        for name, di in zip(headers, (cat, 1, context_size, yi1, len(context_ys1))):
            name2col[name].append(di)
        for name, di in zip(headers, (cat, 2, context_size, yi2, len(context_ys2))):
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
