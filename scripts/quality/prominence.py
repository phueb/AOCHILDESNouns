"""
Research questions:
1. Do contexts in partition 1 of AO-CHILDES contain category members more exclusively?
2. Are contexts more "pure" in the sense that they are ess contaminated by non-category words?

Ideally, the measure in question quantifies the probability of a context
re-occurring with a category member given it has occurred with a category member once before
"""

import pandas as pd
import attr
import matplotlib.pyplot as plt
import pyprind
import pingouin as pg
from typing import Set, List
from tabulate import tabulate

from preppy.legacy import TrainPrep
from categoryeval.probestore import ProbeStore

from wordplay import config
from wordplay.word_sets import excluded
from wordplay.params import PrepParams
from wordplay.docs import load_docs
from wordplay.memory import set_memory_limit

# /////////////////////////////////////////////////////////////////

CORPUS_NAME = 'childes-20180319'
PROBES_NAME = 'syn-4096'

docs = load_docs(CORPUS_NAME)

params = PrepParams()
prep = TrainPrep(docs, **attr.asdict(params))

probe_store = ProbeStore(CORPUS_NAME, PROBES_NAME, prep.store.w2id, excluded=excluded)
t
# /////////////////////////////////////////////////////////////////

CONTEXT_SIZES = [2, 3]
MEASURE_NAME = 'Prominence'
MIN_CO_OCCURRENCE_FREQ = 1  # the higher the less power

measure_name1 = MEASURE_NAME + '-p1'
measure_name2 = MEASURE_NAME + '-p2'


def make_count_df(probes: Set[str],
                  tokens: List[str],
                  distance: int,
                  ) -> pd.DataFrame:
    """
    for each context that co-occurs with a category,
    keep track of all the times it otherwise co-occurs with a category member or not
    """
    print('Collecting co-occurrence information...')

    # collect all contexts, keeping track of whether it is in-category
    pbar = pyprind.ProgBar(len(tokens))
    context2in_category_list = {}
    for loc, token in enumerate(tokens[:-distance]):
        context = tuple(tokens[loc + d] for d in range(-distance, 0) if d != 0)
        if token in probes:
            context2in_category_list.setdefault(context, []).append(1)
        else:
            context2in_category_list.setdefault(context, []).append(0)
        pbar.update()

    # only return information about contexts that occur at least once with category
    # note: this returns a lot of contexts, because lots of generic noun contexts co-occur with semantic probes
    col = []
    for context, in_category_list in context2in_category_list.items():
        if sum(in_category_list) <= MIN_CO_OCCURRENCE_FREQ:
            continue
        for in_category in in_category_list:
            col.append(in_category)
    count_df = pd.DataFrame(data={'in-category': col})
    return count_df


set_memory_limit(prop=0.9)


tokens1 = prep.store.tokens[:prep.midpoint // 1]
tokens2 = prep.store.tokens[-prep.midpoint // 1:]

headers = ['category', 'partition', 'context-size', MEASURE_NAME, 'n']
name2col = {name: [] for name in headers}
cat2context_size2p = {cat: {} for cat in probe_store.cats}
for cat in probe_store.cats:
    cat_probes = probe_store.cat2probes[cat]

    for context_size in CONTEXT_SIZES:

        # the probability that a context occurs with a category member
        num_cat_tokens1 = sum([1 for w in tokens1 if w in cat_probes])
        num_cat_tokens2 = sum([1 for w in tokens2 if w in cat_probes])

        print(cat)
        print(num_cat_tokens1)
        print(num_cat_tokens2)

        # compute measure for contexts associated with a single category in a single partition
        try:
            df1 = make_count_df(cat_probes, tokens1, context_size)
        except MemoryError:
            raise SystemExit('Reached memory limit')

        try:
            df2 = make_count_df(cat_probes, tokens2, context_size)
        except MemoryError:
            raise SystemExit('Reached memory limit')

        # chi-square
        df1['partition'] = 1
        df2['partition'] = 2
        df_at_cat = pd.concat((df1, df2))
        expected, observed, stats = pg.chi2_independence(df_at_cat, x='partition', y='in-category')
        prob = stats['p'][0]
        print('expected:')
        print(expected)
        print('observed"')
        print(observed)
        print(stats)

        col1 = df_at_cat[df_at_cat['partition'] == 1]['in-category']
        col2 = df_at_cat[df_at_cat['partition'] == 2]['in-category']
        yi1 = col1.sum() / len(col1)
        yi2 = col2.sum() / len(col2)

        # console
        print(f'n1={len(df1):,}')
        print(f'n2={len(df2):,}')
        print(f'proportion={yi1 :.6f}')
        print(f'proportion={yi2 :.6f}')
        print()

        # populate tabular data
        for name, di in zip(headers, (cat, 1, context_size, yi1, len(df1))):
            name2col[name].append(di)
        for name, di in zip(headers, (cat, 2, context_size, yi2, len(df2))):
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
    comparison = df_at_context_size.groupby(['category', 'partition'])\
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
