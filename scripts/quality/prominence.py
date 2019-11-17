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
PROBES_NAME = 'syn-4096'
NUM_TYPES = None

docs = load_docs(CORPUS_NAME,
                 shuffle_sentences=False)

params = PrepParams(num_types=NUM_TYPES)
prep = TrainPrep(docs, **attr.asdict(params))

probe_store = ProbeStore(CORPUS_NAME, PROBES_NAME, prep.store.w2id, excluded=excluded)

# /////////////////////////////////////////////////////////////////

CONTEXT_SIZES = [4]
MEASURE_NAME = 'Prominence'

MIN_CO_OCCURRENCE_FREQ = 1  # the higher the less power


def make_count_df(probes: Set[str],
                  tokens: List[str],
                  distance: int,
                  ) -> pd.DataFrame:
    """

    """
    print('Collecting information about probe context locations...')

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

    res = pd.DataFrame(data={'in-category': col})

    return res


set_memory_limit(prop=0.9)


tokens1 = prep.store.tokens[:prep.midpoint // 1]  # TODO // 4 remove this in final analysis
tokens2 = prep.store.tokens[-prep.midpoint // 1:]

name2col = {'Category': [], 'partition': [], MEASURE_NAME: []}
for cat in probe_store.cats:
    cat_probes = probe_store.cat2probes[cat]

    # the probability that a context occurs with a category member
    num_cat_tokens1 = sum([1 for w in tokens1 if w in cat_probes])
    num_cat_tokens2 = sum([1 for w in tokens2 if w in cat_probes])

    print(cat)
    print(num_cat_tokens1)
    print(num_cat_tokens2)

    for context_size in CONTEXT_SIZES:

        # compute measure for contexts associated with a single category
        try:
            df1 = make_count_df(cat_probes, tokens1, context_size)
        except MemoryError:
            raise SystemExit('Reached memory limit')

        try:
            df2 = make_count_df(cat_probes, tokens2, context_size)
        except MemoryError:
            raise SystemExit('Reached memory limit')

        df1['partition'] = 1
        df2['partition'] = 2
        df_master = pd.concat((df1, df2))

        print()
        print(f'n1={len(df1):,}')
        print(f'n2={len(df2):,}')

        # chi-square
        expected, observed, stats = pg.chi2_independence(df_master, x='partition', y='in-category')
        print('expected:')
        print(expected)
        print('observed"')
        print(observed)
        print(stats)

        col1 = df_master[df_master['partition'] == 1]['in-category']
        col2 = df_master[df_master['partition'] == 2]['in-category']
        yi1 = col1.sum() / len(col1)
        yi2 = col2.sum() / len(col2)
        print(f'proportion={yi1 :.6f}')
        print(f'proportion={yi2 :.6f}')
        print()

        name2col[MEASURE_NAME].append(yi1)
        name2col[MEASURE_NAME].append(yi2)
        name2col['Category'].append(cat)
        name2col['Category'].append(cat)
        name2col['partition'].append(1)
        name2col['partition'].append(2)


df = pd.DataFrame(data=name2col)
ax = pg.plot_paired(data=df, dv=MEASURE_NAME, within='partition', subject='Category', dpi=config.Fig.dpi)
ax.set_title(f'context-sizes={CONTEXT_SIZES}')
ax.set_ylabel(MEASURE_NAME)
plt.show()

res = pg.pairwise_ttests(data=df, dv=MEASURE_NAME, within='partition', subject='Category')
print(res)
