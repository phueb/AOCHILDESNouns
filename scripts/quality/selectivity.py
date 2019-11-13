"""
Research questions:
1. Are noun-context types in partition 1 of AO-CHILDES more selective? Are there fewer types than due to chance?

We are interested in how consistently the same multi-word context co-occurs with nouns, and not frequency.
If we were to simply compare the type token ratio of noun contexts in part 1 vs 2 of AO-CHILDES,
the ratio for part 1 would be inflated simply due to greater noun token frequency.
As a remedy, the same type token ratio is computed, but for tokens shuffled within a partition,
and this ratio is divided by the ratio computed for the intact (not shuffled) tokens
"""

import numpy as np
import matplotlib.pyplot as plt
import attr
from tabulate import tabulate

from preppy.legacy import TrainPrep
from categoryeval.probestore import ProbeStore

from wordplay.representation import make_context_by_term_matrix
from wordplay.params import PrepParams
from wordplay.docs import load_docs
from wordplay.measures import calc_selectivity

# /////////////////////////////////////////////////////////////////

CORPUS_NAME = 'childes-20180319'
PROBES_NAME = 'syn-4096'

SHUFFLE_DOCS = False

docs = load_docs(CORPUS_NAME,
                 num_test_take_from_mid=0,
                 num_test_take_random=0,
                 shuffle_docs=SHUFFLE_DOCS)

params = PrepParams()
prep = TrainPrep(docs, **attr.asdict(params))

probe_store = ProbeStore(CORPUS_NAME, PROBES_NAME, prep.store.w2id)

# ///////////////////////////////////////////////////////////////// parameters

CONTEXT_SIZE = 2
MEASURE_NAME = 'Context-Selectivity'  # the ratio of two context type-token ratios


# ////////////////////////////////////////////////////////////////// co-occurrence matrix

# WARNING: prep.store.tokens are shuffled.
# Do NOT shuffle BEFORE calculating matrix for un-shuffled tokens

tw_mat1_observed, xws1_observed, _ = make_context_by_term_matrix(prep.store.tokens,
                                                                 start=0,
                                                                 end=prep.midpoint,
                                                                 context_size=CONTEXT_SIZE)
tw_mat2_observed, xws2_observed, _ = make_context_by_term_matrix(prep.store.tokens,
                                                                 start=prep.midpoint,
                                                                 end=prep.store.num_tokens,
                                                                 context_size=CONTEXT_SIZE)

# now it is safe to shuffle tokens
tw_mat1_chance, xws1_chance, _ = make_context_by_term_matrix(prep.store.tokens,
                                                             start=0,
                                                             end=prep.midpoint,
                                                             context_size=CONTEXT_SIZE,
                                                             shuffle_tokens=True)
tw_mat2_chance, xws2_chance, _ = make_context_by_term_matrix(prep.store.tokens,
                                                             start=prep.midpoint,
                                                             end=prep.store.num_tokens,
                                                             context_size=CONTEXT_SIZE,
                                                             shuffle_tokens=True)

# //////////////////////////////////////////////////////////// compute measure


part_ids = range(2)
part_id2ys = {i: [] for i in part_ids}
part_id2cat2y = {i: {cat: None for cat in probe_store.cats} for i in part_ids}
rows = []  # for table
for part_id, tw_mat_observed, tw_mat_chance, xws_observed, xws_chance in zip(part_ids,
                                                                             [tw_mat1_observed, tw_mat2_observed],
                                                                             [tw_mat1_chance, tw_mat2_chance],
                                                                             [xws1_observed, xws2_observed],
                                                                             [xws1_chance, xws2_chance],
                                                                             ):

    # compute selectivity for each category
    for cat in probe_store.cats:
        y = calc_selectivity(tw_mat_chance,
                             tw_mat_observed,
                             xws_chance,
                             xws_observed,
                             probe_store.cat2probes[cat])
        # collect
        part_id2ys[part_id].append(y)
        part_id2cat2y[part_id][cat] = y


# fig
_, ax = plt.subplots(figsize=(6, 6), dpi=None)
plt.title(f'{PROBES_NAME}\ncontext-size={CONTEXT_SIZE}', fontsize=12)
ax.set_ylabel(MEASURE_NAME)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='both', top=False, right=False)
ax.set_xticklabels(['partition 1', 'partition 2'])
ax.set_ylim(0, 5.0)
# plot
y1 = part_id2ys[0]
y2 = part_id2ys[1]
ax.boxplot([y1, y2])
ax.axhline(y=np.mean(y1), label=f'part 1 mean={np.mean(y1):.4f} n={len(y1)}', color='C0')
ax.axhline(y=np.mean(y2), label=f'part 2 mean={np.mean(y2):.4f} n={len(y2)}', color='C1')
plt.legend()
plt.tight_layout()
plt.show()


# latex table - aggregated
headers = ['Partition', 'CTTR-observed', 'CTTR-chance', MEASURE_NAME]
rows = [
    (1, None, None, np.mean(y1)),
    (2, None, None, np.mean(y2)),
]
print(tabulate(rows,
               headers=headers,
               tablefmt='latex',
               floatfmt=".4f"))

# latex table - by category
headers = ['Category', 'partition 1', 'partition 2']
rows = [(cat, part_id2cat2y[0][cat], part_id2cat2y[1][cat]) for cat in probe_store.cats]
print()
print(tabulate(rows,
               headers=headers,
               tablefmt='latex',
               floatfmt=".2f"))