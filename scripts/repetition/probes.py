"""
Research questions:
1. Are noun-contexts in partition 1 of AO-CHILDES repeated more often than expected due to chance?
Chance is simply controlling for frequency: The more frequent noun tokens are, the more likely repetition is to occur
"""

import numpy as np
import matplotlib.pyplot as plt
import attr
from tabulate import tabulate

from preppy.legacy import TrainPrep
from categoryeval.probestore import ProbeStore

from wordplay.svd import make_context_by_term_matrix
from wordplay.params import PrepParams
from wordplay.docs import load_docs

# /////////////////////////////////////////////////////////////////

CORPUS_NAME = 'childes-20180319'
PROBES_NAME = 'sem-4096'

SHUFFLE_DOCS = False

docs = load_docs(CORPUS_NAME,
                 num_test_take_from_mid=0,
                 num_test_take_random=0,
                 shuffle_docs=SHUFFLE_DOCS)

params = PrepParams()
prep = TrainPrep(docs, **attr.asdict(params))

probe_store = ProbeStore(CORPUS_NAME, PROBES_NAME, prep.store.w2id)

# ///////////////////////////////////////////////////////////////// TW-matrix

LOG_FREQUENCY = False
CONTEXT_SIZE = 6
COLORS = ['C0', 'C1']

MEASURE_NAME = 'UNNAMED'  # TODO

start1, end1 = 0, prep.midpoint
tw_mat1, xws1, yws1 = make_context_by_term_matrix(
    prep, start=start1, end=end1, context_size=CONTEXT_SIZE, log=LOG_FREQUENCY)
start2, end2 = prep.midpoint, prep.store.num_tokens
tw_mat2, xws2, yws2 = make_context_by_term_matrix(
    prep, start=start2, end=end2, context_size=CONTEXT_SIZE, log=LOG_FREQUENCY)

# ////////////////////////////////////////////////////////////


part_ids = range(2)
part_id2ys = {i: [] for i in part_ids}
table_data = []  # for table
for part_id, tw_mat, yws, xws in zip(part_ids,
                                     [tw_mat1, tw_mat2],
                                     [yws1, yws2],
                                     [xws1, xws2],
                                     ):

    # compute ratio for each category
    for cat in probe_store.cats:
        # calculate number of contexts
        col_ids = [n for n, xw in enumerate(xws) if xw in probe_store.cat2probes[cat]]
        cols = tw_mat.tocsc()[:, col_ids].toarray()
        context_distribution = np.sum(cols, axis=1, keepdims=False)
        num_context_types = np.count_nonzero(context_distribution)

        # compute measure
        y = None  # TODO number of types relative to chance
        part_id2ys[part_id].append(y)

        # collect
        table_data.append((part_id + 1, cat, num_context_types, y))
        print(f'{cat:<12} ratio={y:>6.1f} num contexts={num_context_types:>9,}')
    print('------------------------------------------------------')


# fig
_, ax = plt.subplots(figsize=(6, 6), dpi=None)
plt.title(f'{PROBES_NAME}\ncontext-size={CONTEXT_SIZE}', fontsize=12)
ax.set_ylabel(MEASURE_NAME)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='both', top=False, right=False)
ax.set_xticklabels(['partition 1', 'partition 2'])
# ax.set_ylim(0, 2.0)
# plot
y1 = part_id2ys[0]
y2 = part_id2ys[1]
ax.boxplot([y1, y2])
ax.axhline(y=np.mean(y1), label=f'part 1 mean={np.mean(y1):.4f} n={len(y1)}', color='blue')
ax.axhline(y=np.mean(y2), label=f'part 2 mean={np.mean(y2):.4f} n={len(y2)}', color='red')
plt.legend()
plt.tight_layout()
plt.show()


# print pretty table
print()
headers = ['Partition', "Part-of-Speech", "context types", MEASURE_NAME]
print(tabulate(table_data,
               headers=headers,
               tablefmt='fancy_grid'))

# latex
print(tabulate(table_data,
               headers=headers,
               tablefmt='latex',
               floatfmt=".4f"))
