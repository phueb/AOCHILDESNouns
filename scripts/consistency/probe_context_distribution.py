"""
Research questions:
1. Do probes in partition 1 have fewer context types, but more context tokens?

Note:
Contexts are always words to the left of probe.
Looking at only contexts may not tell the full story.
Instead looking right-contexts (called targets) may be useful
"""

import numpy as np
import matplotlib.pyplot as plt
import attr

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
CONTEXT_SIZE = 3
COLORS = ['C0', 'C1']

PLOT_MAX_NUM_CONTEXTS = 100

start1, end1 = 0, prep.midpoint
tw_mat1, xws1, yws1 = make_context_by_term_matrix(
    prep, start=start1, end=end1, context_size=CONTEXT_SIZE, log=LOG_FREQUENCY)
start2, end2 = prep.midpoint, prep.store.num_tokens
tw_mat2, xws2, yws2 = make_context_by_term_matrix(
    prep, start=start2, end=end2, context_size=CONTEXT_SIZE, log=LOG_FREQUENCY)

# ////////////////////////////////////////////////////////////


part_ids = range(2)
cat2part_id2counts = {cat: {part_id: None for part_id in part_ids} for cat in probe_store.cats}
for part_id, tw_mat, yws, xws in zip(part_ids,
                                     [tw_mat1, tw_mat2],
                                     [yws1, yws2],
                                     [xws1, xws2],
                                     ):

    # calculate number of contexts and their frequency for each category
    for cat in probe_store.cats:
        col_ids = [n for n, xw in enumerate(xws) if xw in probe_store.cat2probes[cat]]
        cols = tw_mat[:, col_ids].toarray()
        context_distribution = np.sum(cols, axis=1, keepdims=False)
        counts = np.sort(context_distribution)[::-1]
        cat2part_id2counts[cat][part_id] = counts

        print(cat)
        print(counts[:PLOT_MAX_NUM_CONTEXTS])
    print('------------------------------------------------------')


# fig2
for cat, part_id2ys in cat2part_id2counts.items():
    fig2, ax2 = plt.subplots(figsize=(6, 6), dpi=None)
    plt.title(f'Distribution of left-contexts for {cat.upper()}\ncontext-size={CONTEXT_SIZE}',
              fontsize=12)
    ax2.set_xlabel('Context Types')
    ax2.set_ylabel('Frequency')
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.tick_params(axis='both', which='both', top=False, right=False)
    #
    for part_id, y in part_id2ys.items():
        label = 'partition={}'.format(part_id + 1)
        ax2.plot(y[:PLOT_MAX_NUM_CONTEXTS],
                 label=label + f' (# contexts={np.count_nonzero(y):>6,})',
                 color=COLORS[part_id])
    ax2.legend(frameon=False, loc='upper right')
    plt.tight_layout()
    plt.show()