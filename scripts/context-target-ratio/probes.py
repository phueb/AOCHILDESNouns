"""
Research questions:
1. Do probes in partition 1 of AO-CHILDES have a higher context-target-ratio?

Note:
Contexts are always words to the left of probe.
Looking at only contexts may not tell the full story.
Instead looking also at right-contexts (called targets) may be useful.
it might be best to look at both together:
For example, what is the average number of contexts per target across all members of some category?
The hypothesis being tested is:
The more context types and hte fewer target types (or the higher their ratio) should be be better for category learning.
Therefore, the ratio should be higher in partition 1 of AO-CHILDES.

a reminder:
a window consists of a sequence of context-words and a target.
of interest are contexts which contain a probe in the last position.
[.., context3, context2, context1, probe] [target]
"""

import numpy as np
import matplotlib.pyplot as plt
import attr

from preppy.legacy import TrainPrep
from categoryeval.probestore import ProbeStore

from wordplay.svd import make_term_by_context_co_occurrence_mat
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
CONTEXT_SIZE = 4
COLORS = ['C0', 'C1']

start1, end1 = 0, prep.midpoint
tw_mat1, xws1, yws1 = make_term_by_context_co_occurrence_mat(
    prep, start=start1, end=end1, context_size=CONTEXT_SIZE, log=LOG_FREQUENCY)
start2, end2 = prep.midpoint, prep.store.num_tokens
tw_mat2, xws2, yws2 = make_term_by_context_co_occurrence_mat(
    prep, start=start2, end=end2, context_size=CONTEXT_SIZE, log=LOG_FREQUENCY)

# ////////////////////////////////////////////////////////////


part_ids = range(2)
part_id2ys = {i: [] for i in part_ids}
for part_id, tw_mat, yws, xws in zip(part_ids,
                                     [tw_mat1, tw_mat2],
                                     [yws1, yws2],
                                     [xws1, xws2],
                                     ):

    # compute ratio for each category
    for cat in probe_store.cats:
        # calculate number of contexts
        col_ids = [n for n, xw in enumerate(xws) if xw in probe_store.cat2probes[cat]]
        cols = tw_mat[:, col_ids].toarray()
        context_distribution = np.sum(cols, axis=1, keepdims=False)
        context_types = np.unique(context_distribution)
        num_context_types = len(context_types)

        # calculate number of targets
        row_ids = [n for n, yw in enumerate(yws) if yw[-1] in probe_store.cat2probes[cat]]
        rows = tw_mat[row_ids, :].toarray()
        target_distribution = np.sum(rows, axis=0, keepdims=False)
        target_types = np.unique(target_distribution)
        num_target_types = len(target_types)

        # compute ratio
        ratio = num_context_types / num_target_types
        part_id2ys[part_id].append(ratio)

        print(f'{cat:<12} ratio={ratio:.3f} num contexts={num_context_types:>6,} num targets={num_target_types:>6,}')
    print('------------------------------------------------------')


# fig
_, ax = plt.subplots(figsize=(6, 6), dpi=None)
plt.title(f'context-size={CONTEXT_SIZE}', fontsize=12)
ax.set_ylabel('Context-Target Ratio')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='both', top=False, right=False)
ax.set_xticklabels(['partition 1', 'partition 2'])
ax.set_ylim(0, 2.0)
# plot
y1 = part_id2ys[0]
y2 = part_id2ys[1]
ax.boxplot([y1, y2])
ax.axhline(y=np.mean(y1), label=f'part 1 mean={np.mean(y1):.4f} n={len(y1)}', color='blue')
ax.axhline(y=np.mean(y2), label=f'part 2 mean={np.mean(y2):.4f} n={len(y2)}', color='red')
plt.legend()
plt.tight_layout()
plt.show()