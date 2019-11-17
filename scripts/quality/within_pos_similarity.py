"""
Research questions:
1. Are contexts for a given POS class more similar in part 1 of AO-CHILDES?
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import linalg as slinalg
from sklearn.metrics.pairwise import cosine_similarity
import attr

from categoryeval.probestore import ProbeStore
from preppy.legacy import TrainPrep

from wordplay import config
from wordplay.word_sets import excluded
from wordplay.params import PrepParams
from wordplay.docs import load_docs
from wordplay.representation import make_context_by_term_matrix


# /////////////////////////////////////////////////////////////////

CORPUS_NAME = 'childes-20180319'
PROBES_NAME = 'syn-4096'

REVERSE = False
NUM_PARTS = 2
docs = load_docs(CORPUS_NAME)

params = PrepParams(num_parts=NUM_PARTS, reverse=REVERSE)
prep = TrainPrep(docs, **attr.asdict(params))

probe_store = ProbeStore(CORPUS_NAME, PROBES_NAME, prep.store.w2id, excluded=excluded)

# ///////////////////////////////////////////////////////////////// parameters

CONTEXT_SIZE = 3
NUM_DIMS_LIST = [2, 4, 8, 16, 32, 64, 128]
POS = 'NOUN'

START1, END1 = 0, prep.midpoint // 1
START2, END2 = prep.store.num_tokens - END1, prep.store.num_tokens

# ////////////////////////////////////////////////////////////////

pos_words = probe_store.cat2probes[POS]

# make co-occurrence matrix
tw_mat1, xws1, yws1 = make_context_by_term_matrix(prep.store.tokens, start=START1, end=END1, context_size=CONTEXT_SIZE)
tw_mat2, xws2, yws2 = make_context_by_term_matrix(prep.store.tokens, start=START2, end=END2, context_size=CONTEXT_SIZE)

# compute similarities
num_dims2sims = {num_dims: [] for num_dims in NUM_DIMS_LIST}
for mat, xws in [(tw_mat1.T.asfptype(), xws1),  # after transposition, x-words index rows
                 (tw_mat2.T.asfptype(), xws2)]:

    u, s, v = slinalg.svds(mat, k=max(NUM_DIMS_LIST), return_singular_vectors='u')

    # compute  representations for each num_dim
    for num_dims in NUM_DIMS_LIST:
        bool_ids = [True if w in pos_words else False for w in xws]
        pos_reps = u[bool_ids, :num_dims]

        # cosine similarity
        sim = cosine_similarity(pos_reps).mean()
        num_dims2sims[num_dims].append(sim)

        print(f'Similarity={sim:.4f}')

# fig
fig, ax = plt.subplots(dpi=None, figsize=(5, 5))
plt.title(f'Distributional similarity within {POS}\n'
          f'context-size={CONTEXT_SIZE}\n'
          f'shuffled documents={SHUFFLE_DOCS}')
ax.set_ylabel('Context Similarity')
ax.set_xlabel('Partition')
ax.set_xticks([0, 1])
ax.set_ylim([0, 1])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='both', top=False, right=False)
plt.grid(True, which='both', axis='y', alpha=0.2)
# plot
for num_dims, sims in num_dims2sims.items():
    ax.plot(sims, label=f'num sing. dims={num_dims}')
plt.legend(loc='upper right', frameon=False)
fig.tight_layout()
plt.show()