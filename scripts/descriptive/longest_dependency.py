import attr
import numpy as np
from tabulate import tabulate

from preppy.legacy import TrainPrep
from preppy.legacy import make_windows_mat
from categoryeval.probestore import ProbeStore

from wordplay.word_sets import excluded
from wordplay.params import PrepParams
from wordplay.docs import load_docs
from wordplay.measures import calc_kl_divergence

# /////////////////////////////////////////////////////////////////

CORPUS_NAME = 'childes-20180319'
PROBES_NAME = 'syn-nva'

CONTEXT_SIZE = 24
NUM_PARTS = 1  # must be 1 to include all windows in windows matrix

docs = load_docs(CORPUS_NAME)

params = PrepParams(context_size=CONTEXT_SIZE, num_types=4096, num_parts=NUM_PARTS)  # TODO num types
prep = TrainPrep(docs, **attr.asdict(params))

probe_store = ProbeStore(CORPUS_NAME, PROBES_NAME, prep.store.w2id, excluded=excluded)

# /////////////////////////////////////////////////////////////////

windows_mat = make_windows_mat(prep.store.token_ids, prep.num_windows_in_part, prep.num_tokens_in_window)

all_outcomes, c = np.unique(windows_mat, return_counts=True)
p = c / np.sum(c)

for cat in probe_store.cats:

    print(cat)

    # condition on a subset of words
    cat_probes = probe_store.cat2probes[cat]
    cat_probe_ids = [prep.store.w2id[p] for p in cat_probes]
    row_ids = np.isin(windows_mat[:, -1], cat_probe_ids)

    for n, col in enumerate(windows_mat[row_ids].T):
        outcomes, c = np.unique(col, return_counts=True)
        o2p = dict(zip(outcomes, c / np.sum(c)))
        q = np.array([o2p.setdefault(o, 0) for o in all_outcomes])

        kl = calc_kl_divergence(p, q)
        print(f'distance={CONTEXT_SIZE - n:>2} kl={kl:.3f}')

    print()
