import attr
import numpy as np

from preppy import PartitionedPrep as TrainPrep
from preppy.utils import make_windows_mat
from categoryeval.probestore import ProbeStore

from wordplay.io import load_docs
from wordplay.measures import calc_kl_divergence

# /////////////////////////////////////////////////////////////////

CORPUS_NAME = 'childes-20191112_terms'
PROBES_NAME = 'syn-nva'

CONTEXT_SIZE = 12
NUM_PARTS = 1  # must be 1 to include all windows in windows matrix
SHUFFLE_SENTENCES = True # if not True, document-level dependencies result in very long distance dependencies

docs = load_docs(CORPUS_NAME, shuffle_sentences=SHUFFLE_SENTENCES)

params = PrepParams(context_size=CONTEXT_SIZE, num_types=None, num_parts=NUM_PARTS)
prep = TrainPrep(docs, **attr.asdict(params))

probe_store = ProbeStore(CORPUS_NAME, PROBES_NAME, prep.store.w2id)

# /////////////////////////////////////////////////////////////////

windows_mat = make_windows_mat(prep.store.token_ids, prep.num_windows_in_part, prep.num_tokens_in_window)

# un-conditional probability of words in the whole text
token_id_types, token_id_counts = np.unique(windows_mat, return_counts=True)
p = token_id_counts / np.sum(token_id_counts)


for cat in probe_store.cats:

    print(cat)

    # condition on a subset of words
    cat_probes = probe_store.cat2probes[cat]
    cat_probe_ids = [prep.store.w2id[p] for p in cat_probes]
    bool_ids = np.isin(windows_mat[:, -1], cat_probe_ids)

    # kl divergence due to chance
    col_chance = np.random.choice(prep.store.token_ids, size=np.sum(bool_ids), replace=True)
    outcomes, c = np.unique(col_chance, return_counts=True)
    o2p = dict(zip(outcomes, c / np.sum(c)))
    q_chance = np.array([o2p.setdefault(o, 0) for o in token_id_types])
    kl_chance = calc_kl_divergence(p, q_chance)

    for n, col in enumerate(windows_mat[bool_ids].T):

        # actual kl divergence
        outcomes, c = np.unique(col, return_counts=True)
        o2p = dict(zip(outcomes, c / np.sum(c)))
        q = np.array([o2p.setdefault(o, 0) for o in token_id_types])
        kl = calc_kl_divergence(p, q)

        print(f'distance={CONTEXT_SIZE - n:>2} kl={kl:.3f} {"<" if kl <= kl_chance else ">"} {kl_chance:.3f}')

    print()
