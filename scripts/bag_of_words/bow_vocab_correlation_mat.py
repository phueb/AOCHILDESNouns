"""
Research questions:
1. Does correlation matrix for probes look more hierarchical in partition 1?
"""

import attr
from sklearn.decomposition import PCA

from preppy import PartitionedPrep
from preppy.utils import make_windows_mat

from wordplay.figs import plot_heatmap
from wordplay.representation import make_bow_token_representations
from wordplay.utils import to_corr_mat, cluster
from wordplay.params import PrepParams
from wordplay.docs import load_docs
from wordplay.pos import load_pos_words

# /////////////////////////////////////////////////////////////////

CORPUS_NAME = 'childes-20180319'

docs = load_docs(CORPUS_NAME)
params = PrepParams(num_parts=2, context_size=1, num_types=4096)
prep = PartitionedPrep(docs, **attr.asdict(params))


# /////////////////////////////////////////////////////////////////

N_COMPONENTS = 32  # 512
PART_IDS = [0, 1]  # this is useful because clustering of second corr_mat is based on dg0 and dg1 of first


dg0, dg1 = None, None
for part_id in PART_IDS:

    # a window is [x1, x2, x3, x4, x5, x6, x7, y] if context_size=7
    windows_mat = make_windows_mat(prep.reordered_parts[part_id],
                                   prep.num_windows_in_part,
                                   prep.num_tokens_in_window)
    print('shape of windows_mat={}'.format(windows_mat.shape))
    token_reps = make_bow_token_representations(windows_mat, prep.store.types)
    print('shape of reps={}'.format(token_reps.shape))
    assert len(token_reps) == prep.store.num_types, (len(token_reps), prep.store.num_types)

    # pca
    pca = PCA(n_components=N_COMPONENTS)
    token_reps = pca.fit_transform(token_reps)
    print('shape after PCA={}'.format(token_reps.shape))

    # plot
    corr_mat = to_corr_mat(token_reps)
    print('shape of corr_mat={}'.format(corr_mat.shape))
    clustered_corr_mat, rls, cls, dg0, dg1 = cluster(corr_mat, dg0, dg1, prep.store.types, prep.store.types)
    pos_words = load_pos_words(f'{CORPUS_NAME}-nouns')
    rls = [rl if rl in pos_words else '' for rl in rls]
    plot_heatmap(clustered_corr_mat, rls, cls, save_name=None, label_interval=1)
    print('------------------------------------------------------')

