import numpy as np
from typing import Dict
from pyitlib import discrete_random_variable as drv
from scipy import sparse
from sklearn.metrics.cluster import adjusted_mutual_info_score
import pandas as pd
from sklearn.preprocessing import normalize
import pickle

from aochildesnouns.co_occurrence import CoData
from aochildesnouns.params import Params
from aochildesnouns.reconstruct import plot_reconstructions
from aochildesnouns.util import calc_projection
from aochildesnouns import configs


def measure_dvs(params: Params,
                co_data: CoData,
                ) -> Dict[str, float]:
    """
    collect all DVs in a single condition.

    a condition is a specific configuration of IV realizations
    """

    res = {}

    co_mat_coo: sparse.coo_matrix = co_data.as_matrix(params.direction)
    co_mat_csr: sparse.csr_matrix = co_mat_coo.tocsr()

    # save for offline analysis
    path_to_pkl = configs.Dirs.co_data / f'co_data_age={params.age}' \
                                         f'_punct={params.punctuation}' \
                                         f'_contr={params.targets_control}' \
                                         f'_lemma={params.lemmas}.pkl'
    with path_to_pkl.open('wb') as f:
        pickle.dump(co_data, f)

    # type and token frequency
    res['x-tokens'] = co_mat_coo.sum().item() // 2 if params.direction == 'b' else co_mat_coo.sum().item()
    res['x-types'] = co_mat_coo.shape[0]
    res['y-types'] = co_mat_coo.shape[1]

    # normalize columns
    if params.normalize_cols:
        co_mat_csr = normalize(co_mat_csr, axis=1, copy=False)
        print(co_mat_csr.sum())

    # svd
    # don't use sparse svd: doesn't result in accurate reconstruction.
    # don't normalize before svd: otherwise relative differences between rows and columns are lost
    u, s, vt = np.linalg.svd(co_mat_csr.toarray(), compute_uv=True)
    assert np.max(s) == s[0]
    res[f's1/sum(s)'] = s[0] / np.sum(s)
    res[f'frag'] = 1 - (s[0] / np.sum(s))

    # info theory analysis
    if params.direction == 'b':
        xs, ys, zs = co_data.get_x_y_z()
        xyz = np.vstack((xs, ys, zs))
        xyz_je = drv.entropy_joint(xyz)
        ii = drv.information_interaction(xyz).item()
        nii = ii / xyz_je
    else:
        ii = np.nan  # need 3 rvs to compute interaction information
        nii = np.nan
    xs, ys = co_data.get_x_y(params.direction)
    xy = np.vstack((xs, ys))
    xy_je = drv.entropy_joint(xy)

    # in order to compare entropies between groups, we need to map them to the same scale:
    # to do this, we normalize by the joint entropy (but there are probably many other ways)

    xy_ce = drv.entropy_conditional(xs, ys).item()
    yx_ce = drv.entropy_conditional(ys, xs).item()
    res['xy'] = xy_ce  # biased
    res['yx'] = yx_ce

    res['joint'] = xy_je
    res['mi'] = drv.information_mutual(xs, ys)
    res['nmi'] = drv.information_mutual_normalised(xs, ys)

    res['x'] = drv.entropy(xs)
    res['y'] = drv.entropy(ys)

    res['x/joint'] = drv.entropy(xs) / xy_je
    res['y/joint'] = drv.entropy(ys) / xy_je

    res['xy/joint'] = xy_ce / xy_je
    res['yx/joint'] = yx_ce / xy_je

    res['ii'] = ii
    res['ii/joint'] = nii
    # res['nmi'] = drv.information_mutual_normalised(xs, ys, norm_factor='XY').item()
    # res['ami'] = adjusted_mutual_info_score(xs, ys, average_method="arithmetic")
    res['je'] = xy_je

    # round
    for k, v in res.items():
        if isinstance(v, float):
            res[k] = round(v, 3)

    if configs.Fig.max_projection > 0:
        plot_reconstructions(co_mat_coo, params, max_dim=configs.Fig.max_projection)

    # which row or column is most active in projection on first singular dim?
    # note: if lemmas=True, row words may include experimental targets
    # because lemmas of control target plural nouns are singular nouns
    row_words, col_words = co_data.get_words_ordered_by_id(params.direction)
    if len(row_words) != co_mat_csr.shape[0]:
        raise RuntimeError(f'Number of row words ({len(row_words)}) != Number of rows ({co_mat_csr.shape[0]})')
    if len(col_words) != co_mat_csr.shape[1]:
        raise RuntimeError(f'Number of column words ({len(col_words)}) != Number of columns ({co_mat_csr.shape[1]})')
    projection1 = calc_projection(u, s, vt, 0)
    max_row_id = np.argmax(projection1.sum(axis=1)).item()
    max_col_id = np.argmax(projection1.sum(axis=0)).item()
    print(f'Word with largest sum={np.max(projection1.sum(axis=1))} in first projection row="{row_words[max_row_id]}"')
    print(f'Word with largest sum={np.max(projection1.sum(axis=0))} in first projection col="{col_words[max_col_id]}"')

    # find "entropy-maximizing contexts" (so-called, but has no direct relation to entropy)
    top_k = 10
    p1_sum0 = projection1.sum(axis=0)
    idx = np.argpartition(p1_sum0, -top_k)[-top_k:]  # Indices not sorted
    idx_sorted = idx[np.argsort(p1_sum0[idx])][::-1]  # Indices sorted by value from largest to smallest
    df = pd.DataFrame({'loading': [p1_sum0[i] for i in idx_sorted],
                       'word': [col_words[i] for i in idx_sorted],
                       'frequency': [co_mat_csr[:, i].sum().item() for i in idx_sorted]})
    # print('Entropy-maximizing contexts:')
    # print(df.to_latex(index=False))
    total_freq_of_entropy_max_contexts = co_mat_csr[:, idx_sorted].sum().item() / co_mat_csr.sum().item()
    print(f'prop. of total frequency that are top-10 entropy-max contexts = {total_freq_of_entropy_max_contexts:,}')

    # find most fragmenting contexts
    idx = np.argpartition(p1_sum0, top_k)[:top_k]  # Indices not sorted
    idx_sorted = idx[np.argsort(p1_sum0[idx])]  # Indices sorted by value from smallest to largest
    df = pd.DataFrame({'Loading': [p1_sum0[i] for i in idx_sorted],
                       'Left-context': [col_words[i] for i in idx_sorted],
                       'Frequency': [co_mat_csr[:, i].sum().item() for i in idx_sorted]})
    # print('Fragmenting contexts:')
    # print(df.to_latex(index=False))
    total_freq_of_entropy_max_contexts = co_mat_csr[:, idx_sorted].sum().item() / co_mat_csr.sum().item()
    print(f'prop. of total frequency that are top-10 fragmenting contexts = {total_freq_of_entropy_max_contexts:,}')

    return res
