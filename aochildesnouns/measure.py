import numpy as np
from typing import Dict
from pyitlib import discrete_random_variable as drv
from scipy import sparse
from sklearn.metrics.cluster import adjusted_mutual_info_score
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
        nii = drv.information_interaction(xyz).item() / xyz_je
    else:
        nii = np.nan  # need 3 rvs to compute interaction information
    xs, ys = co_data.get_x_y(params.direction)
    xy = np.vstack((xs, ys))
    xy_je = drv.entropy_joint(xy)

    # compute entropy on permuted data for de-biasing estimates
    bias_xy = np.mean([drv.entropy_conditional(np.random.permutation(xs), np.random.permutation(ys), base=2).item()
                       for _ in range(10)])
    bias_yx = np.mean([drv.entropy_conditional(np.random.permutation(ys), np.random.permutation(xs), base=2).item()
                      for _ in range(10)])
    print(f'bias_xy={bias_xy:.4f}')
    print(f'bias_yx={bias_yx:.4f}')

    xy_ce = drv.entropy_conditional(xs, ys).item()
    yx_ce = drv.entropy_conditional(ys, xs).item()
    res[' xy'] = xy_ce  # biased
    res[' yx'] = yx_ce
    res['dxy'] = bias_xy - xy_ce  # de-biased
    res['dyx'] = bias_yx - yx_ce
    res['nxy'] = xy_ce / xy_je  # biased + normalized
    res['nyx'] = yx_ce / xy_je
    # res['nii'] = nii
    # res['nmi'] = drv.information_mutual_normalised(xs, ys, norm_factor='XY').item()
    # res['ami'] = adjusted_mutual_info_score(xs, ys, average_method="arithmetic")
    # res[' je'] = xy_je

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
    max_row_id = np.argmax(projection1.sum(axis=1))
    max_col_id = np.argmax(projection1.sum(axis=0))
    print(f'Word with largest sum={np.max(projection1.sum(axis=1))} in first projection row="{row_words[max_row_id]}"')
    print(f'Word with largest sum={np.max(projection1.sum(axis=0))} in first projection col="{col_words[max_col_id]}"')

    return res
