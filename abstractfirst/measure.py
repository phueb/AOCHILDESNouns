import numpy as np
from typing import Dict
from pyitlib import discrete_random_variable as drv
from scipy import sparse
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.preprocessing import normalize

from abstractfirst.co_occurrence import make_sparse_co_occurrence_mat, CoData
from abstractfirst.params import Params
from abstractfirst.reconstruct import plot_reconstructions
from abstractfirst import configs


def measure_dvs(params: Params,
                co_data: CoData,
                ) -> Dict[str, float]:
    """
    collect all DVs in a single condition.

    a condition is a specific configuration of IV realizations
    """

    res = {}

    co_mat_coo: sparse.coo_matrix = make_sparse_co_occurrence_mat(co_data, params)
    co_mat_csr: sparse.csr_matrix = co_mat_coo.tocsr()

    # type and token frequency
    res[f'x-tokens'] = co_mat_coo.sum().item() // 2 if params.direction == 'b' else co_mat_coo.sum().item()
    res[f'x-types '] = co_mat_coo.shape[0]
    res[f'y-types '] = co_mat_coo.shape[1]

    # normalize columns
    if params.normalize_cols:
        co_mat_csr = normalize(co_mat_csr, axis=1, copy=False)
        print(co_mat_csr.sum())

    # svd
    # don't use sparse svd: doesn't result in accurate reconstruction.
    # don't normalize before svd: otherwise relative differences between rows and columns are lost
    s = np.linalg.svd(co_mat_csr.toarray(), compute_uv=False)
    assert np.max(s) == s[0]
    res[f' s1/s1+s2'] = s[0] / (s[0] + s[1])
    res[f's1/sum(s)'] = s[0] / np.sum(s)

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
    res[f'nxy'] = drv.entropy_conditional(xs, ys).item() / xy_je
    res[f'nyx'] = drv.entropy_conditional(ys, xs).item() / xy_je
    res[f'nii'] = nii
    res[f'nmi'] = drv.information_mutual_normalised(xs, ys, norm_factor='XY').item()
    res[f'ami'] = adjusted_mutual_info_score(xs, ys, average_method="arithmetic")
    res[f' je'] = xy_je

    if configs.Fig.max_projection > 0:
        plot_reconstructions(co_mat_coo, params, max_dim=configs.Fig.max_projection)

    return res
