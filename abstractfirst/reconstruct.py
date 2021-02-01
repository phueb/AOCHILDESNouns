import numpy as np
from scipy import sparse
from sklearn.preprocessing import quantile_transform
from pathlib import Path
from typing import List, Optional

from abstractfirst import configs
from abstractfirst.figs import plot_heatmap
from abstractfirst.util import cluster
from abstractfirst.params import Params


def make_path(age: int,
              direction: str,
              ) -> Path:
    res = configs.Dirs.images / f'age_{age}' / direction
    if not res.is_dir():
        res.mkdir(parents=True)
    return res


def make_title(params: Params,
               excluded_attrs: Optional[List[str]] = None,
               ) -> str:
    if excluded_attrs is None:
        excluded_attrs = ['num_days', 'num_tokens_per_age', 'max_sum']

    old = params.__repr__()
    old = old.replace('Params(', '')
    res = ''
    for line in old.split(', '):
        if [a for a in excluded_attrs if a in line]:
            continue
        new_line = line.strip(")")
        delim = "" if "{" in new_line else "\n"  # prevent splitting of lines that should not be split
        res += f'{new_line}{delim}'
    return res


def plot_reconstructions(co_mat_coo: sparse.coo_matrix,
                         params: Params,
                         max_dim: int,
                         plot_interval: int = 1,
                         ):

    # remove skew for better visualisation
    co_mat_normal_csr: sparse.csr_matrix = quantile_transform(co_mat_coo,
                                                              axis=0,
                                                              output_distribution='normal',
                                                              n_quantiles=co_mat_coo.shape[0],
                                                              copy=True,
                                                              ignore_implicit_zeros=True)
    # don't use sparse svd: doesn't result in accurate reconstruction
    co_mat_normal_dense = co_mat_normal_csr.toarray()
    U, s, VT = np.linalg.svd(co_mat_normal_dense, compute_uv=True)
    fig_size = (co_mat_normal_dense.shape[1] // 1000 + 1 * 2,
                co_mat_normal_dense.shape[0] // 1000 + 1 * 2 + 0.5,
                )
    print(f'fig size={fig_size}')
    print(params.direction)
    base_title = make_title(params)
    base_title += f'num co-occurrences={np.sum(co_mat_coo)}\n'
    base_title += f'age={params.age}\n'
    # plot projection of co_mat onto sing dims
    dg0, dg1 = None, None
    projections = np.zeros(co_mat_normal_dense.shape, dtype=np.float)
    num_s = sum(s > 0)
    for dim_id in range(max_dim):
        projection = s[dim_id] * U[:, dim_id].reshape(-1, 1) @ VT[dim_id, :].reshape(1, -1)
        projection_clustered, dg0, dg1 = cluster(projection, dg0, dg1)
        projections += projection_clustered
        if dim_id % plot_interval == 0:
            plot_heatmap(projections,
                         title=base_title + f'projections={dim_id}/{num_s}',
                         save_path=make_path(params.age, params.direction) / f'dim{dim_id:04}.png',
                         vmin=np.min(co_mat_normal_dense),
                         vmax=np.max(co_mat_normal_dense),
                         figsize=fig_size,
                         )
    plot_heatmap(cluster(co_mat_normal_dense, dg0, dg1)[0],
                 title=base_title + 'original',
                 save_path=make_path(params.age, params.direction) / f'dim{num_s:04}.png',
                 vmin=np.min(co_mat_normal_dense),
                 vmax=np.max(co_mat_normal_dense),
                 figsize=fig_size,
                 )