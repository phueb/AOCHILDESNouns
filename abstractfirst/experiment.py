import numpy as np
import pandas as pd
from spacy.tokens import Doc
from pyitlib import discrete_random_variable as drv
from scipy import sparse
from sklearn.metrics.cluster import adjusted_mutual_info_score
from tabulate import tabulate
import attr
from typing import List, Dict
from sortedcontainers import SortedSet

from abstractfirst.binned import make_age_bin2data, adjust_binned_data
from abstractfirst.co_occurrence import collect_left_and_right_co_occurrences, make_sparse_co_occurrence_mat
from abstractfirst.params import Params
from abstractfirst.reconstruct import plot_reconstructions


def prepare_data(params: Params,
                 verbose: bool = False,
                 ) -> Dict[float, str]:

    age_bin2text_unadjusted = make_age_bin2data(params)
    
    if verbose:
        print('Raw age bins:')
        for age, txt in age_bin2text_unadjusted.items():
            print(f'age bin={age:>4} num tokens={len(txt.split()):,}')

    age_bin2text = adjust_binned_data(age_bin2text_unadjusted, params.num_tokens_per_bin)

    if verbose:
        print('Adjusted age bins:')
        for age, txt in sorted(age_bin2text.items(), key=lambda i: i[0]):
            num_tokens_adjusted = len(txt.split())
            print(f'age bin={age:>4} num tokens={num_tokens_adjusted :,}')
            assert num_tokens_adjusted >= params.num_tokens_per_bin

    return age_bin2text


def collect_dvs(params: Params,
                doc: Doc,
                targets: SortedSet,
                max_projection: int = 3,
                ) -> pd.DataFrame:
    """
    collect all DVs, and return df with single row
    """

    name2col = {}
    directions = ['l', 'r', 'b']

    # get co-occurrence data
    co_data = collect_left_and_right_co_occurrences(doc, targets, params)

    # for each direction (left, right, both)
    for direction in directions:

        print(f'direction={direction}')
        params = attr.evolve(params, direction=direction)
        name2col.setdefault(f'direction', []).append(direction)

        co_mat_coo: sparse.coo_matrix = make_sparse_co_occurrence_mat(co_data, params)
        name2col.setdefault(f'x-types', []).append(co_mat_coo.shape[0])
        name2col.setdefault(f'y-types', []).append(co_mat_coo.shape[1])

        # svd
        print('SVD...')
        # don't use sparse svd: doesn't result in accurate reconstruction.
        # don't normalize before svd: otherwise relative differences between rows and columns are lost
        s = np.linalg.svd(co_mat_coo.toarray(), compute_uv=False)
        assert np.max(s) == s[0]
        with np.printoptions(precision=2, suppress=True):
            print(s[:4])
        name2col.setdefault(f's1-s2/s1+s2', []).append((s[0] - s[1]) / (s[0] + s[1]))
        name2col.setdefault(f'  s1/sum(s)', []).append(s[0] / np.sum(s))

        # info theory analysis
        print('Info theory analysis...')
        xs, ys = co_data.make_rvs(direction)  # todo also get zs + calc interaction info
        xy = np.vstack((xs, ys))
        je = drv.entropy_joint(xy)
        name2col.setdefault(f'nxy', []).append(drv.entropy_conditional(xs, ys).item() / je)
        name2col.setdefault(f'nyx', []).append(drv.entropy_conditional(ys, xs).item() / je)
        name2col.setdefault(f'nmi', []).append(drv.information_mutual_normalised(xs, ys, norm_factor='XY').item())
        name2col.setdefault(f'ami', []).append(adjusted_mutual_info_score(xs, ys, average_method="arithmetic"))
        name2col.setdefault(f' je', []).append(je)

        if max_projection > 0:
            plot_reconstructions(co_mat_coo, params, max_dim=max_projection)

    df = pd.DataFrame(data=name2col, columns=name2col.keys())

    print(params)
    print(tabulate(df,
                   headers=list(df.columns),
                   tablefmt='simple'))

    return df