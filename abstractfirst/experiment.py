import numpy as np
import pandas as pd
from spacy.tokens import Doc
from pyitlib import discrete_random_variable as drv
from scipy import sparse
from sklearn.metrics.cluster import adjusted_mutual_info_score
import attr
from typing import List, Dict
from sortedcontainers import SortedSet

from abstractfirst.pre_processing import make_age2docs
from abstractfirst.co_occurrence import collect_left_and_right_co_occurrences, make_sparse_co_occurrence_mat
from abstractfirst.params import Params
from abstractfirst.reconstruct import plot_reconstructions
from abstractfirst import configs


def prepare_data(params: Params,
                 ) -> Dict[str, Doc]:

    age2docs: Dict[str, List[Doc]] = make_age2docs(params)

    # make each age bin equally large
    age2doc = {}
    for age, docs in age2docs.items():

        doc_combined = Doc.from_docs(docs)
        age2doc[age] = doc_combined
        print(f'Num tokens at age={age} is {len(doc_combined):,}')

    return age2doc


def measure_dvs(params: Params,
                doc: Doc,
                targets: SortedSet,
                ) -> pd.DataFrame:
    """
    collect all DVs, and return df with single row
    """

    name2col = {}

    # get co-occurrence data
    co_data = collect_left_and_right_co_occurrences(doc, targets, params)

    # for each direction (left, right, both)
    for direction in configs.Conditions.directions:

        params = attr.evolve(params, direction=direction)
        name2col.setdefault(f'direction', []).append(direction)

        # adjust max_sum
        if direction == 'b' and isinstance(params.max_sum_one_direction, int):
            params = attr.evolve(params, max_sum_one_direction=params.max_sum_one_direction * 2)

        co_mat_coo: sparse.coo_matrix = make_sparse_co_occurrence_mat(co_data, params)
        name2col.setdefault(f'x-tokens', []).append(co_mat_coo.sum().item())
        name2col.setdefault(f'x-types ', []).append(co_mat_coo.shape[0])
        name2col.setdefault(f'y-types ', []).append(co_mat_coo.shape[1])

        # svd
        # don't use sparse svd: doesn't result in accurate reconstruction.
        # don't normalize before svd: otherwise relative differences between rows and columns are lost
        s = np.linalg.svd(co_mat_coo.toarray(), compute_uv=False)
        assert np.max(s) == s[0]
        name2col.setdefault(f' s1/s1+s2', []).append(s[0] / (s[0] + s[1]))
        name2col.setdefault(f's1/sum(s)', []).append(s[0] / np.sum(s))

        # info theory analysis
        xs, ys = co_data.make_rvs(direction)  # todo also get zs + calc interaction info
        xy = np.vstack((xs, ys))
        je = drv.entropy_joint(xy)
        name2col.setdefault(f'nxy', []).append(drv.entropy_conditional(xs, ys).item() / je)
        name2col.setdefault(f'nyx', []).append(drv.entropy_conditional(ys, xs).item() / je)
        name2col.setdefault(f'nmi', []).append(drv.information_mutual_normalised(xs, ys, norm_factor='XY').item())
        name2col.setdefault(f'ami', []).append(adjusted_mutual_info_score(xs, ys, average_method="arithmetic"))
        name2col.setdefault(f' je', []).append(je)

        if configs.Fig.max_projection > 0:
            plot_reconstructions(co_mat_coo, params, max_dim=configs.Fig.max_projection)

    df = pd.DataFrame(data=name2col, columns=name2col.keys())

    return df
