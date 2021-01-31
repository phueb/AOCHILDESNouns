from typing import Tuple, Optional, List, Set
from scipy.cluster.hierarchy import linkage, dendrogram

import numpy as np
from abstractfirst.co_occurrence import CoData

from abstractfirst import configs


def split(l, split_size):
    for i in range(0, len(l), split_size):
        yield l[i:i + split_size]


def load_words(file_name: str
               ) -> Set[str]:
    p = configs.Dirs.words / f'{file_name}.txt'
    res = p.read_text().split('\n')
    return set(res)


def to_pyitlib_format(co_data: CoData,
                      ) -> Tuple[List[int], List[int], List[int]]:
    """
    convert data to three lists, x, y, z, each with realisations of one discrete RV.
    """
    xs = []
    ys = []
    zs = []
    for i, j, v in zip(co_mat.row, co_mat.col, co_mat.data):
        if v > 0:
            xs += [i] * v
            ys += [j] * v

    return xs, ys, zs


def cluster(mat: np.ndarray,
            dg0: Optional[dict] = None,
            dg1: Optional[dict] = None,
            method: str = 'complete',
            metric: str = 'euclidean'):

    if dg0 is None:
        print('Clustering rows...')
        lnk0 = linkage(mat, method=method, metric=metric, optimal_ordering=True)
        dg0 = dendrogram(lnk0,
                         ax=None,
                         color_threshold=None,
                         no_labels=True,
                         no_plot=True)
    res = mat[dg0['leaves'], :]  # reorder rows

    if dg1 is None:
        print('Clustering cols...')
        lnk1 = linkage(mat.T, method=method, metric=metric, optimal_ordering=True)
        dg1 = dendrogram(lnk1,
                         ax=None,
                         color_threshold=None,
                         no_labels=True,
                         no_plot=True)
    res = res[:, dg1['leaves']]  # reorder cols

    return res, dg0, dg1
