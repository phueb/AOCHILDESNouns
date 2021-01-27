from typing import Tuple, Set, List

import numpy as np
from scipy.sparse import dok_matrix

from abstractfirst import configs


def split(l, split_size):
    for i in range(0, len(l), split_size):
        yield l[i:i + split_size]


def load_words(file_name: str
               ) -> Set[str]:
    p = configs.Dirs.words / f'{file_name}.txt'
    res = p.read_text().split('\n')
    return set(res)


def to_pyitlib_format(co_mat: dok_matrix) -> Tuple[List[int], List[int]]:
    """
    convert data in co-occurrence matrix to two lists, each with realisations of one discrete RV.
    """
    xs = []
    ys = []
    for (i, j), num in co_mat.items():
        if num > 0:
            xs += [i] * num
            ys += [j] * num

    return xs, ys
