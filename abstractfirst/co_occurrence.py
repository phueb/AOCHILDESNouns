from typing import List, Tuple
import attr
from spacy.tokens import Doc
import numpy as np
import itertools
from scipy import sparse
from sortedcontainers import SortedSet


from abstractfirst.params import Params
from abstractfirst import configs


@attr.s
class CoData:
    """store co-occurrence data, for left, right, and both context directions separately"""
    row_ids_l = attr.ib()
    col_ids_l = attr.ib()

    row_ids_r = attr.ib()
    col_ids_r = attr.ib()

    row_ids_b = attr.ib()
    col_ids_b = attr.ib()

    def get_x_y(self, direction: str,
                ) -> Tuple[List[int], List[int]]:
        """get realizations of two random variables, x, and y, corresponding to row and column of co matrix"""
        if direction == 'l':
            return self.row_ids_l, self.col_ids_l
        elif direction == 'r':
            return self.row_ids_r, self.col_ids_r
        elif direction == 'b':
            return self.row_ids_b, self.col_ids_b
        else:
            raise AttributeError('Invalid arg')

    def get_x_y_z(self):
        """get realizations of three random variables, x, y, z, where y are left and z are right context type ids"""
        assert self.row_ids_r == self.row_ids_l
        return self.row_ids_r, self.col_ids_l, self.col_ids_r


def collect_left_and_right_co_occurrences(doc: Doc,
                                          targets: SortedSet,
                                          params: Params,
                                          ) -> CoData:
    """
    collect co-occurrences in format suitable for scipy.sparse.coo.

    note: collects left, right and "both" co-occurrences.
    in the latter case, both a right and left co-occurrence are recorded separately.
    """

    cw2id = {}
    lw2id = {}
    rw2id = {}
    row_ids_l = []
    row_ids_r = []
    row_ids_b = []
    col_ids_l = []
    col_ids_r = []
    col_ids_b = []
    for n in range(len(doc) - 2):

        if n == 0:
            continue
        if doc[n].text not in targets:
            continue
        if not params.targets_control and doc[n].tag_ not in params.tags:  # do not filter when using control targets
            continue

        # handle punctuation
        lwo = 1  # left word offset
        rwo = 1  # right word offset
        while True:
            is_break = True
            # left word, center word, right word
            if params.lemmas:
                lw = doc[n - lwo].lemma_
                cw = doc[n].lemma_
                rw = doc[n + rwo].lemma_
            else:
                lw = doc[n - lwo].text
                cw = doc[n].text
                rw = doc[n + rwo].text

            if lw in configs.Data.punctuation:
                if params.punctuation == 'remove':
                    lwo += 1
                    is_break = False
                elif params.punctuation == 'merge':
                    lw = configs.Data.eos
            if rw in configs.Data.punctuation:
                if params.punctuation == 'remove':
                    rwo += 1
                    is_break = False
                elif params.punctuation == 'merge':
                    rw = configs.Data.eos

            if is_break:
                break

        # collect left co-occurrence
        row_ids_l.append(cw2id.setdefault(cw, len(cw2id)))
        col_ids_l.append(lw2id.setdefault(lw, len(lw2id)))

        # collect right co-occurrence
        row_ids_r.append(cw2id.setdefault(cw, len(cw2id)))
        col_ids_r.append(rw2id.setdefault(rw, len(rw2id)))

        # combine both co-occurrences
        row_ids_b.append(row_ids_l[-1])
        row_ids_b.append(row_ids_r[-1])
        col_ids_b.append(col_ids_l[-1])
        col_ids_b.append(col_ids_r[-1])

    return CoData(row_ids_l, col_ids_l,
                  row_ids_r, col_ids_r,
                  row_ids_b, col_ids_b,
                  )


def make_sparse_co_occurrence_mat(co_data: CoData,
                                  params: Params,
                                  ) -> sparse.coo_matrix:
    """make sparse matrix (contexts are col labels, targets are row labels)"""

    if params.direction == 'l':
        data = [1] * len(co_data.col_ids_l)
        row_ids = co_data.row_ids_l
        col_ids = co_data.col_ids_l
        assert max(co_data.col_ids_l) + 1 == len(set(co_data.col_ids_l))  # check that col ids are consecutive
    elif params.direction == 'r':
        data = [1] * len(co_data.col_ids_r)
        row_ids = co_data.row_ids_r
        col_ids = co_data.col_ids_r
        assert max(co_data.col_ids_r) + 1 == len(set(co_data.col_ids_r))  # check that col ids are consecutive
    elif params.direction == 'b':
        data = [1] * len(co_data.col_ids_l + co_data.col_ids_r)
        # interleave lists - so that result can be truncated without affecting left or right contexts more
        row_ids = list(itertools.chain(*zip(co_data.row_ids_l, co_data.row_ids_r)))
        col_ids = list(itertools.chain(*zip(co_data.col_ids_l, co_data.row_ids_r)))
    else:
        raise AttributeError('Invalid arg')

    # constrain the sum of the matrix by removing some data
    max_sum = configs.Data.max_sum[params.targets_control]
    assert isinstance(max_sum, int) or max_sum is None
    if max_sum is not None:
        tmp = np.vstack((data, row_ids, col_ids)).T
        data = tmp[:max_sum, 0]
        row_ids = tmp[:max_sum, 1]
        col_ids = tmp[:max_sum, 2]

    res = sparse.coo_matrix((data, (row_ids, col_ids)))

    # check that there are no skipped column ids or zero columns
    res_csc = res.tocsc()
    for col_id in range(res.shape[1]):
        assert col_id in col_ids
        assert res_csc[:, col_id].sum() != 0

    print(f'Co-occurrence matrix has sum={res.sum():,} and shape={res.shape}')

    return res
