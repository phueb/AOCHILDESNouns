from typing import Set, List, Optional, Tuple, Dict
import attr
from spacy.tokens import Doc, Token
import numpy as np
import itertools
from scipy import sparse
from sortedcontainers import SortedSet


from abstractfirst.params import Params


@attr.s
class CoData:
    row_ids_l = attr.ib()
    col_ids_l = attr.ib()

    row_ids_r = attr.ib()
    col_ids_r = attr.ib()

    row_ids_b = attr.ib()
    col_ids_b = attr.ib()

    def make_rvs(self, direction: str,
                 ) -> Tuple[List[int], List[int]]:
        if direction == 'l':
            return self.row_ids_l, self.col_ids_l
        elif direction == 'r':
            return self.row_ids_r, self.col_ids_r
        elif direction == 'b':
            return self.row_ids_b, self.col_ids_b
        else:
            raise AttributeError('Invalid arg')


def collect_left_and_right_co_occurrences(doc: Doc,
                                          targets: SortedSet,
                                          params: Params,
                                          ) -> CoData:
    """
    collect left and right co-occurrences in format suitable for scipy.sparse.coo.

    note: also collects combined (left + right) co-occurrences.
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
    for n, cw in enumerate(doc[:-2]):

        if n == 0:
            continue
        if cw.text not in targets:
            continue
        if cw.tag_ not in params.tags:
            continue

        # left word, center word, right word
        if params.lemmas:
            lw = doc[n - 1].lemma
            cw = cw.lemma
            rw = doc[n + 1].lemma
        else:
            lw = doc[n - 1].text
            cw = cw.text
            rw = doc[n + 1].text

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

    print(f'Collected {len(row_ids_r):,} right and {len(row_ids_l):,} left co-occurrences')

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
    assert isinstance(params.max_sum_one_direction, int) or params.max_sum_one_direction is None
    if params.max_sum_one_direction is not None:
        tmp = np.vstack((data, row_ids, col_ids)).T
        data = tmp[:params.max_sum_one_direction, 0]
        row_ids = tmp[:params.max_sum_one_direction, 1]
        col_ids = tmp[:params.max_sum_one_direction, 2]

    res = sparse.coo_matrix((data, (row_ids, col_ids)))

    # check that there are no skipped column ids or zero columns
    res_csc = res.tocsc()
    for col_id in range(res.shape[1]):
        assert col_id in col_ids
        assert res_csc[:, col_id].sum() != 0

    print(f'Co-occurrence matrix has sum={res.sum():,} and shape={res.shape}')

    return res
