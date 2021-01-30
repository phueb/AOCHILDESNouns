from typing import Set, List, Optional, Tuple, Dict
from spacy.tokens import Doc, Token
import numpy as np
import itertools
from scipy import sparse
from sortedcontainers import SortedSet


def collect_left_and_right_co_occurrences(doc: Doc,
                                          targets: SortedSet,
                                          left_only: bool = False,
                                          right_only: bool = False,
                                          allowed_tags: Optional[Set[str]] = None,
                                          ) -> Tuple[List, List, List]:
    """
    collect left and right co-occurrences in format suitable for scipy.sparse.coo.

    note: left and right contexts receive unique column ids.
    this is achieved by incrementing the ids in each dictionary (left and right) by consulting each other's length
    """

    assert not (right_only and left_only)

    if allowed_tags is None:
        allowed_tags = {'NN', 'NNS'}

    print(f'Collecting co-occurrences with:\n'
          f'left={not right_only}\n'
          f'right={not left_only}\n'
          f'allowed_tags={allowed_tags}')

    cw2id = {}
    lw2id = {}
    rw2id = {}
    row_ids_l = []
    row_ids_r = []
    col_ids_l = []
    col_ids_r = []
    for n, cw in enumerate(doc[:-2]):

        if n == 0:
            continue
        if cw.text not in targets:
            continue
        if cw.tag_ not in allowed_tags:
            continue

        # left word, center word, right word
        lw = doc[n - 1].text
        cw = cw.text
        rw = doc[n + 1].text

        # collect left co-occurrence
        row_ids_l.append(cw2id.setdefault(cw, len(cw2id)))
        col_ids_l.append(lw2id.setdefault(lw, len(lw2id) if left_only else len(lw2id) + len(rw2id)))

        # collect right co-occurrence
        row_ids_r.append(cw2id.setdefault(cw, len(cw2id)))
        col_ids_r.append(rw2id.setdefault(rw, len(rw2id) if right_only else len(lw2id) + len(rw2id)))

    if left_only:
        data = [1] * len(col_ids_l)
        row_ids = row_ids_l
        col_ids = col_ids_l
        assert max(col_ids_l) + 1 == len(set(col_ids_l))  # check that col ids are consecutive
    elif right_only:
        data = [1] * len(col_ids_r)
        row_ids = row_ids_r
        col_ids = col_ids_r
        assert max(col_ids_r) + 1 == len(set(col_ids_r))  # check that col ids are consecutive
    else:
        data = [1] * len(col_ids_l + col_ids_r)
        # interleave lists - so that result can be truncated without affecting left or right contexts more
        row_ids = list(itertools.chain(*zip(row_ids_l, row_ids_r)))
        col_ids = list(itertools.chain(*zip(col_ids_l, row_ids_r)))

    print(f'Collected {len(data):,} co-occurrences')

    return data, row_ids, col_ids


def make_sparse_co_occurrence_mat(data: List[int],
                                  row_ids: List[int],
                                  col_ids: List[int],
                                  max_sum: Optional[int] = None,
                                  ) -> sparse.coo_matrix:
    """make sparse matrix (contexts are col labels, targets are row labels)"""

    # constrain the sum of the matrix by removing some data
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
