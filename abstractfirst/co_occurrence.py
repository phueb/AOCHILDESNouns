import random
from typing import Set, List, Optional, Tuple, Dict
from cytoolz import itertoolz

from scipy import sparse

from sortedcontainers import SortedSet


def make_sparse_co_occurrence_mat(tokens: List[str],
                                  targets: SortedSet,
                                  shuffle_tokens: bool = False,
                                  stop_n: Optional[int] = None,  # stop collection if sum of matrix is stop_n
                                  ) -> sparse.coo_matrix :
    """
    targets in rows, 1-left and 1-right contexts in columns
    """

    window_size = 3  # (1-left context word, target, 1-right context word)

    print('Making co-occurrence matrix...')

    # shuffle
    if shuffle_tokens:
        print('WARNING: Shuffling tokens')
        tokens = tokens.copy()
        random.shuffle(tokens)

    target2row_id = {t: n for n, t in enumerate(targets)}
    context2col_id = {}

    # make sparse matrix (contexts/y-words in rows, targets/x-words in cols)
    data = []
    row_ids = []
    col_ids = []
    for lw, cw, rw in itertoolz.sliding_window(window_size, tokens):  # left word, center word, right word

        if cw not in targets:
            continue

        # get ids
        row_id = target2row_id[cw]
        col_id_left = context2col_id.setdefault(rw, len(context2col_id))
        col_id_right = context2col_id[rw]

        # collect left co-occurrence
        row_ids.append(row_id)
        col_ids.append(col_id_left)
        data.append(1)  # it is okay to append 1s because final value is sum over 1s in same position in matrix

        # collect right co-occurrence
        row_ids.append(row_id)
        col_ids.append(col_id_right)
        data.append(1)  # it is okay to append 1s because final value is sum over 1s in same position in matrix

        if stop_n is not None:
            if len(data) >= stop_n:
                print(f'Sopping co-occurrence collection early because sum>=stop_n={len(data)}')
                break

    # make sparse matrix once (updating it is expensive)
    res = sparse.coo_matrix((data, (row_ids, col_ids)))

    if res.shape[0] != len(targets):
        print(f'WARNING: Number of targets={len (targets)} but number of rows={res.shape[0]}.'
              f'This probably means not all targets occur in this corpus partition.')

    print(f'Done. Co-occurrence matrix has sum={res.sum():,} and shape={res.shape}')

    return res
