import random
from typing import Set, List, Optional, Tuple, Dict
from copy import deepcopy
from itertools import islice
from collections import deque
from scipy import sparse
from sortedcontainers import SortedSet


def sliding_window_iter(iterable, size):
    """..."""
    iterable = iter(iterable)
    window = deque(islice(iterable, size), maxlen=size)
    for item in iterable:
        yield tuple(window)
        window.append(item)
    if window:
        # needed because if iterable was already empty before the `for`,
        # then the window would be yielded twice.
        yield tuple(window)


def make_sparse_co_occurrence_mat(tokens: List[str],
                                  targets: SortedSet,
                                  left_only: bool = False,
                                  right_only: bool = False,
                                  separate_left_and_right: bool = True,
                                  shuffle_tokens: bool = False,
                                  stop_n: Optional[int] = None,  # stop collection if sum of matrix is stop_n
                                  ) -> sparse.coo_matrix:
    """
    targets in rows, 1-left and 1-right contexts in columns
    """

    if separate_left_and_right:
        if right_only or left_only:
            raise ValueError('Cannot use arg left_only or right_only with arg separate_left_and_right')
    else:
        assert not (right_only and left_only)

    left = not right_only
    right = not left_only

    print(f'Making co-occurrence matrix with left={left} and right={right} and separate={separate_left_and_right}...')

    # shuffle
    if shuffle_tokens:
        print('WARNING: Shuffling tokens')
        tokens = tokens.copy()
        random.shuffle(tokens)

    get_row_id = {}
    get_col_id = {}

    tokens_copy = deepcopy(tokens)[1:-2]  # careful: use same span as that used in for_loop

    # make sparse matrix (contexts/y-words in rows, targets/x-words in cols)
    data = []
    row_ids = []
    col_ids = []
    for n, cw in enumerate(tokens[1:-2]):

        if cw not in targets:
            continue

        # left word, center word, right word
        lw = tokens_copy[n - 1]
        cw = cw
        rw = tokens_copy[n + 1]

        # distinguish left and right words in columns by making them unique
        if separate_left_and_right:
            lw = deepcopy(lw) + 'l'
            rw = deepcopy(rw) + 'r'

        # collect left co-occurrence
        if left:
            row_ids.append(get_row_id.setdefault(cw, len(get_row_id)))
            col_ids.append(get_col_id.setdefault(lw, len(get_col_id)))
            data.append(1)  # it is okay to append 1s because final value is sum over 1s in same position in matrix

        # collect right co-occurrence
        if right:
            row_ids.append(get_row_id.setdefault(cw, len(get_row_id)))
            col_ids.append(get_col_id.setdefault(rw, len(get_col_id)))
            data.append(1)  # it is okay to append 1s because final value is sum over 1s in same position in matrix

        if stop_n is not None:
            if len(data) >= stop_n:
                print(f'Stopping co-occurrence collection early because sum>=stop_n={len(data)}')
                break

    # make sparse matrix once (updating it is expensive)
    res = sparse.coo_matrix((data, (row_ids, col_ids)))

    if res.shape[0] < len(targets):
        print(f'WARNING: Number of targets={len (targets)} but number of rows={res.shape[0]}.'
              f'This probably means not all targets occur in this corpus partition.')
    if res.shape[0] > len(targets):
        raise RuntimeError(f'Number of targets={len (targets)} but number of rows={res.shape[0]}.')

    print(f'Done. Co-occurrence matrix has sum={res.sum():,} and shape={res.shape}')

    return res
