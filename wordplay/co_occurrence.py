import random
from typing import Set, List, Optional, Tuple, Dict

from categoryeval.probestore import ProbeStore
from scipy import sparse

from sortedcontainers import SortedSet

from wordplay.util import get_sliding_windows


def make_context_by_term_matrix(tokens: List[str],
                                start: Optional[int] = None,
                                end: Optional[int] = None,
                                shuffle_tokens: bool = False,
                                context_size: Optional[int] = 7,
                                probe_store: Optional[ProbeStore] = None):
    """
    terms are in cols, contexts are in rows.
    y_words are contexts (possibly multi-word), x_words are targets (always single-word).
    a context precedes a target.
    """

    if (start is None and end is not None) or (end is not None and start is None):
        raise ValueError('Specifying only start or end is not allowed.')

    print('Making context-term matrix')

    # tokens
    if start is not None and end is not None:
        print(f'from tokens between {start:,} & {end:,}')
        tokens = tokens[start:end]

    # shuffle
    if shuffle_tokens:
        print('WARNING: Shuffling tokens')
        tokens = tokens.copy()
        random.shuffle(tokens)

    # x_words
    if probe_store is not None:
        x_words = probe_store.types
    else:
        x_words = SortedSet(tokens)
    num_xws = len(x_words)
    xw2col_id = {t: n for n, t in enumerate(x_words)}

    # contexts
    contexts_in_order = get_sliding_windows(context_size, tokens)
    y_words = SortedSet(contexts_in_order)
    num_y_words = len(y_words)
    yw2row_id = {c: n for n, c in enumerate(y_words)}

    # make sparse matrix (contexts/y-words in rows, targets/x-words in cols)
    data = []
    row_ids = []
    cold_ids = []
    for n, context in enumerate(contexts_in_order[:-context_size]):
        # row_id + col_id
        row_id = yw2row_id[context]
        next_context = contexts_in_order[n + 1]
        next_word = next_context[-1]  # -1 is correct because windows slide by 1 word
        try:
            col_id = xw2col_id[next_word]
        except KeyError:  # when probe_store is passed, only probes are n xw2col_id
            continue
        # collect
        row_ids.append(row_id)
        cold_ids.append(col_id)
        data.append(1)  # it is okay to append 1s because final value is sum over 1s in same position in matrix

    # make sparse matrix once (updating it is expensive)
    res = sparse.coo_matrix((data, (row_ids, cold_ids)))

    print(f'Co-occurrence matrix has sum={res.sum():,} and shape={res.shape}')
    expected_shape = (num_y_words, num_xws)
    if res.shape != expected_shape and not probe_store:
        raise SystemExit(f'Result does not match expected shape={expected_shape}')

    return res, x_words, y_words
