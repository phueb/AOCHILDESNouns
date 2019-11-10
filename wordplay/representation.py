import random

import numpy as np
from categoryeval.probestore import ProbeStore
from scipy import sparse
from sklearn.preprocessing import normalize
from typing import Set, List, Optional

from wordplay.utils import get_sliding_windows


def make_bow_probe_representations(windows_mat: np.ndarray,
                                   vocab: Set[str],
                                   probes: Set[str],
                                   norm: str = 'l1',
                                   direction: int = -1,
                                   ) -> np.ndarray:
    """
    return a matrix containing bag-of-words representations of probes in the rows
    """

    num_types = len(vocab)
    id2w = {i: w for i, w in enumerate(vocab)}

    probe2rep = {p: np.zeros(num_types) for p in probes}
    for window in windows_mat:
        first_word = id2w[window[0]]
        last_word = id2w[window[-1]]
        if direction == -1:  # context is defined to be words left of probe
            if last_word in probes:
                for word_id in window[:-1]:
                    probe2rep[last_word][word_id] += 1
        elif direction == 1:
            if first_word in probes:
                for word_id in window[0:]:
                    probe2rep[first_word][word_id] += 1
        else:
            raise AttributeError('Invalid arg to "DIRECTION".')
    # representations
    res = np.asarray([probe2rep[p] for p in probes])
    if norm is not None:
        res = normalize(res, axis=1, norm=norm, copy=False)
    return res


def make_bow_token_representations(windows_mat: np.ndarray,
                                   vocab: Set[str],
                                   norm: str = 'l1',
                                   ):
    num_types = len(vocab)
    res = np.zeros((num_types, num_types))
    for window in windows_mat:
        obs_word_id = window[-1]
        for var_word_id in window[:-1]:
            res[obs_word_id, var_word_id] += 1  # TODO which order?
    # norm
    if norm is not None:
        res = normalize(res, axis=1, norm=norm, copy=False)
    return res


def make_context_by_term_matrix(tokens: List[str],
                                start: Optional[int] = None,
                                end: Optional[int] = None,
                                shuffle_tokens: bool = False,
                                context_size: Optional[int] = 7,
                                probe_store: Optional[ProbeStore] = None):
    """
    terms are in cols, contexts are in rows.
    y_words are contexts, x_words are terms.
    y_words always occur after x_words in the input.
    this format matches that used in TreeTransitions
    """

    print('Making context-term matrix')
    if start and end:
        print(f'from tokens between {start:,} & {end:,}')

    # tokens
    if start is not None and end is not None:
        tokens = tokens[start:end]

    # shuffle
    if shuffle_tokens:
        print('WARNING: Shuffling tokens')
        random.shuffle(tokens)

    # x_words
    if probe_store is not None:
        print('WARNING: Using only probes as x-words')
        x_words = probe_store.types
    else:
        x_words = sorted(set(tokens))
    num_xws = len(x_words)
    xw2col_id = {t: n for n, t in enumerate(x_words)}

    # contexts
    contexts_in_order = get_sliding_windows(context_size, tokens)
    unique_contexts = sorted(set(contexts_in_order))
    num_unique_contexts = len(unique_contexts)
    context2row_id = {w: n for n, w in enumerate(unique_contexts)}

    # make sparse matrix (contexts/y-words in rows, targets/x-words in cols)
    data = []
    row_ids = []
    cold_ids = []
    for n, context in enumerate(contexts_in_order[:-context_size]):
        # row_id + col_id
        row_id = context2row_id[context]
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
    expected_shape = (num_unique_contexts, num_xws)
    if res.shape != expected_shape and not probe_store:
        raise SystemExit(f'Result does not match expected shape={expected_shape}')

    y_words = unique_contexts
    return res, x_words, y_words