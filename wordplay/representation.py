import random
from collections import Counter
from typing import Set, List, Optional, Tuple, Dict

import numpy as np
import pyprind
from categoryeval.probestore import ProbeStore
from scipy import sparse
from sklearn.preprocessing import normalize
from sortedcontainers import SortedSet, SortedDict

from wordplay.utils import get_sliding_windows


def make_bow_probe_representations(windows_mat: np.ndarray,
                                   vocab: Set[str],
                                   probes: Set[str],
                                   norm: Optional[str] = None,
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


def make_probe_reps_median_split(probe2contexts: Dict[str, Tuple[str]],
                                 context_types: SortedSet,
                                 split_id: int,
                                 ) -> np.ndarray:
    """
    make probe representations based on first or second median split of each probe's contexts.
    representation can be BOW or preserve word-order, depending on how contexts were collected.
    """
    num_context_types = len(context_types)
    probes = SortedSet(probe2contexts.keys())
    num_probes = len(probe2contexts)
    context2col_id = {c: n for n, c in enumerate(context_types)}

    probe_reps = np.zeros((num_probes, num_context_types))
    progress_bar = pyprind.ProgBar(num_probes, stream=2, title='Making representations form contexts')
    for row_id, p in enumerate(probes):
        probe_contexts = probe2contexts[p]
        num_probe_contexts = len(probe_contexts)
        num_in_split = num_probe_contexts // 2

        # get either first half or second half of contexts
        if split_id == 0:
            probe_contexts_split = probe_contexts[:num_in_split]
        elif split_id == 1:
            probe_contexts_split = probe_contexts[-num_in_split:]
        else:
            raise AttributeError('Invalid arg to split_id.')

        # make probe representation
        c2f = Counter(probe_contexts_split)
        for c, f in c2f.items():
            col_id = context2col_id[c]
            probe_reps[row_id, col_id] = f

        progress_bar.update()

    # check each representation has information
    num_zero_rows = np.sum(~probe_reps.any(axis=1))
    assert num_zero_rows == 0

    return probe_reps


def get_probe_contexts(probes: SortedSet,
                       tokens: List[str],
                       context_size: int,
                       preserve_order: bool,
                       ) -> Tuple[Dict[str, Tuple[str]], SortedSet]:
    # get all probe contexts
    probe2contexts = SortedDict({p: [] for p in probes})
    contexts_in_order = get_sliding_windows(context_size, tokens)
    context_types = SortedSet()
    for n, context in enumerate(contexts_in_order[:-context_size]):
        next_context = contexts_in_order[n + 1]
        target = next_context[-1]
        if target in probes:

            if preserve_order:
                probe2contexts[target].append(context)
                context_types.add(context)
            else:
                single_word_contexts = [(w,) for w in context]
                probe2contexts[target].extend(single_word_contexts)
                context_types.update(single_word_contexts)

    return probe2contexts, context_types