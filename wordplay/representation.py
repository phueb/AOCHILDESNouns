import numpy as np
from sklearn.preprocessing import normalize
from typing import Set


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