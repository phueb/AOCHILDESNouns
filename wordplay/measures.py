import numpy as np
import pandas as pd
from collections import Counter
from scipy.sparse import coo_matrix

from typing import List, Set, Tuple


def calc_selectivity(tw_mat_chance: coo_matrix,
                     tw_mat_observed: coo_matrix,
                     xws_chance: List[str],
                     xws_observed: List[str],
                     words: Set[str]
                     ) -> Tuple[float, float, float]:
    # cttr_chance
    col_ids = [n for n, xw in enumerate(xws_chance) if xw in words]
    cols = tw_mat_chance.tocsc()[:, col_ids].toarray()
    context_distribution = np.sum(cols, axis=1, keepdims=False)
    num_context_types = np.count_nonzero(context_distribution)
    num_context_tokens = np.sum(context_distribution)
    cttr_chance = num_context_types / num_context_tokens

    # cttr_observed
    col_ids = [n for n, xw in enumerate(xws_observed) if xw in words]
    cols = tw_mat_observed.tocsc()[:, col_ids].toarray()
    context_distribution = np.sum(cols, axis=1, keepdims=False)
    num_context_types = np.count_nonzero(context_distribution)
    num_context_tokens = np.sum(context_distribution)
    cttr_observed = num_context_types / num_context_tokens

    print(f'cttr_chance={cttr_chance:>6.2f} cttr_observed={cttr_observed:>6.2f}')

    # compute ratio such that the higher the better (the more selective)
    sel = cttr_chance / cttr_observed
    return cttr_chance, cttr_observed, sel


def calc_utterance_lengths(tokens: List[str],
                           rolling_avg: bool = False,
                           rolling_std: bool = False,
                           window_size: int = 1000,
                           ) -> np.ndarray:

    assert not(rolling_avg and rolling_std)

    # make sent_lengths
    last_period = 0
    sent_lengths = []
    for n, item in enumerate(tokens):
        if item in ['.', '!', '?']:
            sent_length = n - last_period - 1
            sent_lengths.append(sent_length)
            last_period = n

    # rolling window
    if rolling_avg:
        df = pd.Series(sent_lengths)
        print('Making sentence length rolling average...')
        return df.rolling(window_size).std().values
    elif rolling_std:
        df = pd.Series(sent_lengths)
        print('Making sentence length rolling std...')
        return df.rolling(window_size).mean().values
    else:
        return np.array(sent_lengths)


def calc_entropy(words):
    num_labels = len(words)
    probabilities = np.asarray([count / num_labels for count in Counter(words).values()])
    result = - probabilities.dot(np.log2(probabilities))
    return result


def safe_divide(numerator, denominator):
    if denominator == 0 or denominator == 0.0:
        index = 0
    else:
        index = numerator/denominator
    return index


def ttr(text):
    ntokens = len(text)
    ntypes = len(set(text))

    return safe_divide(ntypes, ntokens)


def mtld(words, min=10):
    """
    Measure of lexical textual diversity (MTLD), described in Jarvis & McCarthy.
    'average number of words until text is saturated, or stable'
    implementation obtained from
    https://github.com/kristopherkyle/lexical_diversity/blob/master/lexical_diversity/lex_div.py
    """
    def mtlder(text):
        factor = 0
        factor_lengths = 0
        start = 0
        for x in range(len(text)):
            factor_text = text[start:x+1]
            if x+1 == len(text):
                factor += safe_divide((1 - ttr(factor_text)),(1 - .72))
                factor_lengths += len(factor_text)
            else:
                if ttr(factor_text) < .720 and len(factor_text) >= min:
                    factor += 1
                    factor_lengths += len(factor_text)
                    start = x+1
                else:
                    continue

        mtld = safe_divide(factor_lengths,factor)
        return mtld
    input_reversed = list(reversed(words))
    res = safe_divide((mtlder(words) + mtlder(input_reversed)), 2)
    return res


