import pandas as pd
import numpy as np
from scipy.signal import lfilter
from cytoolz import itertoolz


def smooth(l, strength):
    b = [1.0 / strength] * strength
    a = 1
    result = lfilter(b, a, l)
    return result


def roll_mean(l, size):
    result = pd.DataFrame(l).rolling(size).mean().values.flatten()
    return result


def split(l, split_size):
    for i in range(0, len(l), split_size):
        yield l[i:i + split_size]


def fit_line(x, y, eval_x=None):
    poly = np.polyfit(x, y, 1)
    result = np.poly1d(poly)(eval_x or x)
    return result


def get_sliding_windows(window_size, tokens):

    if not isinstance(window_size, int):
        raise TypeError('This function was changed by PH in May 2019 because'
                        'previously used sklearn Countvectorizer uses stopwords'
                        ' and removes punctuation')
    res = list(itertoolz.sliding_window(window_size, tokens))
    return res


def reorder_parts_from_midpoint(parts: np.ndarray
                                ) -> np.ndarray:
    """
    deterministically reorder partitions such that the first partitions
    are guaranteed to be mid-partitions
    """
    # roll such that both matrices start at midpoints, and then get every other row
    a = np.roll(parts, len(parts) // 2, axis=0)[::-2]
    b = np.roll(parts, len(parts) // 2, axis=0)[::+2]
    # interleave rows of a and b
    res = np.hstack((a, b)).reshape(parts.shape)
    assert len(res) == len(parts)
    return res