import pandas as pd
import random
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


def get_term_id_windows(self, term, roll_left=False, num_samples=64):
    locations = random.sample(self.term_unordered_locs_dict[term], num_samples)
    if not roll_left:  # includes term in window
        result = [self.train_terms.token_ids[loc - self.params.window_size + 1: loc + 1]
                  for loc in locations if loc > self.params.window_size]
    else:
        result = [self.train_terms.token_ids[loc - self.params.window_size + 0: loc + 0]
                  for loc in locations if loc > self.params.window_size]
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
    ngrams = list(itertoolz.sliding_window(window_size, tokens))
    return ngrams