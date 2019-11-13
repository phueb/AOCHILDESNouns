from itertools import chain
import numpy as np
from typing import Set, List, Dict


def make_w2locations(tokens):
    print('Making w2locations...')
    result = {item: [] for item in set(tokens)}
    for loc, term in enumerate(tokens):
        result[term].append(loc)
    return result


def probes_reordered_loc():
    n_sum = 0
    num_ns = 0
    for n, term in enumerate(self.reordered_tokens):
        if term in self.probe_store.types:
            n_sum += n
            num_ns += 1
    result = n_sum / num_ns
    return result


def split_probes_by_loc(num_splits, is_reordered=False):
    if is_reordered:
        d = self.term_avg_reordered_loc_dict
    else:
        d = self.term_avg_unordered_loc_dict
    probe_loc_pairs = [(probe, loc) for probe, loc in d.items()
                       if probe in self.probe_store.types]
    sorted_probe_loc_pairs = sorted(probe_loc_pairs, key=lambda i: i[1])
    num_in_split = self.probe_store.num_probes // num_splits
    for split_id in range(num_splits):
        start = num_in_split * split_id
        probes, _ = zip(*sorted_probe_loc_pairs[start: start + num_in_split])
        yield probes


def make_locations_xy(w2locations: Dict[str, List[int]],
                      words: Set[str],
                      num_bins: int = 20,
                      ):
    """
    return x and y coordinates corresponding to histogram which reflects location distribution of words in corpus
    """

    word_locations_list = [w2locations[i] for i in words]
    locations_list = list(chain(*word_locations_list))
    hist, b = np.histogram(locations_list, bins=num_bins)
    result = (b[:-1], np.squeeze(hist))
    return result


def calc_loc_asymmetry(term, num_bins=200):  # num_bins=200 for terms
    (x, y) = self.make_locs_xy([term], num_bins=num_bins)
    y_fitted = self.fit_line(x, y)
    result = linregress(x / self.train_terms.num_tokens, y_fitted)[0]  # divide x to increase slope
    return result