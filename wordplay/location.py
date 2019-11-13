from itertools import chain
import numpy as np
from typing import Set, List, Dict


def make_w2locations(tokens):
    print('Making w2locations...')
    result = {item: [] for item in set(tokens)}
    for loc, term in enumerate(tokens):
        result[term].append(loc)
    return result


def split_probes_by_loc(w2avg_location, probe_store, num_splits):
    probe_loc_pairs = [(w, loc) for w, loc in w2avg_location.items()
                       if w in probe_store.types]
    sorted_probe_loc_pairs = sorted(probe_loc_pairs, key=lambda i: i[1])
    num_in_split = probe_store.num_probes // num_splits
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
