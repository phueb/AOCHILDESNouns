"""
Research questions:
1. Do contexts in partition 1 of AO-CHILDES contain category members more exclusively?
2. Are contexts more "pure" in the sense that they are ess contaminated by non-category words?

Note:
    This measure differs only slightly from coverage:
    Coverage computes KL divergence between distributions over words in a single category only,
    bu purity considers all words in the vocabulary



"""

import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import attr
import numpy as np
import pyprind
import seaborn as sns
from typing import Set, List, Dict, Tuple, Any

from preppy.legacy import TrainPrep
from categoryeval.probestore import ProbeStore

from wordplay import config
from wordplay.word_sets import excluded
from wordplay.params import PrepParams
from wordplay.docs import load_docs
from wordplay.measures import calc_kl_divergence
from wordplay.memory import set_memory_limit

# /////////////////////////////////////////////////////////////////

CORPUS_NAME = 'childes-20180319'
PROBES_NAME = 'sem-all'
NUM_TYPES = 4096  # a small vocabulary is needed to compute purity, otherwise vocabulary is too large

SHUFFLE_DOCS = False
NUM_MID_TEST_DOCS = 0

docs = load_docs(CORPUS_NAME,
                 num_test_take_from_mid=NUM_MID_TEST_DOCS,
                 num_test_take_random=0,
                 shuffle_docs=SHUFFLE_DOCS)

params = PrepParams(num_types=NUM_TYPES)
prep = TrainPrep(docs, **attr.asdict(params))

probe_store = ProbeStore(CORPUS_NAME, PROBES_NAME, prep.store.w2id, excluded=excluded)

# /////////////////////////////////////////////////////////////////

MIN_CONTEXT_FREQ = 10
CONTEXT_SIZES = [1, 2, 3]
SHOW_HISTOGRAM = False


def make_context2info(probes: Set[str],
                      tokens: List[str],
                      distance: int,
                      ) -> Dict[Tuple[str], Dict[str, Any]]:
    """
    compute KL divergence for each probe context type across entire corpus.
    keep track of information about each context type
    """
    print('Collecting information about probe context locations...')

    res = {}
    pbar = pyprind.ProgBar(len(tokens))
    for loc, token in enumerate(tokens[:-distance]):
        context = tuple(tokens[loc + d] for d in range(-distance, 0) if d != 0)
        if token in probes:
            try:
                res[context]['freq_by_probe'][token] += 1
                res[context]['total_freq'] += 1
                res[context]['term_freq'] += 0
                res[context]['probe_freq'] += 1
                res[context]['locations'].append(loc)
            except KeyError:
                res[context] = {'freq_by_probe': {probe: 0.0 for probe in probes},
                                'total_freq': 0,
                                'term_freq': 0,
                                'probe_freq': 0,
                                'locations': [],
                                }
        else:

            raise NotImplementedError('Purity is not yet implemented')  # TODO

        pbar.update()

    return res


def make_kld_location_tuples(d, probes):
    """
    generate tuples like [(kld, locations), (kld, locations, ...]
    """
    probes_expected_probabilities = np.array([1 / len(probes) for _ in probes])

    print('Calculating KL divergences...')
    for context in d.keys():
        context_freq = d[context]['total_freq']
        if context_freq > MIN_CONTEXT_FREQ:

            # kld
            probes_observed_probabilities = np.array(
                [d[context]['freq_by_probe'][probe] / d[context]['total_freq']
                 for probe in probes])
            kl = calc_kl_divergence(probes_expected_probabilities, probes_observed_probabilities)  # asymmetric

            yield kl, d[context]['locations']


set_memory_limit(prop=0.9)

headers = ['Category', 'partition', 'context-size', 'purity', 'n']
name2col = {name: [] for name in headers}
cat2context_size2p = {cat: {} for cat in probe_store.cats}
for cat in probe_store.cats:
    cat_probes = probe_store.cat2probes[cat]

    for context_size in CONTEXT_SIZES:

        # compute KL values for contexts associated with a single category
        try:
            context2info = make_context2info(cat_probes, prep.store.tokens, context_size)
        except MemoryError:
            raise SystemExit('Reached memory limit')

        # separate kl values by location
        y1 = []
        y2 = []
        for kld, locations in make_kld_location_tuples(context2info, cat_probes):
            num_locations_in_part1 = len(np.where(np.array(locations) < prep.midpoint)[0])
            num_locations_in_part2 = len(np.where(np.array(locations) > prep.midpoint)[0])
            y1 += [kld] * num_locations_in_part1
            y2 += [kld] * num_locations_in_part2
        y1 = np.array(y1)
        y2 = np.array(y2)

        # convert kld to purity # TODO
        raise NotImplementedError('how to convert?')

        # t test
        t, prob = ttest_ind(y1, y2, equal_var=True)
        cat2context_size2p[cat][context_size] = prob
        print('t={}'.format(t))
        print('p={:.6f}'.format(prob))
        print()

        # populate tabular data
        for name, di in zip(headers, (cat, 1, context_size, np.mean(y1), len(y1))):
            name2col[name].append(di)
        for name, di in zip(headers, (cat, 2, context_size, np.mean(y2), len(y2))):
            name2col[name].append(di)
