from typing import Tuple, Optional, List, Dict
from spacy.tokens import Doc
from scipy.cluster.hierarchy import linkage, dendrogram
from sortedcontainers import SortedSet
import numpy as np
from collections import Counter

from abstractfirst.params import Params
from abstractfirst import configs


def split(it: List,
          split_size: int,
          ):
    for i in range(0, len(it), split_size):
        yield it[i:i + split_size]


def make_targets(params: Params,
                 age2doc: Dict[str, Doc],
                 verbose: bool = False,
                 ) -> Tuple[SortedSet, SortedSet]:

    # load experimental targets - but not all may occur in corpus
    p = configs.Dirs.targets / f'{params.targets_name}.txt'
    targets_exp_ = p.read_text().split('\n')
    assert targets_exp_[-1]

    # count targets
    w2f = Counter()
    for doc in age2doc.values():
        w2f.update([t.text for t in doc])

    # make control + experimental targets that match in frequency
    targets_ctl = SortedSet()
    targets_exp = SortedSet()
    vocab = [w for w, f in w2f.most_common()]
    for n, v in enumerate(vocab):
        if v in targets_exp_:
            t_ctl = vocab[n - 1]  # targets should be slightly more frequent so that params.max_sum works
            targets_exp.add(v)
            targets_ctl.add(t_ctl)
            if verbose:
                print(f'{v:<18} {w2f[v]:>6,} {t_ctl:<18} {w2f[t_ctl]:>6,}')

    assert len(targets_exp) == len(targets_ctl)

    return targets_exp, targets_ctl


def cluster(mat: np.ndarray,
            dg0: Optional[dict] = None,
            dg1: Optional[dict] = None,
            method: str = 'complete',
            metric: str = 'euclidean'):

    if dg0 is None:
        print('Clustering rows...')
        lnk0 = linkage(mat, method=method, metric=metric, optimal_ordering=True)
        dg0 = dendrogram(lnk0,
                         ax=None,
                         color_threshold=None,
                         no_labels=True,
                         no_plot=True)
    res = mat[dg0['leaves'], :]  # reorder rows

    if dg1 is None:
        print('Clustering cols...')
        lnk1 = linkage(mat.T, method=method, metric=metric, optimal_ordering=True)
        dg1 = dendrogram(lnk1,
                         ax=None,
                         color_threshold=None,
                         no_labels=True,
                         no_plot=True)
    res = res[:, dg1['leaves']]  # reorder cols

    return res, dg0, dg1
