from collections import Counter
from typing import Dict, Tuple

from sortedcontainers import SortedSet
from spacy.tokens.doc import Doc

from aochildesnouns import configs
from aochildesnouns.params import Params


def make_targets(params: Params,
                 age2doc: Dict[str, Doc],
                 verbose: bool = True,
                 ) -> Tuple[SortedSet, SortedSet]:

    # load experimental targets - but not all may occur in corpus
    p = configs.Dirs.targets / f'{params.targets_name}.txt'
    targets_exp_ = p.read_text().split('\n')
    assert targets_exp_[-1]

    # load all nouns
    p = configs.Dirs.targets / 'nouns-sing_and_plural.txt'
    nouns = p.read_text().split('\n')
    assert nouns[-1]

    # count targets
    w2f = Counter()
    for doc in age2doc.values():
        w2f.update([t.text for t in doc])

    # make control + experimental targets that match in frequency
    targets_ctl = SortedSet()
    targets_exp = SortedSet()
    vocab = [w for w, f in w2f.most_common()]  # in order of most to least common (lower id -> higher frequency)
    for n, v in enumerate(vocab):
        if v in targets_exp_:

            # find control target that is not in set of experimental targets, and is not already in control target set
            offset = -1
            while True:
                target_ctl = vocab[n - offset]  # targets should be slightly more frequent so that params.max_sum works
                offset += 1
                if target_ctl in targets_exp_:
                    continue
                if target_ctl in targets_ctl:
                    continue
                if target_ctl in {'.', '?', '!'}:
                    continue
                if target_ctl in nouns:
                    continue
                break

            targets_exp.add(v)
            targets_ctl.add(target_ctl)
            if verbose:
                print(f'{v:<18} {w2f[v]:>6,} {target_ctl:<18} {w2f[target_ctl]:>6,}')

    assert len(targets_exp) == len(targets_ctl)

    return targets_exp, targets_ctl