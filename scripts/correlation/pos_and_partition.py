"""
Research questions:
1. Does the density of nouns, verb, adjectives, etc. vary with partition in AO-CHILDES?

Caveat:
This is similar to asking whether density of nouns, verbs, adjectives, etc vary with age,
but it is NOT the same, because a partition may include speech to children at different ages
"""

from scipy import stats
import numpy as np
import attr
from tabulate import tabulate

from preppy.legacy import TrainPrep

from wordplay.params import PrepParams
from wordplay.docs import load_docs
from wordplay.utils import split
from wordplay.pos import tag2pos

# /////////////////////////////////////////////////////////////////

CORPUS_NAME = 'childes-20180319_tags'
PROBES_NAME = 'syn-4096'

NUM_PARTS = 64  # spearman correlation requires more than just 2

docs = load_docs(CORPUS_NAME)

params = PrepParams(num_parts=NUM_PARTS)
prep = TrainPrep(docs, **attr.asdict(params))

# /////////////////////////////////////////////////////////////////

POS_LIST = set([pos for pos in tag2pos.values() if pos.isupper()])

# collect counts
pos2counts = {pos: [] for pos in POS_LIST}
for tags in split(prep.store.tokens, prep.num_tokens_in_part):
    pos_tags = [tag2pos.get(t, None) for t in tags]
    print()
    print(f'{"excluded":<16} num={pos_tags.count(None):>9,}')
    for pos in POS_LIST:
        y = pos_tags.count(pos)
        print(f'{pos:<16} num={y:>9,}')
        # collect
        pos2counts[pos].append(y)


# calculate Spearman's correlation
data = []
a = np.arange(NUM_PARTS)
for pos in POS_LIST:
    b = np.array(pos2counts[pos]) / prep.num_tokens_in_part
    rho, p = stats.spearmanr(a, b)
    print(f'{pos:<12} rho={rho:+.2f} p={p:.4f}')

    # collect for pretty-printed table
    data.append((pos, rho, p))

# print pretty table
print()
print(tabulate(data,
               headers=["Part-of-Speech", "Spearman's Rho", "p-value"]))
print(f'Number of corpus partitions={NUM_PARTS}')

# latex
print(tabulate(sorted(data, key=lambda d: d[0]),
               headers=["Part-of-Speech", "Spearman's Rho", "p-value"],
               tablefmt='latex',
               floatfmt=".4f"))

