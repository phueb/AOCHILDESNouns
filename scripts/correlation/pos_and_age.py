"""
Research questions:
1. Does the density of nouns, verb, adjectives, etc. vary with age (not partition) in AO-CHILDES?

Caveat:
Because age bins contain an unequal number of tokens, care must be taken this does not influence results
"""

from scipy import stats
import numpy as np
from tabulate import tabulate

from wordplay.binned import make_age_bin2data
from wordplay.binned import make_age_bin2data_with_min_size
from wordplay.pos import tag2pos

# ///////////////////////////////////////////////////////////////// parameters

CORPUS_NAME = 'childes-20191112'
AGE_STEP = 100
NUM_TOKENS_PER_BIN = 50 * 1000  # 100K is good with AGE_STEP=100

# ///////////////////////////////////////////////////////////////// combine docs by age

age_bin2tags_ = make_age_bin2data(CORPUS_NAME, AGE_STEP, suffix='_tags')

for word_tokens in age_bin2tags_.values():  # this is used to determine maximal NUM_TOKENS_PER_BIN
    print(f'{len(word_tokens):,}')

# combine small bins
age_bin2tags = make_age_bin2data_with_min_size(age_bin2tags_, NUM_TOKENS_PER_BIN)

num_bins = len(age_bin2tags)

# /////////////////////////////////////////////////////////////////

POS_LIST = set([pos for pos in tag2pos.values() if pos.isupper()])

# collect counts
pos2counts = {pos: [] for pos in POS_LIST}
for age_bin, tags in age_bin2tags.items():

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
a = np.arange(num_bins)
for pos in POS_LIST:
    b = np.array(pos2counts[pos]) / NUM_TOKENS_PER_BIN
    rho, p = stats.spearmanr(a, b)
    print(f'{pos:<12} rho={rho:+.2f} p={p:.4f}')

    # collect for pretty-printed table
    data.append((pos, rho, p))

# print pretty table
print()
print(tabulate(data,
               headers=["Part-of-Speech", "Spearman's Rho", "p-value"]))
print(f'Number of age bins={num_bins}')

# latex
print(tabulate(sorted(data, key=lambda d: d[0]),
               headers=["Part-of-Speech", "Spearman's Rho", "p-value"],
               tablefmt='latex',
               floatfmt=".4f"))

