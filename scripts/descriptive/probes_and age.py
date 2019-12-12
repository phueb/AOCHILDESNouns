import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from categoryeval.probestore import ProbeStore

from wordplay.utils import human_format
from wordplay.word_sets import excluded
from wordplay.binned import make_age_bin2data
from wordplay import config
from wordplay.pos import tag2pos

# ///////////////////////////////////////////////////////////////// parameters

CORPUS_NAME = 'childes-20191112'
PROBES_NAME = 'sem-concrete'
POS_LIST = []
AGE_STEP = 200
NORMALIZE = True
OTHER_ARE_NOUNS_ONLY = True

OTHER = 'other nouns' if OTHER_ARE_NOUNS_ONLY else 'other'

# ///////////////////////////////////////////////////////////////// combine docs by age

age_bin2tokens = make_age_bin2data(CORPUS_NAME, AGE_STEP)
age_bin2tags = make_age_bin2data(CORPUS_NAME, AGE_STEP, suffix='_tags')
if AGE_STEP == 100:
    del age_bin2tokens[0]
    del age_bin2tags[0]
print(f'Number of bins={len(age_bin2tokens)}')

# ///////////////////////////////////////////////////////////////// prepare POS words

w2id = {}
for tokens in age_bin2tokens.values():
    for w in set(tokens):
        w2id[w] = len(w2id)
probe_store = ProbeStore('childes-20180319', PROBES_NAME, w2id, excluded=excluded)

# ///////////////////////////////////////////////////////////////// count POS

# count
cat2y = {pos: [] for pos in probe_store.cats}
used_probes = set()
for cat in probe_store.cats:
    print(f'Counting number of {cat} words...')
    cat_probes = probe_store.cat2probes[cat]
    for tokens in age_bin2tokens.values():
        ones = [1 for w in tokens if w in cat_probes]
        num = len(ones)
        if NORMALIZE:
            cat2y[cat].append(num / len(tokens))
        else:
            cat2y[cat].append(num)
    used_probes.update(cat_probes)

# add remaining categories not included above
print(f'Counting number of {OTHER} words...')
cat2y[OTHER] = []
for tokens, tags in zip(age_bin2tokens.values(), age_bin2tags.values()):
    if OTHER_ARE_NOUNS_ONLY:
        ones = [1 for token, tag in zip(tokens, tags) if token not in used_probes and tag2pos.get(tag) == 'NOUN']
    else:
        ones = [1 for token in tokens if token not in used_probes]
    num = len(ones)
    if NORMALIZE:
        cat2y[OTHER].append(num / len(tags))
    else:
        cat2y[OTHER].append(num)

for cat, y in cat2y.items():
    print(cat)
    print(y)

# make x
x1 = [k for k, v in sorted(age_bin2tokens.items(), key=lambda i: i[0])][:-1]
x2 = [k + AGE_STEP for k, v in sorted(age_bin2tokens.items(), key=lambda i: i[0])][:-1]
x = [*sum(zip(x1, x2), ())]

# stack plot
fig, ax = plt.subplots(dpi=config.Fig.dpi, figsize=(8, 6))
if NORMALIZE:
    y_label = 'Token Density'
else:
    y_label = 'Token Frequency'
ax.set_ylabel(y_label, fontsize=config.Fig.ax_fontsize)
ax.set_xlabel('Age of Target Child (days)', fontsize=config.Fig.ax_fontsize)
if not NORMALIZE:
    ax.yaxis.set_major_formatter(FuncFormatter(human_format))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='both', top=False, right=False)
plt.grid(True, which='both', axis='y', alpha=0.2)
ax.set_xticks(x1)
if AGE_STEP < 200:
    ax.set_xticklabels([xi if n % 2 == 0 else '' for n, xi in enumerate(x1)])
else:
    ax.set_xticklabels(x1)
# plot
labels = [k for k, v in sorted(cat2y.items(), key=lambda i: i[0])]
ys = [np.repeat(v, 2)[:-2] for k, v in sorted(cat2y.items(), key=lambda i: i[0])]
ax.stackplot(x, *ys, labels=labels)
plt.legend(loc='center', fontsize=config.Fig.leg_fontsize, frameon=False, bbox_to_anchor=(1.2, 0.5))
plt.show()


