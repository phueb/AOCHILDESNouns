import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from wordplay.utils import human_format
from wordplay.pos import pos2tags
from wordplay.binned import make_age_bin2data
from wordplay import config

# ///////////////////////////////////////////////////////////////// parameters

CORPUS_NAME = 'childes-20191112'
POS_LIST = []
OTHER = 'other'
AGE_STEP = 200
NORMALIZE = False

# ///////////////////////////////////////////////////////////////// combine docs by age

age_bin2tags = make_age_bin2data(CORPUS_NAME, AGE_STEP, suffix='_tags')
if AGE_STEP == 100:
    del age_bin2tags[0]
print(f'Number of bins={len(age_bin2tags)}')

# ///////////////////////////////////////////////////////////////// prepare POS words

pos2tags['NOUN'].update(pos2tags['P-NOUN'])
del pos2tags['P-NOUN']

pos_list = POS_LIST or list(sorted(pos2tags.keys()))

# ///////////////////////////////////////////////////////////////// count POS

# count tags
pos2y = {pos: [] for pos in pos_list}
used_tags = set()
for pos in pos_list:
    print(f'Counting number of {pos} words...')
    requested_tags = set(pos2tags[pos])
    for tags in age_bin2tags.values():
        ones = [1 for tag in tags if tag in requested_tags]
        num = len(ones)
        if NORMALIZE:
            pos2y[pos].append(num / len(tags))
        else:
            pos2y[pos].append(num)
    used_tags.update(requested_tags)

# add remaining categories not included above
print(f'Counting number of {OTHER} words...')
pos2y[OTHER] = []
for tags in age_bin2tags.values():
    ones = [1 for tag in tags if tag not in used_tags]
    num = len(ones)
    if NORMALIZE:
        pos2y[OTHER].append(num / len(tags))
    else:
        pos2y[OTHER].append(num)


for pos, y in pos2y.items():
    print(pos)
    print(y)

# make x
x1 = [k for k, v in sorted(age_bin2tags.items(), key=lambda i: i[0])][:-1]
x2 = [k + AGE_STEP for k, v in sorted(age_bin2tags.items(), key=lambda i: i[0])][:-1]
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
labels = [k for k, v in sorted(pos2y.items(), key=lambda i: i[0])]
ys = [np.repeat(v, 2)[:-2] for k, v in sorted(pos2y.items(), key=lambda i: i[0])]
ax.stackplot(x, *ys, labels=labels)
# plt.legend(loc='center', fontsize=config.Fig.leg_fontsize, frameon=False, bbox_to_anchor=(1.2, 0.5))
plt.show()


