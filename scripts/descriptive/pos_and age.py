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
AGE_STEP = 100
INTERPOLATE = 'hermite'
BAR = True

# ///////////////////////////////////////////////////////////////// combine docs by age

age_bin2tags = make_age_bin2data(CORPUS_NAME, AGE_STEP, suffix='_tags')
print(f'Number of bins={len(age_bin2tags)}')

# /////////////////////////////////////////////////////////////////

pos_list = POS_LIST or list(sorted(pos2tags.keys()))
# count tags
pos2y = {pos: [] for pos in pos_list}
used_tags = set()
for pos in pos_list:
    print(f'Counting number of {pos} words...')

    requested_tags = set(pos2tags[pos])

    for tags in age_bin2tags.values():
        ones = [1 for tag in tags if tag in requested_tags]
        num = len(ones)
        pos2y[pos].append(num)

    used_tags.update(requested_tags)

# add remaining categories not included above
print(f'Counting number of {OTHER} words...')
pos2y[OTHER] = []
for tags in age_bin2tags.values():
    ones = [1 for tag in tags if tag not in used_tags]
    num = len(ones)
    pos2y[OTHER].append(num)


for pos, y in pos2y.items():
    print(pos)
    print(y)

# delete minor contributions
del pos2y['other']
del pos2y['particle']
del pos2y['CCONJ']
del pos2y['P-NOUN']

# stack plot
fig, ax = plt.subplots(dpi=config.Fig.dpi, figsize=(8, 6))
ax.set_ylabel('Token Frequency', fontsize=config.Fig.ax_fontsize)
ax.set_xlabel('Age of Target Child (days)', fontsize=config.Fig.ax_fontsize)
ax.yaxis.set_major_formatter(FuncFormatter(human_format))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='both', top=False, right=False)
plt.grid(True, which='both', axis='y', alpha=0.2)
# plot
x = [k for k, v in sorted(age_bin2tags.items(), key=lambda i: i[0], reverse=True)]
labels = [k for k, v in sorted(pos2y.items(), key=lambda i: i[0], reverse=True)]
ys = [np.array(v) for k, v in sorted(pos2y.items(), key=lambda i: i[0], reverse=True)]
ax.stackplot(x, *ys, labels=labels)
plt.legend(loc='upper left')
plt.show()


