import numpy as np
import pygal
from pygal.style import DefaultStyle
from itertools import groupby

from wordplay.pos import pos2tags
from wordplay import config


# ///////////////////////////////////////////////////////////////// parameters

CORPUS_NAME = 'childes-20191112'
POS_LIST = []
OTHER = 'other'
AGE_STEP = 100
INTERPOLATE = 'hermite'

# ///////////////////////////////////////////////////////////////// combine docs by age


# TODO put binning logic into a library module

# get ages - this information is not available for childes-20180319
ages_path = config.Dirs.corpora / f'{CORPUS_NAME}_ages.txt'
ages_text = ages_path.read_text(encoding='utf-8')
ages = np.array(ages_text.split(), dtype=np.float)

# get tags
tags_path = config.Dirs.corpora / f'{CORPUS_NAME}_tags.txt'
tags_text = tags_path.read_text(encoding='utf-8')
tags_by_doc = [doc.split() for doc in tags_text.split('\n')[:-1]]

# convert ages to age bins
ages_binned = ages - np.mod(ages, AGE_STEP)
ages_binned = ages_binned.astype(np.int)
data = zip(ages_binned, tags_by_doc)

tags_by_binned_age = []
x_labels = []
for binned_age, data_group in groupby(data, lambda d: d[0]):
    docs = [d[1] for d in data_group]
    tags = list(np.concatenate(docs))
    print(f'Found {len(docs)} transcripts for age-bin={binned_age}')

    tags_by_binned_age.append(tags)
    x_labels.append(str(binned_age))

print(f'Number of bins={len(tags_by_binned_age)}')

# /////////////////////////////////////////////////////////////////

pos_list = POS_LIST or list(sorted(pos2tags.keys()))
# count tags
pos2y = {pos: [] for pos in pos_list}
used_tags = set()
for pos in pos_list:
    print(f'Counting number of {pos} words...')

    requested_tags = set(pos2tags[pos])

    for tags in tags_by_binned_age:
        ones = [1 for tag in tags if tag in requested_tags]
        num = len(ones)
        pos2y[pos].append(num)

    used_tags.update(requested_tags)

# add remaining categories not included above
print(f'Counting number of {OTHER} words...')
pos2y[OTHER] = []
for tags in tags_by_binned_age:
    ones = [1 for tag in tags if tag not in used_tags]
    num = len(ones)
    pos2y[OTHER].append(num)


for pos, y in pos2y.items():
    print(pos)
    print(y)

print('Making chart...')
style = DefaultStyle(plot_background='white',
                     background='white',
                     opacity='1.0')
chart = pygal.StackedLine(fill=True,
                          show_dots=False,
                          human_readable=True,
                          x_title='Age (in days)',
                          y_title='Token frequency',
                          interpolate=INTERPOLATE,
                          style=style)
for pos, y in pos2y.items():
    chart.add(pos, y)
chart.x_labels = x_labels
chart.render_to_png('test.png')  # TODO what's taking so long?