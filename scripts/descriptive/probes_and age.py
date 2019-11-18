import pygal
from pygal.style import DefaultStyle

from categoryeval.probestore import ProbeStore

from wordplay.word_sets import excluded
from wordplay.binned import make_age_bin2tokens

# ///////////////////////////////////////////////////////////////// parameters

CORPUS_NAME = 'childes-20191112'
PROBES_NAME = 'sem-concrete'
POS_LIST = []
OTHER = 'other'
AGE_STEP = 100
INTERPOLATE = 'hermite'
BAR = True

# ///////////////////////////////////////////////////////////////// combine docs by age

age_bin2tokens = make_age_bin2tokens(CORPUS_NAME, AGE_STEP)
print(f'Number of bins={len(age_bin2tokens)}')

# /////////////////////////////////////////////////////////////////

w2id = {}
for tokens in age_bin2tokens.values():
    for w in set(tokens):
        w2id[w] = len(w2id)
probe_store = ProbeStore('childes-20180319', PROBES_NAME, w2id, excluded=excluded)

# count
cat2y = {pos: [] for pos in probe_store.cats}
for cat in probe_store.cats:
    print(f'Counting number of {cat} words...')

    for tokens in age_bin2tokens.values():
        ones = [1 for w in tokens if w in probe_store.cat2probes[cat]]
        num = len(ones)
        cat2y[cat].append(num)


for cat, y in cat2y.items():
    print(cat)
    print(y)

print('Making chart...')

if BAR:
    Bar = pygal.StackedBar
else:
    Bar = pygal.StackedLine

style = DefaultStyle(plot_background='white',
                     background='white',
                     opacity='1.0')
chart = Bar(fill=True,
            show_dots=False,
            human_readable=True,
            x_title='Age (in days)',
            y_title='Token frequency',
            interpolate=INTERPOLATE,
            style=style)
for cat, y in cat2y.items():
    chart.add(cat, y)
chart.x_labels = [str(age_bin) for age_bin in age_bin2tokens.keys()]
chart.render_to_png('test.png')