import pygal
from pygal.style import DefaultStyle


from wordplay.pos import pos2tags
from wordplay.binned import get_binned

# ///////////////////////////////////////////////////////////////// parameters

CORPUS_NAME = 'childes-20191112'
POS_LIST = []
OTHER = 'other'
AGE_STEP = 100
INTERPOLATE = 'hermite'
BAR = True

# ///////////////////////////////////////////////////////////////// combine docs by age

age_bins, tags_by_binned_age = get_binned(CORPUS_NAME, AGE_STEP, suffix='_tags')
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
for pos, y in pos2y.items():
    chart.add(pos, y)
chart.x_labels = age_bins
chart.render_to_png('test.png')