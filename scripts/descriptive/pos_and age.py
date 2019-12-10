import pygal
from pygal.style import DefaultStyle


from wordplay.pos import pos2tags
from wordplay.binned import make_age_bin2tokens

# ///////////////////////////////////////////////////////////////// parameters

CORPUS_NAME = 'childes-20191112'
POS_LIST = []
OTHER = 'other'
AGE_STEP = 100
INTERPOLATE = 'hermite'
BAR = True

# ///////////////////////////////////////////////////////////////// combine docs by age

age_bin2tags = make_age_bin2tokens(CORPUS_NAME, AGE_STEP, suffix='_tags')
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

print('Making chart...')


# TODO make matplotlib stacked figure - matplotlib is easier to work with
x = [1, 2, 3, 4, 5]
y1 = [1, 1, 2, 3, 5]
y2 = [0, 4, 2, 6, 8]
y3 = [1, 3, 5, 7, 9]

y = np.vstack([y1, y2, y3])

labels = ["Fibonacci ", "Evens", "Odds"]

fig, ax = plt.subplots()
ax.stackplot(x, y1, y2, y3, labels=labels)

raise SystemExit

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
chart.x_labels = [str(age_bin) for age_bin in age_bin2tags.keys()]
chart.render_to_png('test.png')