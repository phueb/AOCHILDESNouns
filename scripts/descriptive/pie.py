import pygal
from pygal.style import DefaultStyle


from wordplay.pos import pos2tags
from wordplay import config

# ///////////////////////////////////////////////////////////////// parameters

CORPUS_NAME = 'childes-20191112'
POS_LIST = []
OTHER = 'other'
AGE_STEP = 1

# ///////////////////////////////////////////////////////////////// tags


tags_path = config.Dirs.corpora / f'{CORPUS_NAME}_tags.txt'
tags_text = tags_path.read_text(encoding='utf-8')
tags = tags_text.split()

# /////////////////////////////////////////////////////////////////

pos_list = POS_LIST or list(sorted(pos2tags.keys()))
# count tags
pos2y = {pos: [] for pos in pos_list}
used_tags = set()
for pos in pos_list:
    print(f'Counting number of {pos} words...')

    requested_tags = set(pos2tags[pos])

    ones = [1 for tag in tags if tag in requested_tags]
    num = len(ones)
    pos2y[pos].append(num)

    used_tags.update(requested_tags)

# add remaining categories not included above
print(f'Counting number of {OTHER} words...')
pos2y[OTHER] = []
ones = [1 for tag in tags if tag not in used_tags]
num = len(ones)
pos2y[OTHER].append(num)

# figure
style = DefaultStyle(plot_background='white',
                     background='white',
                     opacity='1.0')
chart = pygal.Pie(inner_radius=0.5,
                  human_readable=True,
                  style=style)
for pos, y in pos2y.items():
    chart.add(pos, y)
chart.render_to_png('pie-test.png')