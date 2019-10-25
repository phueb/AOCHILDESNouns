import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from cytoolz import itertoolz

from childeshub.hub import Hub

"""

Compare the quality of verb-contexts between partition 1 and 2

1) find locations of all verbs in corpus
2) obtain their contexts (single or multi word spans) by collecting word-spans to left of those locations
3) iterate over corpus and collect all fillers for each context, sorting each filler by whether it is a verb or not
4) sort contexts by the number of verb-fillers
5) for each class of contexts (class is determined by number of verb-fillers), plot :
    a. scatter: x= frequency of verbs in a set of contexts, y = avg frequency of non-verb fillers in set of contexts
    b. best fit line: the smaller the slope the better the quality of the verb-contexts
    
"""


CORPUS_NAME = 'childes-20180319'

MIN_verb_FREQ = 100
HUB_MODE = 'sem'
CONTEXT_SIZE = 7

FONTSIZE = 16
FIG_SIZE = (6, 6)

MAX_CONTEXT_CLASS = 50000  # too small -> program does not work, too large -> waste memory
MIN_SUM = 0     # only affects figure and best-fit line
MAX_SUM = 1000  # only affects figure and best-fit line

assert MAX_SUM <= MAX_CONTEXT_CLASS


# filtered_verbs
hub = Hub(mode=HUB_MODE, part_order='inc_age', corpus_name=CORPUS_NAME)
filtered_verbs = {verb for verb in hub.verbs
                  if hub.train_terms.term_freq_dict[verb] > MIN_verb_FREQ}  # set for fast access
num_filtered_verbs = len(filtered_verbs)
print('Using {} verbs for analysis'.format(num_filtered_verbs))

# get verb_locations
verb_locations = []
for verb in filtered_verbs:
    verb_locations += hub.term_reordered_locs_dict[verb]


def make_context2is_filler_verb2freq(start_loc, end_loc):

    verb_locs_in_partition = [loc for loc in verb_locations
                              if start_loc < loc < end_loc]

    # get contexts (can be multi-word spans)
    verb_context_start_locations = np.array(verb_locs_in_partition) - CONTEXT_SIZE
    verb_contexts = {tuple(hub.train_terms.tokens[start_loc: start_loc + CONTEXT_SIZE])
                     for start_loc in verb_context_start_locations}

    # collect
    res = {context: {True: 0, False: 0} for context in verb_contexts}
    for window in itertoolz.sliding_window(CONTEXT_SIZE + 1, hub.train_terms.tokens[start_loc:end_loc]):

        context = window[:-1]
        filler = window[-1]

        if context not in verb_contexts:  # skip any contexts that are never filled by any verbs
            continue

        is_filler_verb = filler in filtered_verbs
        res[context][is_filler_verb] += 1

    return res


#
context2is_verb2freq1 = make_context2is_filler_verb2freq(start_loc=0,
                                                         end_loc=hub.midpoint_loc)
context2is_verb2freq2 = make_context2is_filler_verb2freq(start_loc=hub.midpoint_loc,
                                                         end_loc=hub.train_terms.num_tokens)

# fig
_, ax = plt.subplots(figsize=FIG_SIZE)
ax.set_title('Comparing the quality of verb contexts\ncontext-size={}'.format(CONTEXT_SIZE), fontsize=FONTSIZE)
ax.set_ylabel('Avg. number of non-verbs in context', fontsize=FONTSIZE)
ax.set_xlabel('Number of verbs in context', fontsize=FONTSIZE)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='both', top=False, right=False)
ax.set_xlim([MIN_SUM, MAX_SUM])
ax.set_ylim([MIN_SUM, MAX_SUM])  # make symmetrical
# plot
colors = iter(sns.color_palette("hls", 2)[::-1])
part_names = iter(['partition 1', 'partition 2'])

for d in [context2is_verb2freq1, context2is_verb2freq2]:

    # put corpora in d into classes
    # (classes are determined by verb-filler frequency: how often a context is filled by verbs)
    context_class2is_verb2stats = {cc: {True: {'sum': 0, 'n': 0}, False: {'sum': 0, 'n': 0}}
                                   for cc in range(MAX_CONTEXT_CLASS)}
    for context, is_verb2freq in d.items():

        context_class = is_verb2freq[True]  # number of verbs that fill the context

        if context_class >= MAX_CONTEXT_CLASS:  # extreme cases are not informative
            continue

        context_class2is_verb2stats[context_class][bool(1)]['sum'] += is_verb2freq[True]
        context_class2is_verb2stats[context_class][bool(1)]['n'] += 1
        context_class2is_verb2stats[context_class][bool(0)]['sum'] += is_verb2freq[False]
        context_class2is_verb2stats[context_class][bool(0)]['n'] += 1

    x_yes_verbs = [cc
                   for cc, is_verb2stats in sorted(context_class2is_verb2stats.items())
                   if is_verb2stats[True]['n'] > 0 and MIN_SUM <  is_verb2stats[True]['sum'] < MAX_SUM]
    x_non_verbs = [cc
                   for cc, is_verb2stats in sorted(context_class2is_verb2stats.items())
                   if is_verb2stats[False]['n'] > 0 and MIN_SUM <  is_verb2stats[False]['sum'] < MAX_SUM]

    y_yes_verbs = [is_verb2stats[True]['sum'] / is_verb2stats[True]['n']
                   for cc, is_verb2stats in sorted(context_class2is_verb2stats.items())
                   if is_verb2stats[True]['n'] > 0 and MIN_SUM <  is_verb2stats[True]['sum'] < MAX_SUM]
    y_non_verbs = [is_verb2stats[False]['sum'] / is_verb2stats[False]['n']
                   for cc, is_verb2stats in sorted(context_class2is_verb2stats.items())
                   if is_verb2stats[False]['n'] > 0 and MIN_SUM < is_verb2stats[False]['sum'] < MAX_SUM]

    # this should be a straight line if everything is correct
    ax.plot(x_yes_verbs, y_yes_verbs, color='grey', linestyle='-', zorder=1)

    # for appearance only
    ax.plot([0, MAX_SUM], [0, MAX_SUM], color='grey', linestyle='-', zorder=1)

    color = next(colors)
    part_name = next(part_names)
    ax.scatter(x_non_verbs, y_non_verbs, color=color, label=part_name)
    ax.plot([0, MAX_SUM], hub.fit_line(x_non_verbs, y_non_verbs, eval_x=[0, MAX_SUM]), color=color, lw=3)
    # the smaller the slope of the curve, the better
    # because number of non-nuns do not increase as fast as number of verbs in a context

plt.legend(frameon=True, loc='lower right', fontsize=FONTSIZE)
plt.tight_layout()
plt.show()