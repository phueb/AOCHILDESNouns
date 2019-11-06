"""
Research questions:
1. Are nouns used more consistently in partition 1 vs. 2?
2. How about other Part-of-speech categories?

Consistency is defined as:
A set of words X is used more consistently to the extent their contexts contain more X and less non-X

To compute consistency:

1) find locations of all nouns in corpus
2) obtain their contexts (single or multi word spans) by collecting word-spans to left of those locations
3) iterate over corpus and collect all fillers for each context, sorting each filler by whether it is a noun or not
4) sort contexts by the number of noun-fillers
5) for each class of contexts (class is determined by number of noun-fillers), plot :
    a. scatter: x= frequency of nouns in a set of contexts, y = avg frequency of non-noun fillers in set of contexts
    b. best fit line: the smaller the slope the higher the consistency of the noun-contexts

"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from cytoolz import itertoolz
import attr

from preppy.legacy import TrainPrep

from wordplay.params import PrepParams
from wordplay.docs import load_docs
from wordplay.pos import load_pos_words
from wordplay.location import make_w2location
from wordplay.utils import fit_line

# /////////////////////////////////////////////////////////////////

CORPUS_NAME = 'childes-20180319'

SHUFFLE_DOCS = False

docs = load_docs(CORPUS_NAME,
                 num_test_take_from_mid=0,
                 num_test_take_random=0,
                 start_at_midpoint=False,
                 shuffle_docs=SHUFFLE_DOCS)

params = PrepParams()
prep = TrainPrep(docs, **attr.asdict(params))

# /////////////////////////////////////////////////////////////////

POS = 'nouns+plurals'
CONTEXT_SIZE = 4

MAX_CONTEXT_CLASS = 50000  # too small -> program does not work, too large -> waste memory
MIN_SUM = 0     # only affects figure and best-fit line
MAX_SUM = 1000  # only affects figure and best-fit line

assert MAX_SUM <= MAX_CONTEXT_CLASS


pos_words = load_pos_words(f'{CORPUS_NAME}-{POS}')  # TODO try all semantic probe words

# get locations
locations = []
w2location = make_w2location(prep.store.tokens)
for w in pos_words:
    locations += w2location[w]  # TODO not all nouns are found


def make_context2is_filler_pos2freq(start_loc, end_loc):

    pos_locations_in_partition = [loc for loc in locations
                                  if start_loc < loc < end_loc]

    # get contexts (can be multi-word spans)
    pos_context_start_locations = np.array(pos_locations_in_partition) - CONTEXT_SIZE
    contexts = {tuple(prep.store.tokens[start_loc: start_loc + CONTEXT_SIZE])
                for start_loc in pos_context_start_locations}

    # collect
    res = {c: {True: 0, False: 0} for c in contexts}
    window_size = CONTEXT_SIZE + 1
    for window in itertoolz.sliding_window(window_size, prep.store.tokens[start_loc:end_loc]):

        context = window[:-1]
        filler = window[-1]

        if context not in contexts:  # skip any contexts that are never filled by any pos-word
            continue

        is_filler_pos = filler in pos_words
        res[context][is_filler_pos] += 1

    return res


#
context2is_pos2freq1 = make_context2is_filler_pos2freq(start_loc=0,
                                                       end_loc=prep.midpoint)
context2is_pos2freq2 = make_context2is_filler_pos2freq(start_loc=prep.midpoint,
                                                       end_loc=prep.store.num_tokens)

# fig
_, ax = plt.subplots(figsize=(6, 6))
ax.set_title(f'{POS} consistency\ncontext-size={CONTEXT_SIZE}', fontsize=12)
ax.set_ylabel(f'Avg. number of non-{POS} in context', fontsize=12)
ax.set_xlabel(f'Number of {POS} in context', fontsize=12)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='both', top=False, right=False)
ax.set_xlim([MIN_SUM, MAX_SUM])
ax.set_ylim([MIN_SUM, MAX_SUM])  # make symmetrical
# plot
colors = iter(sns.color_palette("hls", 2)[::-1])
part_names = iter(['partition 1', 'partition 2'])

for d in [context2is_pos2freq1, context2is_pos2freq2]:

    # put corpora in d into classes
    # (classes are determined by pos-filler frequency: how often a context is filled by pos-word)
    context_class2is_pos2stats = {cc: {True: {'sum': 0, 'n': 0}, False: {'sum': 0, 'n': 0}}
                                  for cc in range(MAX_CONTEXT_CLASS)}
    for context, is_pos2freq in d.items():

        context_class = is_pos2freq[True]  # number of pos-words that fill the context

        if context_class >= MAX_CONTEXT_CLASS:  # extreme cases are not informative
            continue

        context_class2is_pos2stats[context_class][bool(1)]['sum'] += is_pos2freq[True]
        context_class2is_pos2stats[context_class][bool(1)]['n'] += 1
        context_class2is_pos2stats[context_class][bool(0)]['sum'] += is_pos2freq[False]
        context_class2is_pos2stats[context_class][bool(0)]['n'] += 1

    x_yes_pos = [cc
                 for cc, is_pos2stats in sorted(context_class2is_pos2stats.items())
                 if is_pos2stats[True]['n'] > 0 and MIN_SUM <  is_pos2stats[True]['sum'] < MAX_SUM]
    x_non_pos = [cc
                 for cc, is_pos2stats in sorted(context_class2is_pos2stats.items())
                 if is_pos2stats[False]['n'] > 0 and MIN_SUM <  is_pos2stats[False]['sum'] < MAX_SUM]

    y_yes_pos = [is_pos2stats[True]['sum'] / is_pos2stats[True]['n']
                 for cc, is_pos2stats in sorted(context_class2is_pos2stats.items())
                 if is_pos2stats[True]['n'] > 0 and MIN_SUM <  is_pos2stats[True]['sum'] < MAX_SUM]
    y_non_pos = [is_pos2stats[False]['sum'] / is_pos2stats[False]['n']
                 for cc, is_pos2stats in sorted(context_class2is_pos2stats.items())
                 if is_pos2stats[False]['n'] > 0 and MIN_SUM < is_pos2stats[False]['sum'] < MAX_SUM]

    # this should be a straight line if everything is correct
    ax.plot(x_yes_pos, y_yes_pos, color='grey', linestyle='-', zorder=1)

    # for appearance only
    ax.plot([0, MAX_SUM], [0, MAX_SUM], color='grey', linestyle='-', zorder=1)

    color = next(colors)
    part_name = next(part_names)
    ax.scatter(x_non_pos, y_non_pos, color=color, label=part_name)
    ax.plot([0, MAX_SUM], fit_line(x_non_pos, y_non_pos, eval_x=[0, MAX_SUM]), color=color, lw=3)
    # the smaller the slope of the curve, the better
    # because number of non-nuns do not increase as fast as number of pos-words in a context

plt.legend(frameon=True, loc='lower right', fontsize=12)
plt.tight_layout()
plt.show()