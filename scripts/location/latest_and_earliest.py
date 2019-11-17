import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import attr
from matplotlib.ticker import FuncFormatter

from preppy.legacy import TrainPrep

from wordplay import config
from wordplay.word_sets import excluded
from wordplay.params import PrepParams
from wordplay.docs import load_docs
from wordplay.location import make_w2locations
from wordplay.location import make_locations_xy
from wordplay.utils import human_format

# /////////////////////////////////////////////////////////////////

CORPUS_NAME = 'childes-20180319'

REVERSE = False
NUM_PARTS = 1


docs = load_docs(CORPUS_NAME,
                 num_test_take_from_mid=0,
                 num_test_take_random=0,
                 )

params = PrepParams(num_parts=NUM_PARTS, reverse=REVERSE)
prep = TrainPrep(docs, **attr.asdict(params))


# /////////////////////////////////////////////////////////////////

NUM_WORDS_PLOT = 20


# location
w2locations = make_w2locations(prep.store.tokens)
w2avg_location = {w: np.mean(locations) for w, locations in w2locations.items()}

# sort words by location
location_sorted_words = sorted(prep.store.types, key=w2avg_location.get)
half = len(location_sorted_words) // 2

# fig
_, axs = plt.subplots(2, 1, sharex='all', dpi=163, figsize=(8, 6))
for ax, title, words in zip(axs,
                            ['Earliest occurring words', 'Latest occuring words'],
                            [location_sorted_words[:half], location_sorted_words[::-1][:half]]):
    if ax == axs[1]:
        ax.set_xlabel('Corpus Location', fontsize=12)
    ax.set_ylabel('Token Freq', fontsize=12)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='both', top=False, right=False)
    plt.setp(ax.get_yticklabels(), fontsize=12)
    plt.setp(ax.get_xticklabels(), fontsize=12)
    ax.set_title(title, fontsize=12, y=0.9)
    ax.xaxis.set_major_formatter(FuncFormatter(human_format))
    # plot
    colors = sns.color_palette("hls", NUM_WORDS_PLOT)
    for term, color in zip(words[:NUM_WORDS_PLOT], colors):
        (x, y) = make_locations_xy(w2locations, {term})
        ax.plot(x, y, label='{}'.format(term), c=color)
    ax.legend(loc='center',
              fontsize=8,
              frameon=False,
              bbox_to_anchor=(1.2, 0.5),
              ncol=2)
# show
plt.show()
