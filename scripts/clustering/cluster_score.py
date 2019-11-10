"""
Research question:
1. Are semantic categories in part 1 of AO-CHILDES more easily clustered according to the gold structure?
"""

from sklearn.metrics.pairwise import cosine_similarity
from cytoolz import itertoolz
import matplotlib.pyplot as plt
import attr

from preppy.legacy import TrainPrep
from categoryeval.probestore import ProbeStore
from categoryeval.score import calc_score

from wordplay.params import PrepParams
from wordplay.docs import load_docs
from wordplay.svd import make_context_by_term_matrix
from wordplay.memory import set_memory_limit

# /////////////////////////////////////////////////////////////////

CORPUS_NAME = 'childes-20180319'
PROBES_NAME = 'sem-4096'  # cannot use sem-all because some are too infrequent to occur in each partition

REVERSE = False
NUM_PARTS = 2
SHUFFLE_DOCS = False

docs = load_docs(CORPUS_NAME,
                 num_test_take_from_mid=0,
                 num_test_take_random=0,
                 shuffle_docs=SHUFFLE_DOCS)

params = PrepParams(num_parts=NUM_PARTS, reverse=REVERSE)
prep = TrainPrep(docs, **attr.asdict(params))

probe_store = ProbeStore(CORPUS_NAME, PROBES_NAME, prep.store.w2id)

# ///////////////////////////////////////////////////////////////// parameters


NUM_EVALUATIONS = 2
WINDOW_SIZES = [1, 2, 3, 4, 5, 6, 7]
METRIC = 'ck'

if METRIC == 'ba':
    y_lims = [0.5, 1.0]
elif METRIC == 'ck':
    y_lims = [0.0, 0.2]
elif METRIC == 'f1':
    y_lims = [0.0, 0.2]
else:
    raise AttributeError('Invalid arg to "METRIC".')


# /////////////////////////////////////////////////////////////////


def plot_score_trajectories(part_id2y, part_id2x, title, fontsize=14):
    fig, ax = plt.subplots(figsize=(8, 5), dpi=None)
    plt.title(title, fontsize=fontsize)
    ax.set_xlabel('Number of Words in Partition', fontsize=fontsize)
    ax.set_ylabel(METRIC, fontsize=fontsize)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='both', top=False, right=False)
    ax.yaxis.grid(True)
    ax.set_ylim(y_lims)
    # plot
    for part_id, y in part_id2y.items():
        x = part_id2x[part_id]
        ax.plot(x, y, label='partition {}'.format(part_id + 1))
    #
    plt.legend(frameon=False, loc='lower right', fontsize=12)
    plt.tight_layout()
    plt.show()


set_memory_limit(prop=0.9)

token_parts = [prep.store.tokens[:prep.midpoint],
               prep.store.tokens[-prep.midpoint:]]

for w_size in WINDOW_SIZES:
    part_ids = range(2)
    part_id2scores = {part_id: [y_lims[0]] for part_id in part_ids}
    part_id2num_windows = {part_id: [0] for part_id in part_ids}
    for part_id in part_ids:
        tokens = token_parts[part_id]
        xi = 0
        num_tokens_in_chunk = len(tokens) // NUM_EVALUATIONS
        for tokens_chunk in itertoolz.partition_all(num_tokens_in_chunk, tokens):

            # compute representations
            tw_mat, xws, yws = make_context_by_term_matrix(tokens_chunk,
                                                           context_size=w_size,
                                                           probe_store=probe_store)
            try:
                probe_reps = tw_mat.toarray().T  # transpose because representations are expected in the rows
            except MemoryError:
                raise SystemExit('Reached memory limit')

            # calc score
            score = calc_score(cosine_similarity(probe_reps), probe_store.gold_sims, metric=METRIC)
            # collect
            xi += len(tokens_chunk)
            part_id2scores[part_id].append(score)
            part_id2num_windows[part_id].append(xi)
            print('part_id={} score={:.3f}'.format(part_id, score))
        print('------------------------------------------------------')

    # plot
    plot_score_trajectories(part_id2scores, part_id2num_windows,
                            title='Semantic category information in AO-CHILDES'
                                  '\ncaptured by term-window co-occurrence matrix\n'
                                  'with window-size={}'.format(w_size))