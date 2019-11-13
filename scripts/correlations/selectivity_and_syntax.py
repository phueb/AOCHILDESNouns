"""
Research questions:
1. How well does noun-context selectivity correlate with syntactic complexity?

A caveat:
The context-selectivity measure is extremely sensitive to the number of tokens.
THis means that comparing selectivity at age bins,
 care must be taken to sample an equal number of words at each bin

"""
import numpy as np
import matplotlib.pyplot as plt

from categoryeval.probestore import ProbeStore

from wordplay.binned import get_binned
from wordplay.representation import make_context_by_term_matrix
from wordplay.measures import calc_selectivity
from wordplay.sentences import get_sentences_from_tokens
from wordplay.utils import plot_best_fit_line

# ///////////////////////////////////////////////////////////////// parameters

CORPUS_NAME = 'childes-20191112'
PROBES_NAME = 'syn-4096'
AGE_STEP = 100
CONTEXT_SIZE = 3
NUM_TOKENS_PER_BIN = 200 * 1000  # 100K is good with AGE_STEP=100
POS = 'NOUN'

# ///////////////////////////////////////////////////////////////// combine docs by age

age_bins, tokens_by_binned_age = get_binned(CORPUS_NAME, AGE_STEP)
_, tags_by_binned_age = get_binned(CORPUS_NAME, AGE_STEP, suffix='_tags')

for word_tokens in tokens_by_binned_age:  # this is used to determine maximal NUM_TOKENS_PER_BIN
    print(f'{len(word_tokens):,}')

# /////////////////////////////////////////////////////////////////

x = []
y = []
for age_bin, word_tokens, tag_tokens in zip(age_bins, tokens_by_binned_age, tags_by_binned_age):

    assert len(word_tokens) == len(tag_tokens)
    assert word_tokens != tag_tokens

    # nouns
    w2id = {w: n for n, w in enumerate(set(word_tokens))}
    probe_store = ProbeStore('childes-20180319', PROBES_NAME, w2id)
    nouns = probe_store.cat2probes[POS]
    print(len(nouns))

    # get same number of tokens at each bin
    if not len(word_tokens) > NUM_TOKENS_PER_BIN:
        print(f'WARNING: Number of tokens at age_bin={age_bin} < NUM_TOKENS_PER_BIN')
        continue
    else:
        word_tokens = word_tokens[:NUM_TOKENS_PER_BIN]

    # compute num unique tag-sequences as measure of syn complexity
    sentences = get_sentences_from_tokens(tag_tokens, punctuation={'.'})
    unique_sentences = np.unique(sentences)
    comp = len(unique_sentences) / len(sentences)
    print(f'Found {len(sentences):>12,} total sentences in part')
    print(f'Found {len(unique_sentences):>12,} unique sentences in part')

    # co-occurrence matrix
    tw_mat_observed, xws_observed, _ = make_context_by_term_matrix(word_tokens,
                                                                   context_size=CONTEXT_SIZE,
                                                                   shuffle_tokens=False)
    tw_mat_chance, xws_chance, _ = make_context_by_term_matrix(word_tokens,
                                                               context_size=CONTEXT_SIZE,
                                                               shuffle_tokens=True)

    # calc selectivity of noun contexts
    cttr_chance, cttr_observed, sel = calc_selectivity(tw_mat_chance,
                                                       tw_mat_observed,
                                                       xws_chance,
                                                       xws_observed,
                                                       nouns)
    print(f'age_bin={age_bin} selectivity={sel}')
    print()

    # collect
    x.append(comp)
    y.append(sel)

# figure
fig, ax = plt.subplots(1, figsize=(7, 7), dpi=192)
ax.set_xlabel('Syntactic Complexity', fontsize=14)
ax.set_ylabel(f'{POS}-Context Selectivity\n(context-size={CONTEXT_SIZE})', fontsize=14)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='both', top=False, right=False)
# plot
ax.scatter(x, y, color='black')
plot_best_fit_line(ax, x, y, fontsize=12, x_pos=0.75, y_pos=0.75)
plt.tight_layout()
plt.show()