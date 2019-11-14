"""
Research questions:
1. How well does noun-context selectivity correlate with semantic complexity?

A caveat:
The context-selectivity measure is extremely sensitive to the number of tokens.
THis means that comparing selectivity at age bins,
 care must be taken to sample an equal number of words at each bin

"""
import spacy
import matplotlib.pyplot as plt

from categoryeval.probestore import ProbeStore

from wordplay.binned import make_age_bin2tokens
from wordplay.representation import make_context_by_term_matrix
from wordplay.measures import calc_selectivity
from wordplay.sentences import get_sentences_from_tokens
from wordplay.utils import plot_best_fit_line
from wordplay.svo import subject_verb_object_triples

# ///////////////////////////////////////////////////////////////// parameters

CORPUS_NAME = 'childes-20191112'
PROBES_NAME = 'syn-4096'
AGE_STEP = 100
CONTEXT_SIZE = 3
NUM_TOKENS_PER_BIN = 50 * 1000  # 100K is good with AGE_STEP=100
POS = 'VERB'

# ///////////////////////////////////////////////////////////////// combine docs by age

age_bin2word_tokens = make_age_bin2tokens(CORPUS_NAME, AGE_STEP)
age_bin2tag_tokens = make_age_bin2tokens(CORPUS_NAME, AGE_STEP, suffix='_tags')

for word_tokens in age_bin2word_tokens.values():  # this is used to determine maximal NUM_TOKENS_PER_BIN
    print(f'{len(word_tokens):,}')

# /////////////////////////////////////////////////////////////////

nlp = spacy.load("en_core_web_sm", disable=['ner'])

x = []
y = []
age_bins = []
for age_bin in age_bin2tag_tokens.keys():

    word_tokens = age_bin2word_tokens[age_bin]
    tag_tokens = age_bin2tag_tokens[age_bin]

    assert len(word_tokens) == len(tag_tokens)
    assert word_tokens != tag_tokens

    # get same number of tokens at each bin
    if len(word_tokens) < NUM_TOKENS_PER_BIN:
        print(f'WARNING: Number of tokens at age_bin={age_bin} < NUM_TOKENS_PER_BIN')
        continue
    else:
        word_tokens = word_tokens[:NUM_TOKENS_PER_BIN]

    # pos_words
    w2id = {w: n for n, w in enumerate(set(word_tokens))}
    probe_store = ProbeStore('childes-20180319', PROBES_NAME, w2id)
    pos_words = probe_store.cat2probes[POS]
    print(len(pos_words))


    # TODO implement this script - it should do what pos_and_partition.py does, but instead operates over
    # TODO age bins rather than equal sized partitions

