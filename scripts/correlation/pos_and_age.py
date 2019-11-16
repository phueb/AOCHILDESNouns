"""
Research questions:
1. Does the density of nouns, verb, adjectives, etc. vary with age (not partition) in AO-CHILDES?

Caveat:
Because age bins contain an unequal number of tokens, care must be taken this does not influence results
"""
import spacy
import matplotlib.pyplot as plt

from categoryeval.probestore import ProbeStore

from wordplay.binned import make_age_bin2tokens
from wordplay.binned import make_age_bin2tokens_with_min_size

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


make_age_bin2tokens_with_min_size(age_bin2word_tokens)  # TODO use this to combine smaller bins

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
    probe_store = ProbeStore('childes-20180319', PROBES_NAME, w2id, excluded=excluded)
    pos_words = probe_store.cat2probes[POS]
    print(len(pos_words))


    # TODO implement this script - it should do what pos_and_partition.py does, but instead operates over
    # TODO age bins rather than equal sized partitions

