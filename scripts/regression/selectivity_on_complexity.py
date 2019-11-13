"""
Research questions:
1. How well does noun-context selectivity correlate with semantic complexity?

A caveat:
The context-selectivity measure is extremely sensitive to the number of tokens.
THis means that comparing selectivity at age bins,
 care must be taken to sample an equal number of words at each bin

"""
import matplotlib.pyplot as plt
import spacy
import numpy as np
import attr
import pandas as pd

from preppy.legacy import TrainPrep
from categoryeval.probestore import ProbeStore

from wordplay.regression import regress
from wordplay.docs import load_docs
from wordplay.params import PrepParams
from wordplay.binned import make_age_bin2tokens
from wordplay.representation import make_context_by_term_matrix
from wordplay.measures import calc_selectivity
from wordplay.measures import calc_utterance_lengths
from wordplay.sentences import get_sentences_from_tokens
from wordplay.svo import subject_verb_object_triples

# /////////////////////////////////////////////////////////////////

CORPUS_NAME = 'childes-20180319'
PROBES_NAME = 'syn-4096'

REVERSE = False
NUM_PARTS = 1
SHUFFLE_DOCS = False

docs = load_docs(CORPUS_NAME,
                 num_test_take_from_mid=0,
                 num_test_take_random=0,
                 shuffle_docs=SHUFFLE_DOCS)

params = PrepParams(num_parts=NUM_PARTS, reverse=REVERSE, num_types=4096)
prep = TrainPrep(docs, **attr.asdict(params))

probe_store = ProbeStore('childes-20180319', PROBES_NAME, prep.store.w2id)

# ///////////////////////////////////////////////////////////////// parameters

CORPUS_NAME = 'childes-20191112'
PROBES_NAME = 'syn-4096'
AGE_STEP = 50
CONTEXT_SIZE = 2
NUM_TOKENS_PER_BIN = 100 * 1000  # 100K is good with AGE_STEP=100
POS = 'NOUN'

MIN_NUM_POS_WORDS = 100

# ///////////////////////////////////////////////////////////////// combine docs by age

age_bin2word_tokens = make_age_bin2tokens(CORPUS_NAME, AGE_STEP)
age_bin2tag_tokens = make_age_bin2tokens(CORPUS_NAME, AGE_STEP, suffix='_tags')

for age_bin, word_tokens in age_bin2word_tokens.items():  # this is used to determine maximal NUM_TOKENS_PER_BIN
    print(f'Number of words at age_bin={age_bin:>6} is {len(word_tokens):>6,}')

# /////////////////////////////////////////////////////////////////

nlp = spacy.load("en_core_web_sm", disable=['ner'])

mlu = []
syn_complexity = []
sem_complexity = []
selectivity = []
included_age_bins = []
for age_bin in age_bin2tag_tokens.keys():

    word_tokens = age_bin2word_tokens[age_bin]
    tag_tokens = age_bin2tag_tokens[age_bin]

    assert len(word_tokens) == len(tag_tokens)
    assert word_tokens != tag_tokens

    # get same number of tokens at each bin
    if len(word_tokens) < NUM_TOKENS_PER_BIN:
        print(f'WARNING: Number of tokens ={len(word_tokens):>12,} < NUM_TOKENS_PER_BIN = {NUM_TOKENS_PER_BIN:>12,}')
        continue
    else:
        word_tokens = word_tokens[:NUM_TOKENS_PER_BIN]

    # check
    pos_words = probe_store.cat2probes[POS]
    num_pos_words_at_bin = len([w for w in pos_words if w in word_tokens])
    if num_pos_words_at_bin < MIN_NUM_POS_WORDS:
        print(f'WARNING: Number of {POS} words = {num_pos_words_at_bin} < {MIN_NUM_POS_WORDS}')
        continue

    # /////////////////////////////////// calc MLU

    lengths = calc_utterance_lengths(word_tokens)
    mlu_i = np.mean(lengths)

    # /////////////////////////////////// calc syntactic complexity

    sentences = get_sentences_from_tokens(tag_tokens, punctuation={'.', '!', '?'})
    unique_sentences = np.unique(sentences)
    syn_complexity_i = len(unique_sentences) / len(sentences)

    # /////////////////////////////////// calc semantic complexity

    # compute num SVO triples as measure of semantic complexity
    sentences = get_sentences_from_tokens(word_tokens, punctuation={'.', '!', '?'})
    texts = [' '.join(s) for s in sentences]
    unique_triples = set()
    for doc in nlp.pipe(texts):
        for t in subject_verb_object_triples(doc):  # only returns triples, not partial triples
            unique_triples.add(t)
    num_unique_triples_in_part = len(unique_triples)
    sem_complexity_i = num_unique_triples_in_part

    # /////////////////////////////////// calc selectivity

    # co-occurrence matrix
    tw_mat_observed, xws_observed, _ = make_context_by_term_matrix(word_tokens,
                                                                   context_size=CONTEXT_SIZE,
                                                                   shuffle_tokens=False)
    tw_mat_chance, xws_chance, _ = make_context_by_term_matrix(word_tokens,
                                                               context_size=CONTEXT_SIZE,
                                                               shuffle_tokens=True)

    # calc selectivity of noun contexts
    cttr_chance, cttr_observed, selectivity_i = calc_selectivity(tw_mat_chance,
                                                                 tw_mat_observed,
                                                                 xws_chance,
                                                                 xws_observed,
                                                                 pos_words)

    print(f'age_bin={age_bin}\n'
          f'selectivity={selectivity_i}\n'
          f'mlu={mlu_i}\n'
          f'syn-complexity={syn_complexity_i}\n'
          f'sem-complexity={sem_complexity_i}\n')
    print()

    # collect
    selectivity.append(selectivity_i)
    mlu.append(mlu_i)
    syn_complexity.append(syn_complexity_i)
    sem_complexity.append(sem_complexity_i)
    included_age_bins.append(age_bin)

# TODO nonlinear reg? log?

# regression
x = pd.DataFrame(data={'mlu': mlu, 'sem-comp': sem_complexity})
y = pd.Series(selectivity)
y.name = f'{POS}-selectivity'
summary = regress(x, y)  # reduces same results as sklearn with intercept + normalization
print(summary)

# regression
x = pd.DataFrame(data={'mlu': mlu, 'syn-comp': syn_complexity})
y = pd.Series(selectivity)
y.name = f'{POS}-selectivity'
summary = regress(x, y)  # reduces same results as sklearn with intercept + normalization
print(summary)


# correlation matrix
x_all = pd.DataFrame(data={'mlu': mlu, 'syn-comp': syn_complexity, 'sem-comp': sem_complexity})
correlations = x_all.corr()
print(correlations.round(3))

# TODO to include more bins - combine bins which are excluded (those age bins at the very start and end

# scatter
xy = pd.concat((x_all, y), axis=1)
ax1 = xy.plot(kind='scatter', x='sem-comp', y=f'{POS}-selectivity')  # nonlinear effect
plt.show()
ax2 = xy.plot(kind='scatter', x='syn-comp', y=f'{POS}-selectivity')  # nonlinear effect
plt.show()
ax3 = xy.plot(kind='scatter', x='mlu', y=f'{POS}-selectivity')  # nonlinear effect
plt.show()

print('age bins included:')
for age_bin in included_age_bins:
    print(age_bin)