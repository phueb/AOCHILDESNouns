"""
make data frame with age_bin in rows, and all measures of quality on columns.
use this to do path analysis in R: does semantic or syntactic complexity mediate relation between age and
measures of distributional quality?

"""
import spacy
import numpy as np
import pandas as pd
import pyprind
from copy import deepcopy

from categoryeval.probestore import ProbeStore

from wordplay.word_sets import excluded
from wordplay.representation import make_context_by_term_matrix
from wordplay.measures import calc_selectivity
from wordplay.sentences import split_into_sentences
from wordplay.svo import subject_verb_object_triples
from wordplay.measures import calc_kl_divergence
from wordplay.binned import make_age_bin2data
from wordplay.binned import make_age_bin2data_with_min_size

# ///////////////////////////////////////////////////////////////// parameters

CORPUS_NAME = 'childes-20191112'
PROBES_NAME = 'syn-nva'
AGE_STEP = 100
NUM_TOKENS_PER_BIN = 50 * 1000  # 100K is good with AGE_STEP=100

CONTEXT_SIZE = 3
POS = 'NOUN'
ADD_SEM_PROBES = True if POS == 'NOUN' else False  # set to True when POS = 'NOUN'
USE_ONLY_SHARED_PROBES = True  # if True, probes must occur in each partition

# ///////////////////////////////////////////////////////////////// combine docs by age

age_bin2tags_ = make_age_bin2data(CORPUS_NAME, AGE_STEP, suffix='_tags')
age_bin2tokens_ = make_age_bin2data(CORPUS_NAME, AGE_STEP)

for word_tokens in age_bin2tags_.values():  # this is used to determine maximal NUM_TOKENS_PER_BIN
    print(f'{len(word_tokens):,}')

# combine small bins
age_bin2tags = make_age_bin2data_with_min_size(age_bin2tags_, NUM_TOKENS_PER_BIN)
age_bin2tokens = make_age_bin2data_with_min_size(age_bin2tokens_, NUM_TOKENS_PER_BIN)

num_bins = len(age_bin2tags)

# ///////////////////////////////////////////////////////////////// get test words

w2id = {}
num_tokens = 0
all_tokens = []
for tokens in age_bin2tokens.values():
    num_tokens += len(tokens)
    all_tokens += tokens
    for w in set(tokens):
        w2id[w] = len(w2id)
probe_store = ProbeStore('childes-20180319', PROBES_NAME, w2id, excluded=excluded)
pos_words = probe_store.cat2probes[POS].copy()

if ADD_SEM_PROBES:
    added_probes = ProbeStore('childes-20180319', 'sem-all', w2id, excluded=excluded).types.copy()
    pos_words.update(added_probes)

# get a subset of pos_words which occur in ALL parts of corpus
for tokens in age_bin2tokens.values():
    types_in_part = set(tokens)
    if USE_ONLY_SHARED_PROBES:
        pos_words.intersection_update(types_in_part)
print(f'Using {len(pos_words)} {POS} words')

nlp = spacy.load("en_core_web_sm", disable=['ner'])

# //////////////////////////////////////////////////////////////// preliminary computations

# preliminaries for computing coverage
info = {'freq_by_probe': {probe: 0.0 for probe in pos_words}, 'total_freq': 0, 'locations': []}
context2kld_info = {}
pbar = pyprind.ProgBar(num_tokens)
for loc, token in enumerate(all_tokens[:-CONTEXT_SIZE]):
    context = tuple(all_tokens[loc + dist] for dist in range(-CONTEXT_SIZE, 0) if dist != 0)
    if token in pos_words:
        context2kld_info.setdefault(context, deepcopy(info))['freq_by_probe'][token] += 1
        context2kld_info.setdefault(context, deepcopy(info))['total_freq'] += 1
        context2kld_info.setdefault(context, deepcopy(info))['locations'].append(loc)
    pbar.update()

# preliminaries for computing prominence
info = {'in-category': [],  'locations': []}
context2prominence_info = {}
pbar = pyprind.ProgBar(num_tokens)
for loc, token in enumerate(all_tokens[:-CONTEXT_SIZE]):
    context = tuple(all_tokens[loc + dist] for dist in range(-CONTEXT_SIZE, 0) if dist != 0)
    if token in pos_words:
        context2prominence_info.setdefault(context, deepcopy(info))['in-category'].append(1)
        context2prominence_info.setdefault(context, deepcopy(info))['locations'].append(loc)
    else:
        context2prominence_info.setdefault(context, deepcopy(info))['in-category'].append(0)
        context2prominence_info.setdefault(context, deepcopy(info))['locations'].append(loc)
    pbar.update()

# //////////////////////////////////////////////////////////////// compute measures for each bin

age = []
syn_complexity = []
sem_complexity = []
coverage = []
selectivity = []
prominence = []
start = 0  # location in corpus
for (age_bin, word_tokens), (_, tag_tokens) in zip(age_bin2tokens.items(),
                                                   age_bin2tags.items()):

    end = start + len(word_tokens)  # location in corpus

    assert len(word_tokens) == len(tag_tokens)
    assert word_tokens != tag_tokens
    assert (end - start) == len(word_tokens)

    # check
    num_pos_words_at_bin = len([w for w in pos_words if w in word_tokens])
    print(f'Using {num_pos_words_at_bin} probes to compute selectivity')

    # /////////////////////////////////// calc syntactic complexity

    tag_sentences = split_into_sentences(tag_tokens, punctuation={'.', '!', '?'})
    unique_sentences = np.unique(tag_sentences)
    syn_complexity_i = len(unique_sentences) / len(tag_sentences)

    # /////////////////////////////////// calc semantic complexity

    # compute num SVO triples as measure of semantic complexity
    word_sentences = split_into_sentences(word_tokens, punctuation={'.', '!', '?'})
    texts = [' '.join(s) for s in word_sentences]
    triples = []
    for doc in nlp.pipe(texts):
        for t in subject_verb_object_triples(doc):  # only returns triples, not partial triples
            triples.append(t)
    num_unique_triples_in_part = len(set(triples))
    num_total_triples_in_part = len(triples)
    sem_complexity_i = num_unique_triples_in_part / num_total_triples_in_part

    # /////////////////////////////////// calc coverage

    # return a context kld for each time a context occurs between start and end
    context_klds = []
    e = np.array([1 / len(pos_words) for _ in pos_words])
    for context in context2kld_info.keys():
        locations_array = np.array(context2kld_info[context]['locations'])
        num_locations_in_partition = np.sum(np.logical_and(start < locations_array, locations_array < end)).item()
        if num_locations_in_partition == 0:
            continue
        o = np.array(
            [context2kld_info[context]['freq_by_probe'][p] / context2kld_info[context]['total_freq'] for p in pos_words])
        y = calc_kl_divergence(e, o)  # asymmetric: expected probabilities, observed probabilities
        for _ in range(num_locations_in_partition):  # get the same kld value each time the context occurs
            context_klds.append(y)
    coverage_i = 1.0 / np.mean(context_klds)

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

    # /////////////////////////////////// calc prominence

    # return a prominence score for each time a context occurs between start and end
    context_prominences = []
    for context in context2prominence_info.keys():
        # a context must occur at least once with the category
        # note: this returns a lot of contexts, because lots of generic noun contexts co-occur with semantic probes
        in_category_list = context2prominence_info[context]['in-category']
        num_in_category = np.sum(in_category_list)
        if num_in_category == 0:
            continue
        y = num_in_category / len(in_category_list)
        locations_array = np.array(context2prominence_info[context]['locations'])
        num_locations_in_partition = np.sum(np.logical_and(start < locations_array, locations_array < end)).item()
        for _ in range(num_locations_in_partition):  # get the same prominence value each time the context occurs
            context_prominences.append(y)
    prominence_i = np.mean(context_prominences)

    # /////////////////////////////////// collect data

    print(
        f'age_bin={age_bin}\n'
        f'coverage={coverage_i}\n'
        f'selectivity={selectivity_i}\n'
        f'prominence={prominence_i}\n'
        f'syn-complexity={syn_complexity_i}\n'
        f'sem-complexity={sem_complexity_i}\n',
    )
    print()

    # collect
    age.append(age_bin)
    coverage.append(coverage_i)
    selectivity.append(selectivity_i)
    prominence.append(prominence_i)

    syn_complexity.append(syn_complexity_i)
    sem_complexity.append(sem_complexity_i)

    # update start location
    start = end

# data frame
df = pd.DataFrame(data={
    'age': age,
    'sem': sem_complexity,
    'syn': syn_complexity,
    'coverage': coverage,
    'selectivity': selectivity,
    'prominence': prominence,
})
normalized_df = (df-df.mean()) / df.std()
normalized_df['age'] = df['age']
normalized_df.to_csv(f'noun_quality_cs{CONTEXT_SIZE}_age_binned.csv')

print(normalized_df)