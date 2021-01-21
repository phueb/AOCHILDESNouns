from scipy import stats
import numpy as np
from tabulate import tabulate

from wordplay.binned import make_age_bin2data
from wordplay.binned import make_age_bin2data_with_min_size


# ///////////////////////////////////////////////////////////////// parameters

CORPUS_NAME = 'childes-20191112'
AGE_STEP = 100
NUM_TOKENS_PER_BIN = 100_000  # 100K is good with AGE_STEP=100

# ///////////////////////////////////////////////////////////////// combine docs by age

age_bin2tags_ = make_age_bin2data(CORPUS_NAME, AGE_STEP, suffix='_tags')

for word_tokens in age_bin2tags_.values():  # this is used to determine maximal NUM_TOKENS_PER_BIN
    print(f'{len(word_tokens):,}')

# combine small bins
age_bin2tags = make_age_bin2data_with_min_size(age_bin2tags_, NUM_TOKENS_PER_BIN)

num_bins = len(age_bin2tags)

# use this to exclude from sem-all probes the 16 least frequent
# such that only those 720 probes remain which were used by Huebner & Willits, 2018
excluded_probes_path = configs.Dirs.words / 'excluded_sem-probes.txt'
excluded = set(excluded_probes_path.read_text().split('\n'))

# TODO collect all metrics in a single loop