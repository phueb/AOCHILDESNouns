import pandas as pd
import pyprind
import sys
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import attr

from preppy import PartitionedPrep as TrainPrep

from wordplay import config
from wordplay.utils import plot_best_fit_line
from wordplay.params import PrepParams
from wordplay.docs import load_docs


# /////////////////////////////////////////////////////////////////

CORPUS_NAME = 'childes-20180319'
docs = load_docs(CORPUS_NAME)

params = PrepParams()
prep = TrainPrep(docs, **attr.asdict(params))

# ///////////////////////////////////////////////////////////////////

MCDIP_PATH = 'mcdip.csv'
CONTEXT_SIZE = 16  # bidirectional


# t2mcdip (map target to its mcdip value)
df = pd.read_csv(MCDIP_PATH, index_col=False)
to_drop = []  # remove targets from df if not in vocab
for n, t in enumerate(df['target']):
    if t not in prep.store.types:
        print('Dropping "{}"'.format(t))
        to_drop.append(n)
df = df.drop(to_drop)
targets = df['target'].values
mcdips = df['MCDIp'].values
t2mcdip = {t: mcdip for t, mcdip in zip(targets, mcdips)}

# collect context words of targets
print('Collecting context words...')
target2context_tokens = {t: [] for t in targets}
pbar = pyprind.ProgBar(prep.store.num_tokens, stream=sys.stdout)
for n, t in enumerate(prep.store.tokens):
    pbar.update()
    if t in targets:
        context_left = [ct for ct in prep.store.tokens[n - CONTEXT_SIZE: n] if ct in targets]
        context_right = [ct for ct in prep.store.tokens[n + 1: n + 1 + CONTEXT_SIZE] if ct in targets]
        target2context_tokens[t] += context_left + context_right

# calculate result for each target (average mcdip of context words weighted by number of times in target context)
res = {t: 0 for t in targets}
for t, cts in target2context_tokens.items():
    counter = Counter(cts)
    total_f = len(cts)
    res[t] = np.average([t2mcdip[ct] for ct in cts], weights=[counter[ct] / total_f for ct in cts])


def plot(xs, ys, xlabel, ylabel, annotations=None):
    fig, ax = plt.subplots(1, figsize=config.Fig.fig_size, dpi=config.Fig.dpi)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='both', top=False, right=False)
    # plot
    if annotations is not None:
        it = iter(annotations)
    for x, y in zip(xs, ys):
        ax.scatter(x, y, color='black')
        if annotations is not None:
            ax.annotate(next(it), (x + 0.005, y))
    # fit line
    plot_best_fit_line(ax, xs, ys, fontsize=12)
    plt.tight_layout()
    plt.show()


target_weighted_context_mcdip = [res[t] for t in targets]
target_mcdips = [t2mcdip[t] for t in targets]
target_freqs = [prep.store.w2f[t] for t in targets]

plot(target_weighted_context_mcdip, target_mcdips,
     'KWOOC', 'MCDIp')

plot(target_weighted_context_mcdip, np.log(target_freqs),
     'target_weighted_context_mcdip', 'target_freqs')

plot(target_mcdips, np.log(target_freqs),
     'target_mcdips', ' log target_freqs')