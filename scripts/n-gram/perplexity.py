import kenlm
from subprocess import Popen, PIPE
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from itertools import cycle
from pathlib import Path
import tempfile
import attr

from wordplay import config
from preppy import PartitionedPrep as TrainPrep
from categoryeval.probestore import ProbeStore

from wordplay.word_sets import excluded
from wordplay.params import PrepParams
from wordplay.docs import load_docs

# /////////////////////////////////////////////////////////////////

CORPUS_NAME = 'childes-20180319'
PROBES_NAME = 'sem-all'

docs = load_docs(CORPUS_NAME)

params = PrepParams()
prep = TrainPrep(docs, **attr.asdict(params))

probe_store = ProbeStore(CORPUS_NAME, PROBES_NAME, prep.store.w2id, excluded=excluded)

# /////////////////////////////////////////////////////////////////

NGRAM_SIZES = [2, 3, 4, 5, 6]  # must be 2, 3, 4, 5, or 6
LMPLZ_PATH = '/home/ph/kenlm/bin/lmplz'  # these binaries must be installed by user
BINARIZE_PATH = '/home/ph/kenlm/bin/build_binary'


def calc_pps(str1, str2):
    result = []
    for s1, s2 in [(str1, str1), (str2, str2)]:

        # train n-gram model
        with tempfile.TemporaryFile('w') as fp:
            fp.write(s1)
            train_process = Popen([LMPLZ_PATH, '-o', str(ngram_size)], stdin=fp, stdout=PIPE)

        # save model
        out_path = Path(__file__).parent / '{}_{}-grams.arpa'.format(CORPUS_NAME, ngram_size)
        if not out_path.exists():
            out_path.touch()
        arpa_file_bytes = train_process.stdout.read()
        out_path.write_text(arpa_file_bytes.decode())

        # binarize model
        klm_file_path = str(out_path).rstrip('arpa') + 'klm'
        binarize_process = Popen([BINARIZE_PATH, str(out_path), klm_file_path])
        binarize_process.wait()

        # load model
        print('Computing perplexity using {}-gram model...'.format(ngram_size))
        model = kenlm.Model(klm_file_path)

        # score
        pp = model.perplexity(s2)
        result.append(pp)
    print(result)
    return result


tokens1 = prep.store.tokens[:prep.midpoint]
tokens2 = prep.store.tokens[-prep.midpoint:]

xys = []
for ngram_size in NGRAM_SIZES:
    y = calc_pps(' '.join(tokens1), ' '.join(tokens2))
    xys.append((y, ngram_size))

# fig
bar_width = 0.35
fig, ax = plt.subplots(dpi=config.Fig.dpi)
palette = cycle(sns.color_palette("hls", 2)[::-1])
ax.set_title('')
ax.set_ylabel('Perplexity')
ax.set_xlabel('N-gram size')
# ax.set_ylim([0, 30])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='both', top=False, right=False)
ax.set_xticks(np.array(NGRAM_SIZES) + bar_width / 2)
ax.set_xticklabels(NGRAM_SIZES)
# plot
for n, (y, ngram_size) in enumerate(xys):
    x = np.array([ngram_size, ngram_size + bar_width])
    for x_single, y_single, c, label in zip(x, y, palette, ['partition 1', 'partition 2']):
        label = label if n == 0 else '_nolegend_'  # label only once
        ax.bar(x_single, y_single, bar_width, color=c, label=label)
plt.legend()
plt.tight_layout()
fig.show()