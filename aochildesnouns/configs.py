from pathlib import Path


class Dirs:
    src = Path(__file__).parent
    root = src.parent
    scripts = root / 'scripts'
    corpora = root / 'corpora'
    targets = root / 'targets'
    images = root / 'images'
    tmp = root / 'tmp'
    results = root / 'results'
    co_data = root / 'co_data'


class Fig:
    ax_fontsize = 20
    leg_fontsize = 12
    dpi = 300
    max_projection = 0  # set to 0 to prevent plotting of reconstructions


class Data:
    make_last_bin_larger = True  # last age range has far fewer transcripts than first age bin: adjust?

    # this is invariant to lemmatisation, direction, normalization, ... everything but age, direction, ctl vs exp
    max_sum = {False: 81_000,   # max num co-occurrences when targets_control is False
               True: 81_000,    # max num co-occurrences when targets_control is True
               }

    punctuation = {'.', '!', '?'}
    eos = '[EOS]'


class Conditions:
    directions = ['l']  # 'l' for 'left', 'r' for 'right', 'b' for 'both'


class Figs:
    ax_font_size = 14
    leg_font_size = 10
    title_font_size = 16
    tick_font_size = 8
