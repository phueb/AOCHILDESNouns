from pathlib import Path


class Dirs:
    src = Path(__file__).parent
    root = src.parent
    scripts = root / 'scripts'
    corpora = root / 'corpora'
    targets = root / 'targets'
    images = root / 'images'
    results = root / 'results'
    co_data = root / 'co_data'


class Fig:
    ax_fontsize = 20
    leg_fontsize = 12
    dpi = 300
    max_projection = 0  # set to 0 to prevent plotting of reconstructions


class Data:
    make_last_bin_larger = True  # last age range occurs far fewer transcripts that first age bin: adjust?
    max_sum = {False: 76_000,   # max num co-occurrences when targets_control is False
               True: 102_000,    # max num co-occurrences when targets_control is True
               }
    exclude_exp_from_ctl_targets = True

    punctuation = {'.', '!', '?'}
    eos = '[EOS]'


class Conditions:
    directions = ['r']  # ['l', 'r', 'b']


class Figs:
    ax_font_size = 14
    leg_font_size = 10
    title_font_size = 16
    tick_font_size = 8