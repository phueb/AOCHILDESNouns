from pathlib import Path


class Dirs:
    src = Path(__file__).parent
    root = src.parent
    scripts = root / 'scripts'
    corpora = root / 'corpora'
    targets = root / 'targets'
    images = root / 'images'
    results = root / 'results'


class Fig:
    ax_fontsize = 20
    leg_fontsize = 12
    dpi = 300
    max_projection = 0  # set to 0 to prevent plotting of reconstructions


class Data:
    make_last_bin_larger = True
    max_sum = {False: 88_000,  # max num co-occurrences when targets_control is False
               True: 49_000,    # max num co-occurrences when targets_control is True
               }
    exclude_exp_from_ctl_targets = False


class Conditions:
    directions = ['r']  # ['l', 'r', 'b']