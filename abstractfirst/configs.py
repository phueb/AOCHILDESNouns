from pathlib import Path


class Dirs:
    src = Path(__file__).parent
    root = src.parent
    scripts = root / 'scripts'
    corpora = root / 'corpora'
    words = root / 'words'
    images = root / 'images'


class Fig:
    ax_fontsize = 20
    leg_fontsize = 12
    dpi = 300


class Data:
    make_last_bin_larger = True


class Conditions:
    directions = ['r']  # ['l', 'r', 'b']