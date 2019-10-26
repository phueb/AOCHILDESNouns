from pathlib import Path


class Dirs:
    root = Path(__file__).parent.parent
    src = Path(__file__).parent
    scripts = src.parent / 'scripts'
    corpora = src / 'corpora'
    words = src / 'words'


class Symbols:
    OOV = 'OOV'
    TITLED = 'TITLED'
    all = [OOV, TITLED, 'xxx']


class Fig:
    fontsize = 12