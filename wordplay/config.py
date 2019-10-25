from pathlib import Path


class Dirs:
    root = Path(__file__).parent.parent
    src = Path(__file__).parent
    analysis = src.parent / 'analysis'
    data = root / 'data'


class Symbols:
    OOV = 'OOV'
    TITLED = 'TITLED'
    all = [OOV, TITLED, 'xxx']


class Fig:
    fontsize = 12