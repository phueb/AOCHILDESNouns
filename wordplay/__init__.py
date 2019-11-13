from wordplay import config

__version__ = '0.0.0'

# use this to exclude from sem-all probes the 16 least frequent
# such that only those 720 probes remain which were used by Huebner & Willits, 2018
excluded_probes_path = config.Dirs.words / 'excluded_sem-probes.txt'
excluded = set(excluded_probes_path.read_text().split('\n'))