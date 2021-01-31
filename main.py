import shutil
from sortedcontainers import SortedSet
import spacy
from spacy.tokens import Doc
import attr
import pandas as pd
from tabulate import tabulate

from abstractfirst.util import load_words
from abstractfirst import configs
from abstractfirst.memory import set_memory_limit
from abstractfirst.experiment import collect_dvs, prepare_data
from abstractfirst.params import Params

nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

set_memory_limit(0.9)

# IVs  # TODO


shutil.rmtree(configs.Dirs.images)
configs.Dirs.images.mkdir()

for params in [Params()]:  # todo

    targets = SortedSet(load_words(params.targets_name))
    age2text = prepare_data(params)

    dfs = []
    for age, text in sorted(age2text.items(), key=lambda i: i[0]):
        print()
        print(f'age={age}')
        params = attr.evolve(params, age=age)

        print('Tagging...')
        nlp.max_length = len(text)
        doc: Doc = nlp(text)

        df_age = collect_dvs(params, doc, targets)
        df_age['age'] = age
        dfs.append(df_age)  # get single row with all dvs in a single condition

    df_condition = pd.concat(dfs, axis=0)
    print(tabulate(df_condition,
                   headers=list(df_condition.columns),
                   tablefmt='simple'))
