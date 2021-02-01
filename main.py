"""
Run a full experiment, by collecting DVs in all conditions into 1 data frame
"""

import shutil

import attr
import pandas as pd
from tabulate import tabulate

from abstractfirst.util import make_targets_ctl
from abstractfirst import configs
from abstractfirst.memory import set_memory_limit
from abstractfirst.experiment import measure_dvs, prepare_data
from abstractfirst.params import Params


set_memory_limit(0.9)

# IVs  # TODO


shutil.rmtree(configs.Dirs.images)
configs.Dirs.images.mkdir()

for params in [Params()]:  # todo

    # load spacy-processed data to analyze
    age2doc = prepare_data(params)

    # all dvs will be collected in data frames, which are concatenated at the end
    dfs = []

    # for each target condition (experimental vs control)
    targets_exp, targets_ctl = make_targets_ctl(params)
    for targets_control, targets in zip([False, True],
                                        [targets_exp, targets_ctl]):

        # for each age
        for age, doc in sorted(age2doc.items(), key=lambda i: i[0]):

            # update + print conditions
            print()
            params = attr.evolve(params, targets_control=targets_control)
            params = attr.evolve(params, age=age)
            print(params)

            # measure Dvs
            df_age = measure_dvs(params, doc, targets)  # a full set of dvs for a single condition

            # add info about condition
            df_age['targets_control'] = targets_control
            df_age['age'] = age
            dfs.append(df_age)

    df_condition = pd.concat(dfs, axis=0)
    print(tabulate(df_condition,
                   headers=list(df_condition.columns),
                   tablefmt='simple'))
