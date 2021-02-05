"""
Run a full experiment, by collecting DVs in all conditions, and exporting resulting data frame
"""

import shutil
import attr
import pandas as pd

from abstractfirst.co_occurrence import collect_left_and_right_co_occurrences
from abstractfirst.targets import make_targets
from abstractfirst import configs
from abstractfirst.measure import measure_dvs
from abstractfirst.pre_processing import prepare_data
from abstractfirst.params import Conditions


shutil.rmtree(configs.Dirs.images)
configs.Dirs.images.mkdir()

# all dvs will be collected in data frames, which are concatenated at the end
df_rows = []
row_id = 0

for params in Conditions.all():  # each param holds information about IVs in a single condition

    # load spacy-processed data to analyze, and targets
    age2doc = prepare_data(params)
    targets_exp, targets_ctl = make_targets(params, age2doc)

    # for each target condition (experimental vs control)
    for targets_control, targets in zip([False, True],
                                        [targets_exp, targets_ctl]):

        # for each age
        for age, doc in sorted(age2doc.items(), key=lambda i: i[0]):

            # for each direction (left, right, both)
            for direction in configs.Conditions.directions:

                # update + print IV realizations in current condition
                print()
                params = attr.evolve(params,
                                     age=age,
                                     targets_control=targets_control,
                                     direction=direction,
                                     )
                print(params)

                # get co-occurrence data
                # warning: do not call this function until attr.evolve(params) has been called
                co_data = collect_left_and_right_co_occurrences(doc, targets, params)

                # measure Dvs
                data: dict = measure_dvs(params, co_data)  # a full set of dvs for a single condition
                row_id += 1

                # add info about condition to df row - don't use magic methods which are uninfluenced attr.evolve()
                ivs = {ivn: attr.asdict(params)[ivn] for ivn in Conditions.ivs}
                data.update(ivs)
                row = pd.DataFrame(data=data, index=[row_id])

                # collect each row here
                df_rows.append(row)


df = pd.concat(df_rows, axis=0)
df.to_csv(configs.Dirs.results / 'results.csv', index=False)
