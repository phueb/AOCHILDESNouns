import statsmodels.api as sm
import pandas as pd


def regress(x, y):
    """
    @param x: pd.DataFrame
    @param y: pd.DataFrame or pd.Series
    """

    xy = sm.add_constant(pd.concat([x, y], axis=1))
    res = sm.OLS(xy.iloc[:, -1], xy.iloc[:, :-1], missing='drop').fit()

    summary = res.summary()

    return summary