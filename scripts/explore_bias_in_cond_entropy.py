from pyitlib import discrete_random_variable as drv
import numpy as np


N = 80_000
NUM_ROWS_AGE1 = 600
NUM_COLS_AGE1 = 2000  # age group 1
NUM_REPEAT = 10

x_ticks = np.arange(2, 2000, 50)


def compute_ce(num_add_rows: int = 0,
               num_add_cols: int = 0,
               ):

    vx = np.arange(NUM_ROWS_AGE1 + num_add_rows)
    vy = np.arange(NUM_COLS_AGE1 + num_add_cols)
    px = np.array([1 / (i + 1) for i in range(NUM_ROWS_AGE1 + num_add_rows)])
    px = px / px.sum()
    py = np.array([1 / (i + 1) for i in range(NUM_COLS_AGE1 + num_add_cols)])
    py = py / py.sum()

    # get non-entropic observations
    x_ = np.random.choice(vx, size=N, p=px)
    y_ = np.random.choice(vy, size=N, p=py)

    # compute
    xy_ = np.vstack((x_, y_))
    res = drv.entropy_conditional(x_, y_) / drv.entropy_joint(xy_)

    return res


xy_nce_age1 = np.mean([compute_ce(num_add_rows=0, num_add_cols=0) for _ in range(NUM_REPEAT)])
xy_nce_age2 = np.mean([compute_ce(num_add_rows=+30, num_add_cols=+600) for _ in range(NUM_REPEAT)])

# print normalized conditional entropy H(X|Y) (i.e. ce divided by joint entropy)
print(xy_nce_age1)
print(xy_nce_age2)
