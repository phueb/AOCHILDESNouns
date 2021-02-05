import numpy as np


def calc_projection(u: np.array,
                    s: np.array,
                    vt: np.array,
                    dim_id: int,
                    ) -> np.array:
    res = s[dim_id] * u[:, dim_id].reshape(-1, 1) @ vt[dim_id, :].reshape(1, -1)
    return res


