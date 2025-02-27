import numpy as np


def mean(ar, N):
    M = int(np.ceil(len(ar) / 10) * 10 - len(ar))
    return np.nanmean(np.concatenate([ar, np.full(M, np.nan)]).reshape(-1, N), axis=1)
