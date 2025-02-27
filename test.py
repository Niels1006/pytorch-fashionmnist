import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# matplotlib.use("TkAgg")  # Use a different backend (try 'Qt5Agg' or 'Agg' if needed)
plt.rcParams["font.size"] = 18

ar = [i for i in range(100)]

ar = np.array(ar)
N = 10


def mean(ar, N):
    M = int(np.ceil(len(ar) / 10) * 10 - len(ar))
    return np.nanmean(np.concatenate([ar, np.full(M, np.nan)]).reshape(-1, N), axis=1)


plt.plot(mean(ar, N))

plt.show()
