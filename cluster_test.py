from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from typing import cast, no_type_check
from typing_extensions import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import seaborn as sbn
from numpy import ndarray
from pandas import DataFrame, Series
from tqdm import tqdm

# three possible aligned clusterings, but note 2 and 0 are not aligned
C0 = np.array([0, 0, 1], dtype=bool)
C1 = np.array([0, 1, 1], dtype=bool)
C2 = np.array([0, 1, 0], dtype=bool)
C_2_0 = np.array([1, 0, 1], dtype=bool)  # aligned to max agreement with C0

# Hungarian-aligned pairs are
aligned_pairs = [(C0, C1), (C0, C_2_0), (C1, C2)]

# Imagine we keep running the above algorithm, and only those results can
# be produced. Then we can enumerate all possible error / disagreement sets
# result from the pairings below:
# fmt: off
disagreement_pairings = [
    (C0, C0),
    (C0, C1),
    (C0, C_2_0),
    (C1, C1),
    (C1, C2),
    (C2, C2),
]
# fmt: on

# Then the possible disagreement sets are:
all_disagree_intersects = [np.sum(c1 & c2) for c1, c2 in disagreement_pairings]
all_disagree_unions = [np.sum(c1 | c2) for c1, c2 in disagreement_pairings]

N_ITERS = 5000
consistencies, means, sds = [], [], []
for i in tqdm(range(N_ITERS), total=N_ITERS):
    idx = int(np.random.randint(0, len(all_disagree_intersects), size=1))
    union = all_disagree_unions[idx]
    denom = np.sum(union)
    if denom < 1:
        consistency = 1.0
    else:
        intersect = all_disagree_intersects[idx]
        numerator = np.sum(intersect)
        consistency = intersect / union
    consistencies.append(consistency)
    means.append(np.mean(consistencies))
    if i != 0:
        sds.append(np.std(consistencies, ddof=1))

x = list(range(N_ITERS))
sbn.set_style("darkgrid")
fig, ax = plt.subplots()
# ax.plot(x, consistencies, label="consistency", color="black")
ax.plot(x, means, label="mean")
ax.plot(x[1:], sds, label="sd")
ax.legend().set_visible(True)
final_mean = np.round(np.mean(means[-N_ITERS // 10 :]), 2)
final_sd = np.round(np.mean(sds[-N_ITERS // 10 :]), 3)
ax.set_title(
    f"Convergence Rate of Consistency of Disagreement\nFinal: mean={final_mean}, sd={final_sd}"
)
ax.set_xlabel("Number of Clusterings Produced")
plt.show()
