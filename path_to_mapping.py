from typing import List, Tuple

import numpy as np


def path_to_hat_v(path: List[Tuple[int, int]], T1: int, T2: int):
    if T1 == 0 or T2 == 0:
        return np.zeros((T1,), dtype=np.float32), np.zeros((T1,), dtype=np.float32)

    js_by_i: List[List[int]] = [[] for _ in range(T1)]
    for i, j in path:
        if 0 <= i < T1:
            js_by_i[i].append(j)

    hat_v = np.zeros((T1,), dtype=np.float32)
    weights = np.zeros((T1,), dtype=np.float32)

    last_valid = 0
    for i in range(T1):
        js = js_by_i[i]
        if js:
            median_j = int(np.median(js))
            last_valid = median_j
            hat_v[i] = median_j / float(T2 - 1) if T2 > 1 else 0.0
            weights[i] = float(len(js))
        else:
            hat_v[i] = last_valid / float(T2 - 1) if T2 > 1 else 0.0
            weights[i] = 1.0

    return hat_v, weights
