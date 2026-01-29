from typing import Dict, List, Tuple

import numpy as np


def _compute_band_limits(T1: int, T2: int, band_radius: float | None) -> List[Tuple[int, int]]:
    if band_radius is None:
        return [(0, T2 - 1) for _ in range(T1)]

    limits = []
    for i in range(T1):
        if T1 == 1:
            u = 0.0
        else:
            u = i / (T1 - 1)
        # Solve |u - v| <= band_radius for j
        v_center = u * (T2 - 1)
        j_min = int(np.floor((v_center - band_radius * (T2 - 1))))
        j_max = int(np.ceil((v_center + band_radius * (T2 - 1))))
        j_min = max(0, j_min)
        j_max = min(T2 - 1, j_max)
        if j_min > j_max:
            j_min, j_max = 1, 0  # empty range
        limits.append((j_min, j_max))
    return limits


def compute_cost_band(
    X: np.ndarray,
    Y: np.ndarray,
    gamma_time: float,
    band_radius: float | None,
    dist: str = "cosine",
) -> Tuple[Dict[Tuple[int, int], float], List[List[int]]]:
    T1, _ = X.shape
    T2, _ = Y.shape

    if T1 == 0 or T2 == 0:
        return {}, [[] for _ in range(T1)]

    u = np.linspace(0.0, 1.0, T1) if T1 > 1 else np.array([0.0])
    v = np.linspace(0.0, 1.0, T2) if T2 > 1 else np.array([0.0])

    limits = _compute_band_limits(T1, T2, band_radius)

    cost_map: Dict[Tuple[int, int], float] = {}
    allowed_js: List[List[int]] = []

    for i in range(T1):
        j_min, j_max = limits[i]
        if j_min > j_max:
            allowed_js.append([])
            continue
        js = list(range(j_min, j_max + 1))
        allowed_js.append(js)
        y_slice = Y[j_min : j_max + 1]

        if dist == "cosine":
            dots = y_slice @ X[i]
            dots = np.clip(dots, -1.0, 1.0)
            content_cost = 1.0 - dots
        elif dist == "l2sq":
            diff = y_slice - X[i]
            content_cost = np.sum(diff * diff, axis=1)
        else:
            raise ValueError(f"Unsupported dist: {dist}")

        if gamma_time != 0.0:
            time_cost = gamma_time * np.abs(u[i] - v[j_min : j_max + 1])
            total_cost = content_cost + time_cost
        else:
            total_cost = content_cost

        for offset, j in enumerate(js):
            cost_map[(i, j)] = float(total_cost[offset])

    return cost_map, allowed_js
