from __future__ import annotations

from typing import Tuple

import cvxpy as cp
import numpy as np


def _solve_problem(problem: cp.Problem) -> bool:
    try:
        problem.solve(solver=cp.OSQP, verbose=False)
    except Exception:
        try:
            problem.solve(solver=cp.ECOS, verbose=False)
        except Exception:
            return False
    return problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}


def fit_monotone_smooth(
    u: np.ndarray,
    hat_v: np.ndarray,
    w: np.ndarray,
    alpha: float = 1e-2,
    beta: float = 1e-2,
    slope_min: float | None = None,
    slope_max: float | None = None,
) -> np.ndarray:
    T1 = len(u)
    if T1 == 0:
        return np.zeros((0,), dtype=np.float32)
    if T1 == 1:
        return np.array([0.0], dtype=np.float32)

    v = cp.Variable(T1)
    weights = w if w is not None else np.ones((T1,), dtype=np.float32)

    fit_term = cp.sum(cp.multiply(weights, cp.square(v - hat_v)))
    smooth1 = cp.sum_squares(v[1:] - v[:-1])
    smooth2 = cp.sum_squares(v[2:] - 2 * v[1:-1] + v[:-2]) if T1 >= 3 else 0.0
    objective = fit_term + alpha * smooth1 + beta * smooth2

    constraints = [v[0] == 0.0, v[-1] == 1.0, v[1:] >= v[:-1]]

    delta = 1.0 / (T1 - 1)
    if slope_min is not None:
        constraints.append(v[1:] - v[:-1] >= slope_min * delta)
    if slope_max is not None:
        constraints.append(v[1:] - v[:-1] <= slope_max * delta)

    problem = cp.Problem(cp.Minimize(objective), constraints)
    ok = _solve_problem(problem)

    if not ok and (slope_min is not None or slope_max is not None):
        constraints = [v[0] == 0.0, v[-1] == 1.0, v[1:] >= v[:-1]]
        problem = cp.Problem(cp.Minimize(objective), constraints)
        ok = _solve_problem(problem)

    if not ok:
        # Fallback: only fit + monotone + first order smooth
        objective = fit_term + alpha * smooth1
        constraints = [v[0] == 0.0, v[-1] == 1.0, v[1:] >= v[:-1]]
        problem = cp.Problem(cp.Minimize(objective), constraints)
        ok = _solve_problem(problem)

    if not ok or v.value is None:
        raise RuntimeError("QP solve failed")

    v_val = np.clip(np.asarray(v.value).astype(np.float32), 0.0, 1.0)
    return v_val
