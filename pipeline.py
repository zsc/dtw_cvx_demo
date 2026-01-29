from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from cost import compute_cost_band
from features import extract_whisper_features, l2_normalize
from mincost_flow_dtw import FlowResult, NoPathError, solve_dtw_mincost_flow
from monotone_qp import fit_monotone_smooth
from path_to_mapping import path_to_hat_v
from warp import build_warp, Warp


@dataclass
class AlignmentResult:
    mapping: Dict
    warp: Warp
    flow_cost: float


def _linear_mapping(T1: int, T2: int, D1: float, D2: float) -> AlignmentResult:
    if T1 <= 1:
        u = np.array([0.0], dtype=np.float32)
        v = np.array([0.0], dtype=np.float32)
        path = [(0, 0)]
    else:
        u = np.linspace(0.0, 1.0, T1).astype(np.float32)
        v = u.copy()
        if T2 <= 1:
            path = [(i, 0) for i in range(T1)]
        else:
            path = [(i, int(round(i * (T2 - 1) / (T1 - 1)))) for i in range(T1)]
    mapping = {
        "u": u.tolist(),
        "v": v.tolist(),
        "path": path,
        "durations": {"D1": D1, "D2": D2},
    }
    warp = build_warp(u, v, D1, D2)
    return AlignmentResult(mapping=mapping, warp=warp, flow_cost=0.0)


def _path_valid(path: List[Tuple[int, int]], T1: int, T2: int) -> bool:
    if not path:
        return False
    if path[0] != (0, 0) or path[-1] != (T1 - 1, T2 - 1):
        return False
    for (i0, j0), (i1, j1) in zip(path[:-1], path[1:]):
        di = i1 - i0
        dj = j1 - j0
        if (di, dj) not in {(1, 0), (0, 1), (1, 1)}:
            return False
    return True


def _expand_band(band_radius: float | None, expand: float) -> float | None:
    if band_radius is None:
        return None
    return min(1.0, band_radius * expand)


def run_alignment(
    wav1_path: str,
    wav2_path: str,
    feature_mode: str = "whisper_encoder",
    model_name: str = "base",
    device: str = "cpu",
    dist: str = "cosine",
    gamma_time: float = 0.1,
    band_radius: float | None = 0.08,
    step_penalty: Dict[str, float] | None = None,
    cost_scale: float = 1_000_000,
    qp_alpha: float = 1e-2,
    qp_beta: float = 1e-2,
    slope_min: float | None = None,
    slope_max: float | None = None,
    max_band_tries: int = 4,
    band_expand: float = 1.5,
) -> AlignmentResult:
    X, D1 = extract_whisper_features(
        wav1_path, mode=feature_mode, model_name=model_name, device=device
    )
    Y, D2 = extract_whisper_features(
        wav2_path, mode=feature_mode, model_name=model_name, device=device
    )

    T1, T2 = len(X), len(Y)
    if T1 < 2 or T2 < 2:
        return _linear_mapping(max(T1, 1), max(T2, 1), D1, D2)

    if dist == "cosine":
        X = l2_normalize(X)
        Y = l2_normalize(Y)

    step_penalty = step_penalty or {"diag": 0.0, "horiz": 0.2, "vert": 0.2}

    flow_result: FlowResult | None = None
    used_band = band_radius
    for _ in range(max_band_tries):
        cost_map, allowed_js = compute_cost_band(X, Y, gamma_time, used_band, dist)
        try:
            flow_result = solve_dtw_mincost_flow(
                cost_map,
                allowed_js,
                T1,
                T2,
                step_penalty=step_penalty,
                cost_scale=cost_scale,
            )
        except NoPathError:
            flow_result = None
        if flow_result and _path_valid(flow_result.path, T1, T2):
            break
        used_band = _expand_band(used_band, band_expand)

    if flow_result is None:
        raise RuntimeError("Failed to find a feasible DTW path after band expansion")

    path = flow_result.path
    hat_v, weights = path_to_hat_v(path, T1, T2)
    u = np.linspace(0.0, 1.0, T1).astype(np.float32)

    v = fit_monotone_smooth(
        u,
        hat_v,
        weights,
        alpha=qp_alpha,
        beta=qp_beta,
        slope_min=slope_min,
        slope_max=slope_max,
    )

    warp = build_warp(u, v, D1, D2)

    mapping = {
        "u": u.tolist(),
        "v": v.tolist(),
        "path": path,
        "durations": {"D1": D1, "D2": D2},
        "config": {
            "feature_mode": feature_mode,
            "model_name": model_name,
            "dist": dist,
            "gamma_time": gamma_time,
            "band_radius": used_band,
            "step_penalty": step_penalty,
            "qp_alpha": qp_alpha,
            "qp_beta": qp_beta,
            "slope_min": slope_min,
            "slope_max": slope_max,
            "cost_scale": cost_scale,
        },
    }

    return AlignmentResult(mapping=mapping, warp=warp, flow_cost=flow_result.total_cost)
