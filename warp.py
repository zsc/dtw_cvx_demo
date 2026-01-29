from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class Warp:
    u: np.ndarray
    v: np.ndarray
    D1: float
    D2: float

    def warp_u(self, u_query: float) -> float:
        u_clamped = float(np.clip(u_query, 0.0, 1.0))
        return float(np.interp(u_clamped, self.u, self.v))

    def warp_time(self, t1_seconds: float) -> float:
        if self.D1 <= 0.0:
            return 0.0
        t1 = float(np.clip(t1_seconds, 0.0, self.D1))
        u_query = t1 / self.D1
        v_query = self.warp_u(u_query)
        return float(np.clip(v_query * self.D2, 0.0, self.D2))


def build_warp(u: np.ndarray, v: np.ndarray, D1: float, D2: float) -> Warp:
    return Warp(u=u, v=v, D1=D1, D2=D2)
