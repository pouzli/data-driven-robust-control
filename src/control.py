"""Linear feedback control helpers."""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_continuous_are


def lqr_gain(a: np.ndarray, b: np.ndarray | None = None, q: np.ndarray | None = None, r: np.ndarray | None = None) -> np.ndarray:
    if b is None:
        b = np.eye(a.shape[0])
    if q is None:
        q = np.eye(a.shape[0])
    if r is None:
        r = np.eye(b.shape[1])

    p = solve_continuous_are(a, b, q, r)
    k = np.linalg.inv(r) @ b.T @ p
    return k
