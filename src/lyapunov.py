"""Jacobian extraction and Lyapunov analysis."""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_continuous_lyapunov

from src.basis import build_dictionary
from src.identification import DataDrivenModel


def jacobian_at_origin(model: DataDrivenModel, h: float = 1e-6) -> np.ndarray:
    x0 = np.zeros((1, 2))
    j = np.zeros((2, 2))
    for i in range(2):
        xp = x0.copy()
        xm = x0.copy()
        xp[0, i] += h
        xm[0, i] -= h
        fp = model.predict(xp)[0]
        fm = model.predict(xm)[0]
        j[:, i] = (fp - fm) / (2 * h)
    return j


def solve_lyapunov_matrix(a: np.ndarray, q: np.ndarray | None = None) -> np.ndarray:
    if q is None:
        q = np.eye(a.shape[0])
    return solve_continuous_lyapunov(a.T, -q)


def lyapunov_value(x: np.ndarray, p: np.ndarray) -> np.ndarray:
    return np.einsum("...i,ij,...j->...", x, p, x)


def is_hurwitz(a: np.ndarray) -> tuple[bool, np.ndarray]:
    eigvals = np.linalg.eigvals(a)
    return bool(np.all(np.real(eigvals) < 0)), eigvals
