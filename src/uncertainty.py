"""Uncertainty bounds and disturbance generation."""

from __future__ import annotations

import numpy as np


def bounding_box(x: np.ndarray) -> dict:
    return {
        "x1_min": float(np.min(x[:, 0])),
        "x1_max": float(np.max(x[:, 0])),
        "x2_min": float(np.min(x[:, 1])),
        "x2_max": float(np.max(x[:, 1])),
    }


def bounded_disturbance(t: float, epsilon: float) -> np.ndarray:
    """Deterministic bounded disturbance with ||Delta|| <= epsilon."""

    raw = np.array([np.sin(1.7 * t), np.cos(0.9 * t)])
    norm = np.linalg.norm(raw) + 1e-12
    return epsilon * raw / norm
