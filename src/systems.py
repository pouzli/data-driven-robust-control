"""Definitions of benchmark nonlinear systems used in synthetic experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class SystemDefinition:
    """Container for a two-dimensional nonlinear dynamical system."""

    name: str
    params: dict
    dynamics: Callable[[float, np.ndarray, dict], np.ndarray]

    def f(self, t: float, x: np.ndarray) -> np.ndarray:
        return self.dynamics(t, x, self.params)


def quadratic_oscillator(t: float, x: np.ndarray, p: dict) -> np.ndarray:
    x1, x2 = x
    return np.array([
        x2,
        -(p["omega"] ** 2) * x1 - p["alpha"] * x2 + p["beta"] * x1**2,
    ])


def van_der_pol(t: float, x: np.ndarray, p: dict) -> np.ndarray:
    x1, x2 = x
    return np.array([x2, p["mu"] * (1 - x1**2) * x2 - x1])


def cross_nonlinear(t: float, x: np.ndarray, p: dict) -> np.ndarray:
    x1, x2 = x
    return np.array([
        -x1 + x2 + p["gamma"] * x1 * x2,
        -2 * x2 + p["delta"] * x1**2,
    ])


def saturation_system(t: float, x: np.ndarray, p: dict) -> np.ndarray:
    x1, x2 = x
    return np.array([-x1 + np.tanh(x2), -x2])


def get_systems() -> list[SystemDefinition]:
    """Return all systems required by the project statement."""

    return [
        SystemDefinition(
            name="quadratic_oscillator",
            params={"omega": 1.2, "alpha": 0.4, "beta": 0.3},
            dynamics=quadratic_oscillator,
        ),
        SystemDefinition(
            name="van_der_pol",
            params={"mu": 1.0},
            dynamics=van_der_pol,
        ),
        SystemDefinition(
            name="cross_nonlinear",
            params={"gamma": 0.8, "delta": 0.6},
            dynamics=cross_nonlinear,
        ),
        SystemDefinition(
            name="saturation_system",
            params={},
            dynamics=saturation_system,
        ),
    ]
