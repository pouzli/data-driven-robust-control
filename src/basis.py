"""Feature dictionary for data-driven identification."""

from __future__ import annotations

import numpy as np


FEATURE_NAMES = ["1", "x1", "x2", "x1^2", "x1*x2", "x2^2", "x1^3", "x2^3"]


def build_dictionary(x: np.ndarray) -> np.ndarray:
    """Build polynomial feature matrix Theta(x).

    Parameters
    ----------
    x: array, shape (n_samples, 2)

    Returns
    -------
    Theta: array, shape (n_samples, n_features)
    """

    x1 = x[:, 0]
    x2 = x[:, 1]
    return np.column_stack(
        [
            np.ones(len(x)),
            x1,
            x2,
            x1**2,
            x1 * x2,
            x2**2,
            x1**3,
            x2**3,
        ]
    )
