"""Least-squares identification and residual analysis."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression

from src.basis import build_dictionary


class DataDrivenModel:
    """Model f_hat(x) = Theta(x) C."""

    def __init__(self, coef_matrix: np.ndarray):
        self.coef_matrix = coef_matrix

    def predict(self, x: np.ndarray) -> np.ndarray:
        theta = build_dictionary(x)
        return theta @ self.coef_matrix


def fit_least_squares(x: np.ndarray, x_dot: np.ndarray) -> DataDrivenModel:
    theta = build_dictionary(x)
    reg = LinearRegression(fit_intercept=False)
    reg.fit(theta, x_dot)
    c = reg.coef_.T
    return DataDrivenModel(coef_matrix=c)


def residuals(model: DataDrivenModel, x: np.ndarray, x_dot: np.ndarray) -> np.ndarray:
    return x_dot - model.predict(x)


def residual_metrics(r: np.ndarray, quantile: float = 0.95) -> dict:
    norms = np.linalg.norm(r, axis=1)
    rmse = float(np.sqrt(np.mean(np.sum(r**2, axis=1))))
    eps = float(np.quantile(norms, quantile))
    return {"rmse": rmse, "epsilon": eps, "residual_norms": norms}
