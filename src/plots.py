"""Plotting utilities for experiments."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from src.lyapunov import lyapunov_value


def plot_phase_portrait(ax, trajectories, title: str):
    for tr in trajectories:
        x = tr["x"]
        ax.plot(x[:, 0], x[:, 1], lw=1.2)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)


def plot_residual_hist(ax, residual_norms: np.ndarray, epsilon: float, title: str):
    ax.hist(residual_norms, bins=40, alpha=0.75)
    ax.axvline(epsilon, color="r", ls="--", label=f"epsilon={epsilon:.3f}")
    ax.set_title(title)
    ax.set_xlabel("||r||")
    ax.set_ylabel("count")
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_lyapunov_contours(ax, p: np.ndarray, box: dict, title: str):
    x1 = np.linspace(box["x1_min"], box["x1_max"], 160)
    x2 = np.linspace(box["x2_min"], box["x2_max"], 160)
    xx1, xx2 = np.meshgrid(x1, x2)
    grid = np.column_stack([xx1.ravel(), xx2.ravel()])
    v = lyapunov_value(grid, p).reshape(xx1.shape)
    cs = ax.contour(xx1, xx2, v, levels=12)
    ax.clabel(cs, inline=True, fontsize=8)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)


def plot_trajectories(ax, t: np.ndarray, x: np.ndarray, title: str):
    ax.plot(t, x[:, 0], label="x1")
    ax.plot(t, x[:, 1], label="x2")
    ax.set_xlabel("t")
    ax.set_ylabel("states")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
