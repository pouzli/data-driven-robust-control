"""Simulation routines for true and identified systems."""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from src.identification import DataDrivenModel
from src.uncertainty import bounded_disturbance


def simulate_true_system(system_f, x0: np.ndarray, t_eval: np.ndarray):
    sol = solve_ivp(lambda t, x: system_f(t, x), (t_eval[0], t_eval[-1]), x0, t_eval=t_eval, rtol=1e-7, atol=1e-9)
    return sol.t, sol.y.T


def estimate_derivative(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    dt = np.gradient(t)
    dx1 = np.gradient(x[:, 0], t)
    dx2 = np.gradient(x[:, 1], t)
    return np.column_stack([dx1, dx2])


def simulate_identified(model: DataDrivenModel, x0: np.ndarray, t_eval: np.ndarray, epsilon: float = 0.0, k: np.ndarray | None = None):
    def dyn(t, x):
        x_row = x.reshape(1, -1)
        fhat = model.predict(x_row)[0]
        disturbance = bounded_disturbance(t, epsilon) if epsilon > 0 else np.zeros(2)
        control = -(k @ x) if k is not None else np.zeros(2)
        return fhat + disturbance + control

    sol = solve_ivp(dyn, (t_eval[0], t_eval[-1]), x0, t_eval=t_eval, rtol=1e-7, atol=1e-9)
    return sol.t, sol.y.T


def generate_dataset(system_f, initial_conditions: list[np.ndarray], t_eval: np.ndarray):
    all_x = []
    all_xdot = []
    all_tid = []
    all_traj = []
    for traj_id, x0 in enumerate(initial_conditions):
        t, x = simulate_true_system(system_f, x0, t_eval)
        x_dot = estimate_derivative(x, t)
        all_x.append(x)
        all_xdot.append(x_dot)
        all_tid.append(np.full(len(t), traj_id))
        all_traj.append({"traj_id": traj_id, "t": t, "x": x})
    return {
        "x": np.vstack(all_x),
        "x_dot": np.vstack(all_xdot),
        "traj_id": np.concatenate(all_tid),
        "trajectories": all_traj,
    }
