"""Common helpers for IO and serialization."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def ensure_dirs(*paths: str | Path):
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def save_dataset_csv(x: np.ndarray, x_dot: np.ndarray, traj_id: np.ndarray, path: str | Path):
    df = pd.DataFrame(
        {
            "traj_id": traj_id,
            "x1": x[:, 0],
            "x2": x[:, 1],
            "x1_dot": x_dot[:, 0],
            "x2_dot": x_dot[:, 1],
        }
    )
    df.to_csv(path, index=False)


def to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, complex):
        return {"real": obj.real, "imag": obj.imag}
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    return obj


def save_json(data: dict, path: str | Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_serializable(data), f, ensure_ascii=False, indent=2)
