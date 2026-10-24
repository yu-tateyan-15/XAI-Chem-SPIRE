"""Scaling for notebook-based atom-contribution experiments."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Literal, Sequence
import numpy as np


ScalingMode = Literal["none", "standard", "minmax"]


@dataclass(frozen=True)
class AffineLabelScaler:
    """Implement ``z = (y - beta) / alpha`` and its exact inverse."""

    mode: ScalingMode
    beta: float
    alpha: float

    def __post_init__(self) -> None:
        if self.mode not in ("none", "standard", "minmax"):
            raise ValueError(f"Unsupported label scaling mode: {self.mode}")
        if isinstance(self.beta, (bool, np.bool_)) or not isinstance(
            self.beta, Real
        ):
            raise ValueError("Label scaling beta must be a finite real number")
        if not np.isfinite(self.beta):
            raise ValueError("Label scaling beta must be finite")
        if isinstance(self.alpha, (bool, np.bool_)) or not isinstance(
            self.alpha, Real
        ):
            raise ValueError(
                "Label scaling alpha must be a positive finite real number"
            )
        if not np.isfinite(self.alpha) or self.alpha <= 0.0:
            raise ValueError(
                f"Label scaling requires a positive finite alpha, got {self.alpha}"
            )

    def transform(self, values: np.ndarray | Sequence[float]) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if not np.isfinite(array).all():
            raise ValueError("Label scaler transform input contains non-finite values")
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            transformed = (array - self.beta) / self.alpha
        if not np.isfinite(transformed).all():
            raise ValueError("Label scaler transform output contains non-finite values")
        return transformed

    def inverse_transform(self, values: np.ndarray | Sequence[float]) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if not np.isfinite(array).all():
            raise ValueError(
                "Label scaler inverse-transform input contains non-finite values"
            )
        with np.errstate(over="ignore", invalid="ignore"):
            transformed = self.beta + self.alpha * array
        if not np.isfinite(transformed).all():
            raise ValueError(
                "Label scaler inverse-transform output contains non-finite values"
            )
        return transformed


def fit_label_scaler(
    all_targets: np.ndarray | Sequence[float],
    train_indices: Sequence[int],
    *,
    mode: ScalingMode,
) -> AffineLabelScaler:
    """Fit scaling parameters from the training fold only."""

    targets = np.asarray(all_targets, dtype=np.float64).reshape(-1)
    indices = np.asarray(train_indices, dtype=np.int64).reshape(-1)
    if indices.size == 0:
        raise ValueError("train_indices must be non-empty")
    if indices.min() < 0 or indices.max() >= targets.size:
        raise IndexError("train_indices contain an out-of-range row")
    train_targets = targets[indices]
    if not np.isfinite(train_targets).all():
        raise ValueError("Training targets contain non-finite values")

    if mode == "none":
        beta, alpha = 0.0, 1.0
    elif mode == "standard":
        beta = float(train_targets.mean())
        alpha = float(train_targets.std(ddof=0))
    elif mode == "minmax":
        beta = float(train_targets.min())
        alpha = float(train_targets.max() - train_targets.min())
    else:
        raise ValueError(f"Unsupported label scaling mode: {mode}")
    if not np.isfinite(alpha) or alpha <= 0.0:
        raise ValueError(
            f"Label scaling requires a positive finite alpha, got {alpha}"
        )
    return AffineLabelScaler(mode=mode, beta=beta, alpha=alpha)
