"""SPIRE figures following Modeling/src/analysis/visualization.py.

The standalone distribution retains the project's train/validation colours,
prediction-on-x parity plots, raw bwr AC maps, and 300 dpi export policy.
Only endpoint labels and units are generalized; numeric ACs are never rescaled.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

from .results import CompletedFoldResults
from .training import compute_regression_metrics

_SPLIT_ORDER = ("train", "validation")


def decode_png_bytes(image_bytes: bytes) -> np.ndarray:
    """Decode RDKit PNG bytes into a matplotlib-ready array."""

    return plt.imread(BytesIO(image_bytes), format="png")

def compute_white_crop_box(
    image: np.ndarray,
    *,
    white_threshold: float = 0.995,
    pad_px: int = 2,
) -> tuple[int, int, int, int]:
    """Return crop bounds that remove near-white outer margins."""

    array = np.asarray(image)
    if array.ndim != 3 or array.shape[-1] < 3:
        return 0, int(array.shape[0]), 0, int(array.shape[1])

    non_white = np.any(array[..., :3] < float(white_threshold), axis=-1)
    if not np.any(non_white):
        return 0, int(array.shape[0]), 0, int(array.shape[1])

    rows = np.where(np.any(non_white, axis=1))[0]
    columns = np.where(np.any(non_white, axis=0))[0]
    pad = max(int(pad_px), 0)
    return (
        max(int(rows[0]) - pad, 0),
        min(int(rows[-1]) + pad + 1, array.shape[0]),
        max(int(columns[0]) - pad, 0),
        min(int(columns[-1]) + pad + 1, array.shape[1]),
    )

def decode_and_trim_png(
    image_bytes: bytes,
    *,
    white_threshold: float = 0.995,
    pad_px: int = 2,
) -> np.ndarray:
    """Decode PNG bytes and remove excess white margins."""

    image = decode_png_bytes(image_bytes)
    row_start, row_stop, column_start, column_stop = compute_white_crop_box(
        image,
        white_threshold=white_threshold,
        pad_px=pad_px,
    )
    return image[row_start:row_stop, column_start:column_stop]

def resolve_contribution_color_range(
    *,
    vmin: float | None,
    vmax: float,
) -> tuple[float, float]:
    """Validate a zero-centred diverging contribution range."""

    maximum = float(vmax)
    minimum = -maximum if vmin is None else float(vmin)
    if not np.isfinite(minimum) or not np.isfinite(maximum):
        raise ValueError(f"vmin and vmax must be finite, got {vmin!r}, {vmax!r}")
    if not minimum < 0.0 < maximum:
        raise ValueError(
            "Atom-contribution color ranges must satisfy vmin < 0 < vmax, "
            f"got vmin={minimum!r}, vmax={maximum!r}"
        )
    return minimum, maximum

def _contribution_to_rgba(
    value: float,
    *,
    vmin: float,
    vmax: float,
    cmap_name: str,
) -> tuple[float, float, float, float]:
    """Map one raw value through the fixed zero-centred colour scale."""

    minimum, maximum = resolve_contribution_color_range(vmin=vmin, vmax=vmax)
    normalizer = TwoSlopeNorm(vmin=minimum, vcenter=0.0, vmax=maximum)
    clipped = float(np.clip(value, minimum, maximum))
    return tuple(float(component) for component in plt.get_cmap(cmap_name)(normalizer(clipped)))

def _positive_optional(name: str, value: float | None) -> float | None:
    if value is None:
        return None
    resolved = float(value)
    if not np.isfinite(resolved) or resolved <= 0.0:
        raise ValueError(f"{name} must be positive or None, got {value!r}")
    return resolved

def _alpha_optional(name: str, value: float | None) -> float | None:
    if value is None:
        return None
    resolved = float(value)
    if not np.isfinite(resolved) or not 0.0 <= resolved <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1 or None, got {value!r}")
    return resolved

def _with_alpha(
    rgb: tuple[float, float, float],
    alpha: float | None,
) -> tuple[float, ...]:
    return rgb if alpha is None else (*rgb, float(alpha))

def _atom_highlight_radii(
    contribution: np.ndarray,
    molecule: Chem.Mol,
    *,
    vmax: float,
    atom_highlight_radius: float | None,
    base_radius: float = 0.18,
    max_extra_radius: float = 0.20,
) -> dict[int, float]:
    """Use the project radius, leaving all contribution values unchanged."""

    fixed_radius = _positive_optional("atom_highlight_radius", atom_highlight_radius)
    if fixed_radius is not None:
        return {
            atom_index: fixed_radius
            for atom_index in range(molecule.GetNumAtoms())
        }

    maximum = max(float(vmax), 1.0e-6)
    return {
        atom_index: float(
            base_radius
            + max_extra_radius
            * min(abs(float(contribution[atom_index])) / maximum, 1.0)
        )
        for atom_index in range(molecule.GetNumAtoms())
    }

def _bond_highlights(
    molecule: Chem.Mol,
    contribution: np.ndarray,
    *,
    vmin: float,
    vmax: float,
    cmap_name: str,
    include_bond_colors: bool,
    bond_highlight_alpha: float | None,
) -> tuple[list[int], dict[int, tuple[float, ...]]]:
    """Colour bonds from endpoint means for visual continuity only."""

    if not include_bond_colors:
        return [], {}

    alpha = _alpha_optional("bond_highlight_alpha", bond_highlight_alpha)
    bond_indices: list[int] = []
    bond_colors: dict[int, tuple[float, ...]] = {}
    for bond in molecule.GetBonds():
        bond_index = int(bond.GetIdx())
        midpoint = (
            float(contribution[int(bond.GetBeginAtomIdx())])
            + float(contribution[int(bond.GetEndAtomIdx())])
        ) / 2.0
        rgb = _contribution_to_rgba(
            midpoint,
            vmin=vmin,
            vmax=vmax,
            cmap_name=cmap_name,
        )[:3]
        bond_indices.append(bond_index)
        bond_colors[bond_index] = _with_alpha(rgb, alpha)
    return bond_indices, bond_colors

def _apply_highlight_options(
    draw_options: Any,
    *,
    draw_atom_indices: bool,
    bond_highlight_thinness: float | None,
) -> None:
    """Apply the project RDKit highlight appearance and optional atom indices."""

    draw_options.addAtomIndices = draw_atom_indices
    if hasattr(draw_options, "atomHighlightsAreCircles"):
        draw_options.atomHighlightsAreCircles = True
    if hasattr(draw_options, "fillHighlights"):
        draw_options.fillHighlights = True
    if hasattr(draw_options, "standardColoursForHighlightedAtoms"):
        draw_options.standardColoursForHighlightedAtoms = True

    thinness = _positive_optional("bond_highlight_thinness", bond_highlight_thinness)
    if thinness is not None and hasattr(draw_options, "highlightBondWidthMultiplier"):
        current = float(draw_options.highlightBondWidthMultiplier)
        draw_options.highlightBondWidthMultiplier = max(1, int(round(current / thinness)))

def render_contribution_image(
    *,
    mol: Chem.Mol,
    contrib: np.ndarray,
    y_pred: float,
    y_true: float,
    vmin: float | None = None,
    vmax: float,
    cmap_name: str,
    width: int,
    height: int,
    include_bond_colors: bool,
    draw_atom_indices: bool,
    show_rdkit_legend: bool,
    atom_highlight_radius: float | None,
    bond_highlight_thinness: float | None,
    atom_highlight_alpha: float | None,
    bond_highlight_alpha: float | None,
) -> dict[str, Any]:
    """Render one molecule from an explicitly atom-aligned contribution vector."""

    contribution = np.asarray(contrib, dtype=float).reshape(-1)
    atom_count = int(mol.GetNumAtoms())
    if contribution.size != atom_count:
        raise ValueError(
            "contrib length must exactly match the RDKit atom count: "
            f"{contribution.size} != {atom_count}"
        )
    if not np.isfinite(contribution).all():
        raise ValueError("contrib must contain only finite values")

    minimum, maximum = resolve_contribution_color_range(vmin=vmin, vmax=vmax)
    atom_alpha = _alpha_optional("atom_highlight_alpha", atom_highlight_alpha)
    atom_colors = {
        atom_index: _with_alpha(
            _contribution_to_rgba(
                float(contribution[atom_index]),
                vmin=minimum,
                vmax=maximum,
                cmap_name=cmap_name,
            )[:3],
            atom_alpha,
        )
        for atom_index in range(atom_count)
    }
    atom_radii = _atom_highlight_radii(
        contribution,
        mol,
        vmax=max(abs(minimum), abs(maximum)),
        atom_highlight_radius=atom_highlight_radius,
    )
    bond_indices, bond_colors = _bond_highlights(
        mol,
        contribution,
        vmin=minimum,
        vmax=maximum,
        cmap_name=cmap_name,
        include_bond_colors=include_bond_colors,
        bond_highlight_alpha=bond_highlight_alpha,
    )

    drawer = rdMolDraw2D.MolDraw2DCairo(int(width), int(height))
    _apply_highlight_options(
        drawer.drawOptions(),
        draw_atom_indices=draw_atom_indices,
        bond_highlight_thinness=bond_highlight_thinness,
    )
    legend = (
        f"Pred: {float(y_pred):.2f} | True: {float(y_true):.2f} | "
        f"Err: {float(y_pred) - float(y_true):+.2f}"
        if show_rdkit_legend
        else ""
    )
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer,
        mol,
        highlightAtoms=list(range(atom_count)),
        highlightAtomColors=atom_colors,
        highlightAtomRadii=atom_radii,
        highlightBonds=bond_indices,
        highlightBondColors=bond_colors,
        legend=legend,
    )
    drawer.FinishDrawing()
    below_range = contribution < minimum
    above_range = contribution > maximum
    saturated_count = int(np.count_nonzero(below_range | above_range))
    return {
        "image": drawer.GetDrawingText(),
        "bond_colors": bond_colors,
        "rdkit_legend_text": legend,
        "vmin": minimum,
        "vmax": maximum,
        "color_saturated_atom_count": saturated_count,
        "color_saturated_atom_fraction": (
            float(saturated_count / atom_count) if atom_count else 0.0
        ),
        "color_saturated_negative_atom_count": int(
            np.count_nonzero(below_range)
        ),
        "color_saturated_positive_atom_count": int(
            np.count_nonzero(above_range)
        ),
    }

def _finite_pair(
    target: Sequence[float],
    prediction: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    target_array = np.asarray(target, dtype=float).reshape(-1)
    prediction_array = np.asarray(prediction, dtype=float).reshape(-1)
    if target_array.size != prediction_array.size or target_array.size < 2:
        raise ValueError(
            "Regression metrics require equally sized target/prediction "
            "vectors with at least two observations"
        )
    if not np.isfinite(target_array).all() or not np.isfinite(
        prediction_array
    ).all():
        raise ValueError("Regression metrics require finite values")
    return target_array, prediction_array

def _regression_metrics(
    target: Sequence[float],
    prediction: Sequence[float],
) -> dict[str, float | int]:
    target_array, prediction_array = _finite_pair(target, prediction)
    metrics = compute_regression_metrics(target_array, prediction_array)
    if not np.isfinite(metrics.r2_score):
        raise ValueError("R2 is undefined for a constant target vector")
    if not np.isfinite(metrics.pearson_r_squared):
        raise ValueError(
            "Pearson r-squared is undefined for a constant vector"
        )
    return {
        "n": metrics.count,
        "rmse_raw": metrics.rmse,
        "mae_raw": metrics.mae,
        "r2_score_raw": metrics.r2_score,
        "pearson_r_squared_raw": metrics.pearson_r_squared,
    }

def completed_split_metrics(
    results: CompletedFoldResults,
) -> pd.DataFrame:
    """Return train/validation metrics with explicit R² definitions."""

    rows: list[dict[str, Any]] = []
    for split in _SPLIT_ORDER:
        frame = results.predictions.loc[
            results.predictions["split"] == split
        ]
        if frame.empty:
            raise ValueError(f"Completed result has no {split!r} predictions")
        rows.append(
            {
                "split": split,
                **_regression_metrics(
                    frame["target_raw"],
                    frame["prediction_raw"],
                ),
            }
        )
    return pd.DataFrame(rows)

def _parity_limits(
    predictions: pd.DataFrame,
    limits: tuple[float, float] | None,
) -> tuple[float, float]:
    if limits is not None:
        lower, upper = (float(limits[0]), float(limits[1]))
        if not np.isfinite((lower, upper)).all() or lower >= upper:
            raise ValueError("limits must contain finite increasing values")
        return lower, upper
    values = predictions[["target_raw", "prediction_raw"]].to_numpy(
        dtype=float
    )
    if not np.isfinite(values).all():
        raise ValueError("Parity plot values must be finite")
    lower = float(values.min())
    upper = float(values.max())
    span = max(upper - lower, 1.0)
    padding = 0.03 * span
    return lower - padding, upper + padding

def plot_train_validation_yy(
    results: CompletedFoldResults,
    *,
    run_name: str | None = None,
    limits: tuple[float, float] | None = None,
    target_label: str = "pIC50",
    figsize: tuple[float, float] = (8.0, 8.0),
) -> tuple[Any, pd.DataFrame]:
    """Plot the frozen-notebook train/validation Y–Y visual contract.

    The horizontal axis is predicted and the vertical axis is experimental.
    Units are supplied through target_label. Regression R² and squared Pearson correlation are
    returned separately in the metric table.
    """

    metrics = completed_split_metrics(results)
    lower, upper = _parity_limits(results.predictions, limits)
    figure, axis = plt.subplots(figsize=figsize)
    style = {
        "train": {
            "color": "steelblue",
            "alpha": 0.50,
            "label": "Train",
        },
        "validation": {
            "color": "red",
            "alpha": 0.55,
            "label": "Valid",
        },
    }
    for split in _SPLIT_ORDER:
        frame = results.predictions.loc[
            results.predictions["split"] == split
        ]
        row = metrics.loc[metrics["split"] == split].iloc[0]
        axis.scatter(
            frame["prediction_raw"],
            frame["target_raw"],
            alpha=style[split]["alpha"],
            s=35,
            color=style[split]["color"],
            label=(
                f"{style[split]['label']}: "
                f"n = {int(row['n'])}, R² = {float(row['r2_score_raw']):.3f}"
            ),
        )
    axis.plot(
        [lower, upper],
        [lower, upper],
        color="black",
        linestyle="--",
        linewidth=1.5,
    )
    axis.set_xlabel(f"Pred. {target_label}")
    axis.set_ylabel(f"Exptl. {target_label}")
    title = f"Predicted vs. Experimental {target_label} Values"
    if run_name:
        title += f": {run_name}"
    title += (
        f" (fold {int(results.training_config['validation_fold'])})"
    )
    axis.set_title(title)
    axis.set_xlim(lower, upper)
    axis.set_ylim(lower, upper)
    axis.set_aspect("equal", adjustable="box")
    axis.legend(fontsize=12, loc="lower right")
    axis.grid(True, alpha=0.30)
    figure.tight_layout()
    return figure, metrics

def _smoothed(values: pd.Series, window: int) -> pd.Series:
    if isinstance(window, bool) or not isinstance(window, int) or window < 1:
        raise ValueError("smooth_window must be a positive integer")
    return values.rolling(
        window=window,
        center=True,
        min_periods=1,
    ).mean()

def plot_prediction_loss_curves(
    results: CompletedFoldResults,
    *,
    smooth_window: int = 7,
) -> Any:
    """Plot comparable train/validation prediction MSE in scaled space."""

    history = results.history
    required = {
        "epoch",
        "optimization_prediction_mse_scaled",
        "validation_mse_scaled",
    }
    missing = sorted(required.difference(history.columns))
    if missing:
        raise ValueError(f"History lacks prediction-loss columns: {missing}")
    figure, axis = plt.subplots(figsize=(9.0, 5.4))
    epoch = history["epoch"].to_numpy(dtype=float)
    specifications = (
        (
            "optimization_prediction_mse_scaled",
            "train prediction MSE (scaled)",
            "tab:blue",
        ),
        (
            "validation_mse_scaled",
            "validation prediction MSE (scaled)",
            "tab:orange",
        ),
    )
    for column, label, color in specifications:
        values = pd.to_numeric(history[column], errors="raise")
        if not np.isfinite(values).all():
            raise ValueError(f"History column contains non-finite values: {column}")
        axis.plot(epoch, values, color=color, alpha=0.22, linewidth=1)
        axis.plot(
            epoch,
            _smoothed(values, smooth_window),
            color=color,
            linewidth=2,
            label=label,
        )
    best = history.loc[history["is_best"].astype(bool), "epoch"]
    if len(best):
        axis.axvline(
            float(best.iloc[-1]),
            color="0.35",
            linestyle=":",
            linewidth=1.3,
            label="selected checkpoint epoch",
        )
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Prediction MSE (scaled target space)")
    axis.set_title(
        f"Fold {int(results.training_config['validation_fold']) + 1} "
        "Prediction Loss"
    )
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    return figure

def _prediction_row(
    results: CompletedFoldResults,
    source_row_index: int,
) -> pd.Series:
    rows = results.predictions.loc[
        results.predictions["source_row_index"] == int(source_row_index)
    ]
    if len(rows) != 1:
        raise ValueError(
            "Expected exactly one prediction row for source_row_index="
            f"{source_row_index}; observed {len(rows)}"
        )
    return rows.iloc[0]

def render_atom_contribution_record(
    results: CompletedFoldResults,
    *,
    source_row_index: int,
    symmetric_limit: float,
    cmap_name: str = "bwr",
    width: int = 620,
    height: int = 430,
    draw_atom_indices: bool = False,
    atom_highlight_radius: float = 0.28,
    bond_highlight_thinness: float = 1.35,
    atom_highlight_alpha: float = 0.62,
    bond_highlight_alpha: float | None = 0.45,
) -> dict[str, Any]:
    """Render raw atom contributions without changing their numeric values."""

    limit = float(symmetric_limit)
    if not np.isfinite(limit) or limit <= 0.0:
        raise ValueError("symmetric_limit must be positive and finite")
    prediction = _prediction_row(results, source_row_index)
    molecule = Chem.MolFromSmiles(str(prediction["smiles"]))
    if molecule is None:
        raise ValueError(f"RDKit could not parse SMILES: {prediction['smiles']!r}")
    atoms = results.atom_contributions.loc[
        results.atom_contributions["source_row_index"]
        == int(source_row_index)
    ].sort_values("smiles_atom_index", kind="stable")
    atom_indices = atoms["smiles_atom_index"].to_numpy(dtype=int)
    expected_indices = np.arange(molecule.GetNumAtoms(), dtype=int)
    if not np.array_equal(atom_indices, expected_indices):
        raise ValueError(
            "Atom contribution rows do not exactly cover the SMILES atom order"
        )
    atomic_numbers = atoms["atomic_number"].to_numpy(dtype=int)
    expected_atomic_numbers = np.asarray(
        [atom.GetAtomicNum() for atom in molecule.GetAtoms()],
        dtype=int,
    )
    if not np.array_equal(atomic_numbers, expected_atomic_numbers):
        raise ValueError("Atom contribution atomic numbers differ from SMILES")
    raw = atoms["atom_contribution_raw"].to_numpy(dtype=float)
    if not np.isfinite(raw).all():
        raise ValueError("Raw atom contributions must be finite")
    rendered = render_contribution_image(
        mol=molecule,
        contrib=raw,
        y_pred=float(prediction["prediction_raw"]),
        y_true=float(prediction["target_raw"]),
        vmin=-limit,
        vmax=limit,
        cmap_name=cmap_name,
        width=int(width),
        height=int(height),
        include_bond_colors=True,
        draw_atom_indices=bool(draw_atom_indices),
        show_rdkit_legend=False,
        atom_highlight_radius=float(atom_highlight_radius),
        bond_highlight_thinness=float(bond_highlight_thinness),
        atom_highlight_alpha=float(atom_highlight_alpha),
        bond_highlight_alpha=bond_highlight_alpha,
    )
    return {
        "source_row_index": int(source_row_index),
        "smiles": str(prediction["smiles"]),
        "split": str(prediction["split"]),
        "molecule": molecule,
        "raw_contribution": raw.copy(),
        "raw_contribution_sum": float(raw.sum()),
        "prediction_raw": float(prediction["prediction_raw"]),
        "target_raw": float(prediction["target_raw"]),
        "error_raw": float(
            prediction["prediction_raw"] - prediction["target_raw"]
        ),
        "affine_baseline_raw": float(prediction["affine_baseline_raw"]),
        "additivity_residual_raw": float(
            prediction["additivity_residual_raw"]
        ),
        "contribution_display_mode": "raw",
        "symmetric_limit": limit,
        **rendered,
    }

def _atom_rows(
    record: Mapping[str, Any],
    *,
    side: str,
) -> list[dict[str, Any]]:
    rows = []
    molecule = record["molecule"]
    for atom_index, value in enumerate(record["raw_contribution"]):
        rows.append(
            {
                "side": side,
                "source_row_index": int(record["source_row_index"]),
                "atom_index": int(atom_index),
                "element": molecule.GetAtomWithIdx(atom_index).GetSymbol(),
                "raw_contribution": float(value),
                "absolute_raw_contribution": float(abs(value)),
            }
        )
    return rows

def plot_atom_contribution_grid(
    results: CompletedFoldResults,
    *,
    source_rows: Sequence[int],
    symmetric_limit: float,
    labels: Mapping[int, str] | None = None,
    ncols: int = 2,
    contribution_label: str = "Raw atom contribution (pIC50 scale)",
    cmap_name: str = "bwr",
    draw_atom_indices: bool = False,
) -> tuple[Any, pd.DataFrame, pd.DataFrame]:
    """Display several molecules on one checked raw-contribution scale."""

    rows = tuple(int(value) for value in source_rows)
    if not rows or len(set(rows)) != len(rows):
        raise ValueError("source_rows must contain unique row identifiers")
    if isinstance(ncols, bool) or not isinstance(ncols, int) or ncols < 1:
        raise ValueError("ncols must be a positive integer")
    records = [
        render_atom_contribution_record(
            results,
            source_row_index=source_row,
            symmetric_limit=symmetric_limit,
            cmap_name=cmap_name,
            draw_atom_indices=draw_atom_indices,
        )
        for source_row in rows
    ]
    nrows = int(np.ceil(len(records) / ncols))
    figure = plt.figure(
        figsize=(5.2 * ncols + 0.65, 4.2 * nrows),
        constrained_layout=True,
    )
    grid = figure.add_gridspec(
        nrows,
        ncols + 1,
        width_ratios=[1.0] * ncols + [0.045],
    )
    axes = [
        figure.add_subplot(grid[row_index, column_index])
        for row_index in range(nrows)
        for column_index in range(ncols)
    ]
    for axis, record in zip(axes, records):
        axis.imshow(
            decode_and_trim_png(
                record["image"],
                white_threshold=0.996,
                pad_px=1,
            )
        )
        label = (
            labels.get(int(record["source_row_index"]), "")
            if labels is not None
            else ""
        )
        label_prefix = f"{label} | " if label else ""
        axis.set_title(
            f"{label_prefix}row {record['source_row_index']}\n"
            f"Pred: {record['prediction_raw']:.2f} | "
            f"True: {record['target_raw']:.2f} | "
            f"Err: {record['error_raw']:+.2f}",
            fontsize=10,
            pad=2,
        )
        axis.axis("off")
    for axis in axes[len(records) :]:
        axis.axis("off")
    color_axis = figure.add_subplot(grid[:, -1])
    colorbar = figure.colorbar(
        plt.cm.ScalarMappable(
            norm=TwoSlopeNorm(
                vmin=-float(symmetric_limit),
                vcenter=0.0,
                vmax=float(symmetric_limit),
            ),
            cmap=cmap_name,
        ),
        cax=color_axis,
        orientation="vertical",
        extend="both",
    )
    colorbar.set_label(contribution_label, fontsize=10)
    molecule_table = pd.DataFrame(
        [
            {
                "source_row_index": record["source_row_index"],
                "profile_label": (
                    labels.get(int(record["source_row_index"]), "")
                    if labels is not None
                    else ""
                ),
                "split": record["split"],
                "smiles": record["smiles"],
                "target_raw": record["target_raw"],
                "prediction_raw": record["prediction_raw"],
                "error_raw": record["error_raw"],
                "affine_baseline_raw": record["affine_baseline_raw"],
                "raw_contribution_sum": record["raw_contribution_sum"],
                "additivity_residual_raw": record[
                    "additivity_residual_raw"
                ],
                "color_saturated_atom_count": record[
                    "color_saturated_atom_count"
                ],
                "color_saturated_atom_fraction": record[
                    "color_saturated_atom_fraction"
                ],
            }
            for record in records
        ]
    )
    atom_table = pd.DataFrame(
        [
            atom
            for record in records
            for atom in _atom_rows(record, side="molecule")
        ]
    )
    return figure, molecule_table, atom_table


def plot_reconstruction_yy(
    target: Sequence[float], prediction: Sequence[float], *, unit_label: str, title: str,
) -> tuple[Any, pd.DataFrame]:
    """Use the project parity style for an explicitly in-sample surrogate fit."""
    scores = _regression_metrics(target, prediction)
    values = pd.DataFrame({"target_raw": target, "prediction_raw": prediction})
    lower, upper = _parity_limits(values, None)
    figure, axis = plt.subplots(figsize=(8.0, 8.0))
    axis.scatter(values["prediction_raw"], values["target_raw"], s=35,
                 color="steelblue", alpha=0.50,
                 label=f"In-sample: n = {scores['n']}, R² = {scores['r2_score_raw']:.3f}")
    axis.plot([lower, upper], [lower, upper], color="black", linestyle="--", linewidth=1.5)
    axis.set(xlim=(lower, upper), ylim=(lower, upper),
             xlabel=f"Fitted {unit_label}", ylabel=f"Teacher {unit_label}", title=title)
    axis.set_aspect("equal", adjustable="box")
    axis.legend(fontsize=12, loc="lower right")
    axis.grid(True, alpha=0.30)
    figure.tight_layout()
    return figure, pd.DataFrame([scores])


def save_publication_figure(figure: Any, path: Path | str) -> Path:
    """Save the same white-background 300 dpi figure displayed in the notebook."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white", transparent=False)
    return output_path
