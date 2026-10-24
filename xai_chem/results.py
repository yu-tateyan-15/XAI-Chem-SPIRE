"""Results for notebook-based atom-contribution experiments."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import numpy as np
import pandas as pd
from .training import ATOM_CONTRIBUTION_COLUMNS
from .training import HISTORY_COLUMNS
from .training import PREDICTION_COLUMNS
from .training import RESULT_RECORDS_KEY
from .config import TrainingConfig


_RESULT_FILES = {
    "training_config": "training_config.json",
    "provenance": "provenance.json",
    "history": "history.csv",
    "checkpoint": "best_checkpoint.pt",
    "predictions": "predictions.csv",
    "atom_contributions": "atom_contributions.csv",
}


_RESULT_RECORD_FILENAMES = (
    "result_manifest.json",
    "result_metadata.json",
)


_CSV_COLUMNS = {
    "history": tuple(HISTORY_COLUMNS),
    "predictions": tuple(PREDICTION_COLUMNS),
    "atom_contributions": tuple(ATOM_CONTRIBUTION_COLUMNS),
}


@dataclass(frozen=True)
class CompletedFoldResults:
    root: Path
    metadata: dict[str, Any]
    training_config: dict[str, Any]
    provenance: dict[str, Any]
    history: pd.DataFrame
    predictions: pd.DataFrame
    atom_contributions: pd.DataFrame
    checkpoint: Path


def assert_completed_result_matches_config(
    results: CompletedFoldResults,
    requested: TrainingConfig,
) -> None:
    """Compare only the scientific settings needed for matched reuse."""

    if not isinstance(results, CompletedFoldResults):
        raise TypeError("results must be CompletedFoldResults")
    if not isinstance(requested, TrainingConfig):
        raise TypeError("requested must be TrainingConfig")

    scientific_fields = (
        "architecture_contract", "atom_bias", "backbone_name",
        "backbone_normalization", "backbone_residual", "bond_feature_dim",
        "checkpoint_selection", "dropout", "early_stopping_patience",
        "epochs", "fragment_consistency_weight", "fragment_frequency_weight",
        "fragment_min_count", "fragment_min_frequency",
        "fragment_min_heavy_atoms", "fragment_positive_weight", "heads",
        "hidden_dim", "learning_rate", "mmp_core_weight",
        "loss_gradient_scaling",
        "mmp_difference_weight", "mmp_huber_beta", "mmp_pair_batch_size",
        "mmp_weight", "num_layers", "optimization_mode",
        "output_head_init_std", "prediction_huber_delta",
        "prediction_loss_mode", "readout_hidden_dim", "readout_num_layers",
        "scaling_mode", "seed", "supervised_batch_size", "training_scope",
        "use_edge_features", "validation_fold", "weight_decay",
    )
    expected_config = requested.as_dict()
    observed_config = dict(results.training_config)
    # Historical results omitted the effective default for this model field.
    observed_config.setdefault("backbone_residual", False)
    expected = {field: expected_config.get(field) for field in scientific_fields}
    observed = {field: observed_config.get(field) for field in scientific_fields}
    keys = list(scientific_fields)
    differences = [
        key
        for key in keys
        if observed.get(key, object()) != expected.get(key, object())
    ]
    if differences:
        raise ValueError(
            "Completed result differs in matched scientific settings: "
            f"fields={differences}"
        )


def _require_exact_columns(
    frame: pd.DataFrame,
    columns: Iterable[str],
    *,
    label: str,
) -> None:
    expected = tuple(columns)
    actual = tuple(frame.columns)
    if actual != expected:
        raise ValueError(
            f"{label} columns differ from the strict schema: "
            f"expected={expected}, actual={actual}"
        )


def _load_strict_json_object(path: Path, *, label: str) -> dict[str, Any]:
    def reject_duplicate_keys(
        pairs: list[tuple[str, Any]],
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    def reject_non_finite_constant(value: str) -> None:
        raise ValueError(
            f"{label} contains non-finite JSON constant {value!r}"
        )

    def require_finite_numbers(value: Any, *, location: str) -> None:
        if isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError(
                    f"{label} contains non-finite JSON number at {location}"
                )
            return
        if isinstance(value, dict):
            for key, child in value.items():
                require_finite_numbers(
                    child,
                    location=f"{location}.{key}",
                )
            return
        if isinstance(value, list):
            for index, child in enumerate(value):
                require_finite_numbers(
                    child,
                    location=f"{location}[{index}]",
                )

    payload = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicate_keys,
        parse_constant=reject_non_finite_constant,
    )
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    require_finite_numbers(payload, location="$")
    return payload


def load_completed_fold_results(
    result_dir: Path | str,
) -> CompletedFoldResults:
    """Load one explicitly declared completed fold result."""

    root = Path(result_dir).expanduser().resolve()
    (root / "COMPLETE").read_text(encoding="ascii")
    # Production optimization and training outputs use two schema-v2 names.
    # Exactly one file remains the unambiguous results record.
    record_paths = [
        root / filename
        for filename in _RESULT_RECORD_FILENAMES
        if (root / filename).is_file()
    ]
    if len(record_paths) != 1:
        raise ValueError(
            "Completed result must contain exactly one result record file"
        )
    metadata_path = record_paths[0]
    metadata = _load_strict_json_object(
        metadata_path,
        label=metadata_path.name,
    )
    records = metadata[RESULT_RECORDS_KEY]

    for key, expected_filename in _RESULT_FILES.items():
        record = records[key]
        if record["filename"] != expected_filename:
            raise ValueError(f"Result filename is invalid: {key}")
        if key in _CSV_COLUMNS:
            observed_columns = tuple(record["columns"])
            expected_columns = _CSV_COLUMNS[key]
            if observed_columns != expected_columns:
                raise ValueError(f"Result CSV schema mismatch: {key}")

    predictions = pd.read_csv(root / _RESULT_FILES["predictions"])
    atoms = pd.read_csv(root / _RESULT_FILES["atom_contributions"])
    _require_exact_columns(predictions, PREDICTION_COLUMNS, label="predictions")
    _require_exact_columns(atoms, ATOM_CONTRIBUTION_COLUMNS, label="atoms")
    training_config = _load_strict_json_object(
        root / _RESULT_FILES["training_config"],
        label=_RESULT_FILES["training_config"],
    )
    history = pd.read_csv(root / _RESULT_FILES["history"])
    return CompletedFoldResults(
        root=root,
        metadata=metadata,
        training_config=training_config,
        provenance=_load_strict_json_object(
            root / _RESULT_FILES["provenance"],
            label=_RESULT_FILES["provenance"],
        ),
        history=history,
        predictions=predictions,
        atom_contributions=atoms,
        checkpoint=root / _RESULT_FILES["checkpoint"],
    )


def _finite_numeric(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    for column in columns:
        values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{label}.{column} contains non-finite values")


def _strict_integer_array(series: pd.Series, *, label: str) -> np.ndarray:
    if series.map(lambda value: isinstance(value, (bool, np.bool_))).any():
        raise ValueError(f"{label} must not contain booleans")
    values = pd.to_numeric(series, errors="raise").to_numpy(dtype=float)
    if not np.isfinite(values).all() or not np.equal(values, np.rint(values)).all():
        raise ValueError(f"{label} must contain finite exact integers")
    return values.astype(np.int64)


def check_additivity(
    predictions: pd.DataFrame,
    atom_contributions: pd.DataFrame,
    *,
    absolute_tolerance: float = 1.0e-6,
    relative_tolerance: float = 1.0e-6,
) -> pd.DataFrame:
    """Recompute scaled and affine-raw atom-sum identities per molecule.

    The checked identities are

    ``prediction_scaled = sum(atom_contribution_scaled)`` and
    ``prediction_raw = affine_baseline_raw + sum(atom_contribution_raw)``.
    """

    _require_exact_columns(predictions, PREDICTION_COLUMNS, label="predictions")
    _require_exact_columns(
        atom_contributions,
        ATOM_CONTRIBUTION_COLUMNS,
        label="atom_contributions",
    )
    if absolute_tolerance < 0.0 or relative_tolerance < 0.0:
        raise ValueError("Additivity tolerances cannot be negative")
    if predictions["source_row_index"].duplicated().any():
        raise ValueError("predictions has duplicate source_row_index values")
    if atom_contributions.duplicated(
        ["source_row_index", "graph_atom_index"]
    ).any():
        raise ValueError("atom_contributions has duplicate molecule-atom keys")
    _finite_numeric(
        predictions,
        (
            "prediction_scaled",
            "additivity_residual_scaled",
            "prediction_raw",
            "affine_baseline_raw",
            "atom_sum_raw",
            "additivity_residual_raw",
        ),
        "predictions",
    )
    _finite_numeric(
        atom_contributions,
        (
            "graph_atom_index",
            "smiles_atom_index",
            "atomic_number",
            "atom_contribution_scaled",
            "atom_contribution_raw",
        ),
        "atom_contributions",
    )
    prediction_source_rows = _strict_integer_array(
        predictions["source_row_index"],
        label="predictions.source_row_index",
    )
    atom_source_rows = _strict_integer_array(
        atom_contributions["source_row_index"],
        label="atom_contributions.source_row_index",
    )
    _strict_integer_array(
        predictions["n_atoms"],
        label="predictions.n_atoms",
    )
    _strict_integer_array(
        atom_contributions["graph_atom_index"],
        label="atom_contributions.graph_atom_index",
    )
    _strict_integer_array(
        atom_contributions["smiles_atom_index"],
        label="atom_contributions.smiles_atom_index",
    )
    atomic_numbers = _strict_integer_array(
        atom_contributions["atomic_number"],
        label="atom_contributions.atomic_number",
    )
    if np.any(atomic_numbers <= 0):
        raise ValueError("atom_contributions.atomic_number must be positive")

    prediction_by_row = predictions.set_index("source_row_index", drop=False)
    atom_rows = set(atom_source_rows)
    prediction_rows = set(prediction_source_rows)
    if atom_rows != prediction_rows:
        raise ValueError(
            "Prediction/atom molecule sets differ: "
            f"predictions_only={sorted(prediction_rows - atom_rows)[:10]}, "
            f"atoms_only={sorted(atom_rows - prediction_rows)[:10]}"
        )

    identity_columns = ("molecule_id", "sample_id", "smiles", "split")
    rows: list[dict[str, Any]] = []
    for source_row_index, group in atom_contributions.groupby(
        "source_row_index", sort=True
    ):
        source_row_index = int(source_row_index)
        prediction = prediction_by_row.loc[source_row_index]
        n_atoms = int(prediction["n_atoms"])
        indices = group["graph_atom_index"].astype(int).to_numpy()
        if len(group) != n_atoms or not np.array_equal(
            np.sort(indices), np.arange(n_atoms)
        ):
            raise ValueError(
                f"Atom rows are not a 0..n_atoms-1 permutation: "
                f"source_row_index={source_row_index}"
            )
        smiles_indices = group["smiles_atom_index"].astype(int).to_numpy()
        if not np.array_equal(np.sort(smiles_indices), np.arange(n_atoms)):
            raise ValueError(
                f"SMILES atom rows are not a permutation: "
                f"source_row_index={source_row_index}"
            )
        for column in identity_columns:
            values = group[column].astype(str).unique()
            if len(values) != 1 or values[0] != str(prediction[column]):
                raise ValueError(
                    f"Molecule identity mismatch at {source_row_index}: {column}"
                )

        scaled_sum = float(group["atom_contribution_scaled"].sum())
        raw_sum = float(group["atom_contribution_raw"].sum())
        scaled_residual = float(prediction["prediction_scaled"]) - scaled_sum
        raw_residual = float(prediction["prediction_raw"]) - (
            float(prediction["affine_baseline_raw"]) + raw_sum
        )
        scaled_limit = absolute_tolerance + relative_tolerance * abs(
            float(prediction["prediction_scaled"])
        )
        raw_scale = (
            abs(float(prediction["prediction_raw"]))
            + abs(float(prediction["affine_baseline_raw"]))
            + float(group["atom_contribution_raw"].abs().sum())
        )
        raw_limit = absolute_tolerance + relative_tolerance * raw_scale
        if abs(
            scaled_residual
            - float(prediction["additivity_residual_scaled"])
        ) > scaled_limit:
            raise ValueError(
                f"Declared scaled residual mismatch at {source_row_index}"
            )
        # Float32 affine scaling and cancellation round the exported atom sum.
        # Use the same operand-scaled budget as the raw additivity identity.
        if abs(raw_sum - float(prediction["atom_sum_raw"])) > raw_limit:
            raise ValueError(f"Declared raw atom sum mismatch at {source_row_index}")
        if abs(
            raw_residual
            - float(prediction["additivity_residual_raw"])
        ) > raw_limit:
            raise ValueError(f"Declared raw residual mismatch at {source_row_index}")
        rows.append(
            {
                "source_row_index": source_row_index,
                "split": prediction["split"],
                "n_atoms": n_atoms,
                "atom_sum_scaled_recomputed": scaled_sum,
                "additivity_residual_scaled_recomputed": scaled_residual,
                "atom_sum_raw_recomputed": raw_sum,
                "additivity_residual_raw_recomputed": raw_residual,
                "scaled_tolerance": scaled_limit,
                "raw_tolerance": raw_limit,
                "scaled_pass": abs(scaled_residual) <= scaled_limit,
                "raw_pass": abs(raw_residual) <= raw_limit,
            }
        )
    check = pd.DataFrame(rows)
    if not check["scaled_pass"].all() or not check["raw_pass"].all():
        raise ValueError("Atom-sum additivity check failed")
    return check
