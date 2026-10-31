"""Reconstruction for notebook-based atom-contribution experiments."""

from __future__ import annotations

from collections.abc import Sequence
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path
import pandas as pd
from rdkit import Chem
from .io import exact_integer
from .io import exact_integers
from dataclasses import dataclass
from typing import Sequence
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import json
from typing import Mapping, Sequence


def raw_regression_metrics(
    observed: Sequence[float] | np.ndarray,
    predicted: Sequence[float] | np.ndarray,
) -> dict[str, float | int]:
    """Return sample count, regression R², RMSE, and MAE on the raw target scale."""

    observed_array = np.asarray(observed, dtype=float)
    predicted_array = np.asarray(predicted, dtype=float)
    return {
        "n": int(len(observed_array)),
        "r2": (
            float(r2_score(observed_array, predicted_array))
            if len(observed_array) > 1
            else np.nan
        ),
        "rmse": float(np.sqrt(mean_squared_error(observed_array, predicted_array))),
        "mae": float(mean_absolute_error(observed_array, predicted_array)),
    }


def align_descriptors_to_dataset(
    descriptors: pd.DataFrame,
    dataset_path: Path,
) -> pd.DataFrame:
    """Map descriptor molecules to current graph IDs by identical SMILES."""

    dataset = pd.read_csv(dataset_path, usecols=["index", "cid", "Smiles"])
    molecule_map = dataset.rename(
        columns={
            "index": "dataset_index",
            "cid": "dataset_molecule_id",
            "Smiles": "smiles",
        }
    )
    aligned = descriptors.merge(
        molecule_map,
        on="smiles",
        how="inner",
        validate="many_to_one",
    )
    aligned["graph_id"] = aligned["dataset_index"].astype(int) - 1
    aligned = aligned.rename(columns={"molecule_id": "descriptor_molecule_id"})
    aligned["molecule_id"] = aligned["dataset_molecule_id"].astype(str)

    # RDKit parses the exact dataset SMILES in the same atom order used by the
    # graph builder; descriptor atom symbols must follow that order.
    for smiles, rows in aligned.groupby("smiles", sort=False):
        molecule = Chem.MolFromSmiles(smiles)
        observed = rows.sort_values("atom_idx")["atom_symbol"].astype(str).tolist()
        expected = [atom.GetSymbol() for atom in molecule.GetAtoms()]
        if observed != expected:
            raise ValueError(f"Descriptor atom order disagrees with dataset SMILES: {smiles}")
    return aligned


def _atom_scope_datasets_exact_index_series(
    values: pd.Series,
    *,
    name: str,
    minimum: int,
) -> pd.Series:
    validated = exact_integers(values.tolist(), name=name, minimum=minimum)
    return pd.Series(validated, index=values.index, dtype=np.int64)


def load_teacher_atom_contributions(
    path: Path,
    *,
    teacher_fold: int | None = None,
) -> pd.DataFrame:
    """Load one teacher table from the caller's explicit completed run."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if teacher_fold is not None:
        teacher_fold = exact_integer(
            teacher_fold,
            name="teacher_fold",
            minimum=0,
        )
    columns = pd.read_csv(path, nrows=0).columns.tolist()
    legacy_required = {"graph_id", "atom_idx", "atom_contribution"}
    modeling_required = {
        "source_row_index",
        "graph_atom_index",
        "smiles_atom_index",
        "molecule_id",
        "smiles",
        "split",
        "atom_contribution_raw",
    }
    if legacy_required.issubset(columns):
        schema = "graph_atom_contribution_v1"
        usecols = ["graph_id", "atom_idx", "atom_contribution"]
    elif modeling_required.issubset(columns):
        schema = "modeling_atom_contributions_v1"
        usecols = sorted(modeling_required)
    else:
        raise ValueError(
            "Teacher table does not use a supported atom-contribution schema."
        )

    if "fold" in columns and schema == "graph_atom_contribution_v1":
        if teacher_fold is None:
            raise ValueError("teacher_fold must be explicit for a teacher table containing fold rows.")
        usecols.append("fold")
    elif teacher_fold is not None:
        raise ValueError("teacher_fold was supplied, but the teacher table has no fold column.")

    teacher = pd.read_csv(path, usecols=usecols)
    if schema == "modeling_atom_contributions_v1":
        teacher = teacher.rename(
            columns={
                "source_row_index": "graph_id",
                "graph_atom_index": "atom_idx",
                "atom_contribution_raw": "atom_contribution",
                "molecule_id": "teacher_molecule_id",
                "smiles": "teacher_smiles",
                "split": "teacher_split",
            }
        )
        teacher["smiles_atom_index"] = _atom_scope_datasets_exact_index_series(
            teacher["smiles_atom_index"],
            name="smiles_atom_index",
            minimum=0,
        )
    if "fold" in teacher.columns:
        fold_values = _atom_scope_datasets_exact_index_series(
            teacher["fold"],
            name="fold",
            minimum=0,
        )
        teacher = teacher.loc[fold_values.eq(teacher_fold)].drop(
            columns="fold"
        )
        if teacher.empty:
            raise ValueError(f"Teacher table has no rows for fold={teacher_fold}.")

    for column in ("graph_id", "atom_idx"):
        teacher[column] = _atom_scope_datasets_exact_index_series(
            teacher[column],
            name=column,
            minimum=0,
        )
    contributions = pd.to_numeric(teacher["atom_contribution"], errors="raise")
    if not np.isfinite(contributions).all():
        raise ValueError("Teacher atom contributions must be finite.")
    teacher["atom_contribution"] = contributions.astype(float)
    if teacher.duplicated(["graph_id", "atom_idx"], keep=False).any():
        raise ValueError("Teacher table contains duplicate graph_id/atom_idx rows in the selected source.")

    teacher["teacher_source_path"] = str(path.resolve())
    teacher["teacher_fold"] = pd.Series(
        [teacher_fold] * len(teacher), index=teacher.index, dtype="Int64"
    )
    if schema == "modeling_atom_contributions_v1":
        for graph_id, group in teacher.groupby("graph_id", sort=False):
            expected_indices = np.arange(len(group), dtype=np.int64)
            if not np.array_equal(np.sort(group["atom_idx"].to_numpy()), expected_indices):
                raise ValueError(
                    f"Teacher graph_atom_index is not a complete permutation for graph_id={graph_id}."
                )
            if not np.array_equal(
                np.sort(group["smiles_atom_index"].to_numpy()), expected_indices
            ):
                raise ValueError(
                    f"Teacher smiles_atom_index is not a complete permutation for graph_id={graph_id}."
                )
    teacher["teacher_schema"] = schema
    return teacher.reset_index(drop=True)


@dataclass(frozen=True)
class PairFitResult:
    pair: pd.Series
    atoms: pd.DataFrame
    endpoints: pd.DataFrame
    metrics: pd.DataFrame
    coefficients: pd.DataFrame
    concept_totals: pd.DataFrame
    scaling: pd.DataFrame


def _safe_correlation(observed: np.ndarray, predicted: np.ndarray, kind: str) -> float:
    if len(observed) < 2 or np.std(observed) == 0 or np.std(predicted) == 0:
        return np.nan
    if kind == "pearson":
        return float(pearsonr(observed, predicted).statistic)
    return float(spearmanr(observed, predicted).statistic)


def _metric_row(observed: np.ndarray, predicted: np.ndarray, level: str) -> dict[str, object]:
    return {
        "level": level,
        **raw_regression_metrics(observed, predicted),
        "pearson_r": _safe_correlation(observed, predicted, "pearson"),
        "spearman_r": _safe_correlation(observed, predicted, "spearman"),
    }


def fit_pair(
    pair: pd.Series,
    atom_rows: pd.DataFrame,
    *,
    alpha: float = 1.0,
    features: Sequence[str],
) -> PairFitResult:
    """Fit pair-specific Ridge with explicitly selected descriptor effects."""

    features = tuple(features)
    if "delta_n_diff_atoms" in features:
        raise ValueError("delta_n_diff_atoms is not a permitted regression descriptor.")

    molecule_a = str(pair["molecule_a"])
    molecule_b = str(pair["molecule_b"])
    data = atom_rows[
        atom_rows["context_id"].eq(str(pair["context_id"]))
        & atom_rows["molecule_id"].isin([molecule_a, molecule_b])
    ].copy()
    data = data.dropna(subset=[*features, "atom_contribution"])
    if set(data["molecule_id"].astype(str)) != {molecule_a, molecule_b}:
        raise ValueError(f"Both endpoints are required for pair_id={int(pair['pair_id'])}.")
    data["endpoint"] = np.where(data["molecule_id"].eq(molecule_a), "A", "B")

    x = data[list(features)].to_numpy(dtype=float)
    y = data["atom_contribution"].to_numpy(dtype=float)
    x_scaler = StandardScaler().fit(x)
    x_scaled = x_scaler.transform(x)
    y_mean = float(np.mean(y))
    y_scale = float(np.std(y))
    if y_scale < 1.0e-12:
        y_scale = 1.0
    if alpha < 0.0:
        raise ValueError("alpha must be non-negative.")
    y_scaled = (y - y_mean) / y_scale
    # Fit original atom rows in pair-standardized X/y space; restore raw AC below.
    model = Ridge(alpha=float(alpha), fit_intercept=False)
    model.fit(x_scaled, y_scaled)
    standardized_coefficients = np.ravel(model.coef_)
    effects_scaled = x_scaled * standardized_coefficients
    basis_counts = np.ones(len(features), dtype=int)
    coefficient_l2_scaled = np.abs(standardized_coefficients)

    effects = effects_scaled * y_scale
    prediction = y_mean + effects.sum(axis=1)
    data["prediction"] = prediction
    data["residual"] = y - prediction
    data["atom_label"] = data["atom_symbol"].astype(str) + data["atom_idx"].astype(str)
    for index, feature in enumerate(features):
        data[f"z__{feature}"] = x_scaled[:, index]
        data[f"effect__{feature}"] = effects[:, index]
    if not np.allclose(data["prediction"], y_mean + data[[f"effect__{f}" for f in features]].sum(axis=1)):
        raise AssertionError("Atom predictions do not equal bias plus exact descriptor effects.")

    endpoint_rows: list[dict[str, object]] = []
    for endpoint, group in data.groupby("endpoint", sort=True):
        endpoint_rows.append(
            {
                "endpoint": endpoint,
                "molecule_id": str(group["molecule_id"].iloc[0]),
                "n_diff_atoms": len(group),
                "T_teacher": float(group["atom_contribution"].sum()),
                "T_prediction": float(group["prediction"].sum()),
                "T_residual": float(group["atom_contribution"].sum() - group["prediction"].sum()),
            }
        )
    endpoints = pd.DataFrame.from_records(endpoint_rows)

    observed = data["atom_contribution"].to_numpy(float)
    predicted = data["prediction"].to_numpy(float)
    metric_rows = [_metric_row(observed, predicted, "A+B diff-region atoms")]
    for endpoint, group in data.groupby("endpoint", sort=True):
        metric_rows.append(
            _metric_row(
                group["atom_contribution"].to_numpy(float),
                group["prediction"].to_numpy(float),
                f"endpoint {endpoint} atoms",
            )
        )
    delta_teacher = float(
        endpoints.set_index("endpoint").loc["B", "T_teacher"]
        - endpoints.set_index("endpoint").loc["A", "T_teacher"]
    )
    delta_prediction = float(
        endpoints.set_index("endpoint").loc["B", "T_prediction"]
        - endpoints.set_index("endpoint").loc["A", "T_prediction"]
    )
    metric_rows.append(
        {
            "level": "B-A fragment sum",
            "n": 1,
            "r2": np.nan,
            "rmse": abs(delta_teacher - delta_prediction),
            "mae": abs(delta_teacher - delta_prediction),
            "pearson_r": np.nan,
            "spearman_r": np.nan,
            "observed_delta": delta_teacher,
            "predicted_delta": delta_prediction,
        }
    )
    metrics = pd.DataFrame.from_records(metric_rows)
    metrics.insert(0, "pair_id", int(pair["pair_id"]))
    metrics.insert(0, "context_id", str(pair["context_id"]))
    metrics["model_kind"] = "ridge"
    metrics["alpha"] = float(alpha)
    metrics["n_original_atoms"] = len(data)
    metrics["n_training_rows"] = len(y)

    coefficients = pd.DataFrame(
        {
            "descriptor": list(features),
            "model_kind": "ridge",
            "n_basis": basis_counts,
            "ridge_coefficient_scaled_y": standardized_coefficients,
            "effect_per_descriptor_sd": standardized_coefficients * y_scale,
            "raw_slope": standardized_coefficients * y_scale / x_scaler.scale_,
            "basis_coefficient_l2": coefficient_l2_scaled * y_scale,
            "effect_mean_abs": np.mean(np.abs(effects), axis=0),
            "effect_rms": np.sqrt(np.mean(np.square(effects), axis=0)),
        }
    )
    coefficients["abs_effect_per_descriptor_sd"] = coefficients["effect_per_descriptor_sd"].abs()

    concept_rows: list[dict[str, object]] = []
    for endpoint, group in data.groupby("endpoint", sort=True):
        concept_rows.append(
            {
                "endpoint": endpoint,
                "concept": "shared atom bias",
                "contribution": len(group) * y_mean,
            }
        )
        for feature in features:
            concept_rows.append(
                {
                    "endpoint": endpoint,
                    "concept": feature,
                    "contribution": float(group[f"effect__{feature}"].sum()),
                }
            )
    concept_totals = pd.DataFrame.from_records(concept_rows)
    pivot = concept_totals.pivot(index="concept", columns="endpoint", values="contribution")
    concept_totals = concept_totals.merge(
        (pivot["B"] - pivot["A"]).rename("B_minus_A").reset_index(),
        on="concept",
        how="left",
        validate="many_to_one",
    )
    if not np.isclose(concept_totals.drop_duplicates("concept")["B_minus_A"].sum(), delta_prediction):
        raise AssertionError("Concept differences do not reconstruct predicted B-A delta.")

    scaling = pd.DataFrame(
        {
            "descriptor": list(features),
            "pair_mean": x_scaler.mean_,
            "pair_sd": x_scaler.scale_,
        }
    )
    scaling["target_pair_mean"] = y_mean
    scaling["target_pair_sd"] = y_scale
    scaling["model_kind"] = "ridge"
    scaling["alpha"] = float(alpha)
    scaling["n_original_atoms"] = len(data)
    scaling["n_training_rows"] = len(y)
    return PairFitResult(
        pair=pair,
        atoms=data.reset_index(drop=True),
        endpoints=endpoints,
        metrics=metrics,
        coefficients=coefficients,
        concept_totals=concept_totals,
        scaling=scaling,
    )


PAIR_COLUMNS = [
    "pair_id",
    "document_id",
    "molecule_1_id",
    "molecule_2_id",
    "i_id",
    "j_id",
    "pIC50_i",
    "pIC50_j",
    "dy",
    "mmp_class",
    "cut_count",
    "diff_i_json",
    "diff_j_json",
    "fp_similarity",
    "diff_total_count",
    "core_fraction_min",
    "mmp_transform",
]


@dataclass(frozen=True)
class AllMmpFitResult:
    audit: pd.DataFrame
    pair_features: pd.DataFrame
    fit_metrics: pd.DataFrame
    atom_predictions: pd.DataFrame
    coefficients: pd.DataFrame
    concept_deltas: pd.DataFrame


def _parse_indices(value: object) -> tuple[int, ...]:
    if isinstance(value, (list, tuple, np.ndarray)):
        return exact_integers(value, name="atom_indices", minimum=0)
    if value is None:
        return ()
    parsed = json.loads(str(value))
    if not isinstance(parsed, list):
        raise ValueError("atom_indices JSON must contain an array")
    return exact_integers(parsed, name="atom_indices", minimum=0)


def _all_mmp_fidelity_exact_index_series(
    values: pd.Series,
    *,
    name: str,
    minimum: int,
) -> pd.Series:
    validated = exact_integers(values.tolist(), name=name, minimum=minimum)
    return pd.Series(validated, index=values.index, dtype=np.int64)


def _validated_atom_keys(atoms: pd.DataFrame) -> pd.DataFrame:
    validated = atoms.copy()
    for column in ("graph_id", "atom_idx"):
        validated[column] = _all_mmp_fidelity_exact_index_series(
            validated[column],
            name=column,
            minimum=0,
        )
    return validated


def load_mmp_inputs(
    *,
    mmp_path: Path,
    descriptor_path: Path,
    dataset_path: Path,
    teacher_path: Path,
    teacher_fold: int | None = None,
    descriptor_columns: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load canonical MMP rows and the frozen descriptor/teacher atom table.

    ``teacher_fold`` must be explicit when one teacher CSV contains distinct
    predictions for the same atom from multiple folds.
    """

    mmp = pd.read_csv(mmp_path, usecols=PAIR_COLUMNS)
    if mmp["pair_id"].duplicated().any():
        raise ValueError("MMP table contains duplicate pair_id values.")
    for column in ("i_id", "j_id"):
        mmp[column] = _all_mmp_fidelity_exact_index_series(mmp[column], name=column, minimum=0)

    descriptor_columns = [
        "molecule_id",
        "input_index",
        "atom_idx",
        "atom_symbol",
        *descriptor_columns,
    ]
    if "smiles" in pd.read_csv(descriptor_path, nrows=0).columns:
        descriptor_columns.append("smiles")
    atoms = pd.read_csv(descriptor_path, usecols=descriptor_columns)
    atoms["input_index"] = _all_mmp_fidelity_exact_index_series(
        atoms["input_index"],
        name="input_index",
        minimum=1,
    )
    atoms["atom_idx"] = _all_mmp_fidelity_exact_index_series(
        atoms["atom_idx"],
        name="atom_idx",
        minimum=0,
    )
    atoms = align_descriptors_to_dataset(atoms, dataset_path)
    if atoms.duplicated(["graph_id", "atom_idx"]).any():
        raise ValueError("Descriptor table contains duplicate graph_id/atom_idx keys.")

    teacher = load_teacher_atom_contributions(
        teacher_path,
        teacher_fold=teacher_fold,
    )

    atoms = atoms.merge(
        teacher,
        on=["graph_id", "atom_idx"],
        how="outer",
        validate="one_to_one",
        indicator="_atom_source",
    )
    if "smiles" in atoms.columns and "teacher_smiles" in atoms.columns:
        comparable = atoms["smiles"].notna() & atoms["teacher_smiles"].notna()
        mismatch = comparable & ~atoms["smiles"].astype(str).eq(
            atoms["teacher_smiles"].astype(str)
        )
        if mismatch.any():
            bad = atoms.loc[
                mismatch,
                ["graph_id", "smiles", "teacher_smiles"],
            ].drop_duplicates()
            raise ValueError(f"Teacher/descriptor SMILES mismatch:\n{bad.head(10)}")
    return mmp.sort_values("pair_id").reset_index(drop=True), atoms


def _pair_keys(row: pd.Series) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    graph_a = exact_integer(row["i_id"], name="i_id", minimum=0)
    graph_b = exact_integer(row["j_id"], name="j_id", minimum=0)
    keys_a = [(graph_a, index) for index in _parse_indices(row["diff_i_json"])]
    keys_b = [(graph_b, index) for index in _parse_indices(row["diff_j_json"])]
    return keys_a, keys_b


def audit_pair_coverage(
    mmp: pd.DataFrame,
    atoms: pd.DataFrame,
    *,
    descriptors: Sequence[str],
) -> pd.DataFrame:
    """Return one coverage/status row for every canonical MMP."""

    atoms = _validated_atom_keys(atoms)
    descriptor_complete = atoms[list(descriptors)].notna().all(axis=1)
    teacher_complete = atoms["atom_contribution"].notna()
    descriptor_keys = set(
        map(
            tuple,
            atoms.loc[descriptor_complete, ["graph_id", "atom_idx"]]
            .dropna()
            .to_numpy(),
        )
    )
    teacher_keys = set(
        map(
            tuple,
            atoms.loc[teacher_complete, ["graph_id", "atom_idx"]]
            .dropna()
            .to_numpy(),
        )
    )
    records: list[dict[str, object]] = []
    for row in mmp.to_dict("records"):
        pair = pd.Series(row)
        keys_a, keys_b = _pair_keys(pair)
        keys = keys_a + keys_b
        missing_descriptor = [key for key in keys if key not in descriptor_keys]
        missing_teacher = [key for key in keys if key not in teacher_keys]
        if not keys_a or not keys_b:
            status = "empty_diff_region"
        elif missing_descriptor and missing_teacher:
            status = "missing_descriptor_and_teacher"
        elif missing_descriptor:
            status = "missing_descriptor"
        elif missing_teacher:
            status = "missing_teacher"
        else:
            status = "eligible"
        records.append(
            {
                "pair_id": int(pair["pair_id"]),
                "document_id": str(pair["document_id"]),
                "mmp_class": str(pair["mmp_class"]),
                "molecule_A": str(pair["molecule_1_id"]),
                "molecule_B": str(pair["molecule_2_id"]),
                "n_diff_atoms_A": len(keys_a),
                "n_diff_atoms_B": len(keys_b),
                "n_diff_atoms_total": len(keys),
                "n_missing_descriptor_atoms": len(missing_descriptor),
                "n_missing_teacher_atoms": len(missing_teacher),
                "fit_status": status,
                "fit_eligible": status == "eligible",
            }
        )
    return pd.DataFrame.from_records(records)


def _pair_atom_rows(
    pair: pd.Series,
    atom_lookup: pd.DataFrame,
) -> pd.DataFrame:
    keys_a, keys_b = _pair_keys(pair)
    blocks: list[pd.DataFrame] = []
    for endpoint, molecule_id, keys in (
        ("A", str(pair["molecule_1_id"]), keys_a),
        ("B", str(pair["molecule_2_id"]), keys_b),
    ):
        block = atom_lookup.loc[keys].reset_index()
        block["context_id"] = f"pair_{int(pair['pair_id'])}"
        block["molecule_id"] = molecule_id
        block["endpoint"] = endpoint
        blocks.append(block)
    return pd.concat(blocks, ignore_index=True)


def _raw_pair_feature_row(
    pair: pd.Series,
    atom_rows: pd.DataFrame,
    *,
    descriptors: Sequence[str],
) -> dict[str, object]:
    by_endpoint = atom_rows.groupby("endpoint", sort=True)
    teacher_sum = by_endpoint["atom_contribution"].sum()
    row: dict[str, object] = {
        "pair_id": int(pair["pair_id"]),
        "document_id": str(pair["document_id"]),
        "mmp_class": str(pair["mmp_class"]),
        "molecule_A": str(pair["molecule_1_id"]),
        "molecule_B": str(pair["molecule_2_id"]),
        "n_diff_atoms_A": int((atom_rows["endpoint"] == "A").sum()),
        "n_diff_atoms_B": int((atom_rows["endpoint"] == "B").sum()),
        "teacher_delta_B_minus_A": float(teacher_sum.loc["B"] - teacher_sum.loc["A"]),
        "experimental_delta_pIC50_B_minus_A": float(pair["pIC50_j"] - pair["pIC50_i"]),
        "fp_similarity": float(pair["fp_similarity"]),
        "diff_total_count": int(pair["diff_total_count"]),
        "core_fraction_min": float(pair["core_fraction_min"]),
    }
    for descriptor in descriptors:
        sums = by_endpoint[descriptor].sum()
        means = by_endpoint[descriptor].mean()
        row[f"delta_sum__{descriptor}"] = float(sums.loc["B"] - sums.loc["A"])
        row[f"delta_mean__{descriptor}"] = float(means.loc["B"] - means.loc["A"])
    return row


def fit_all_mmp_pairs(
    mmp: pd.DataFrame,
    atoms: pd.DataFrame,
    *,
    variants: Mapping[str, Mapping[str, object]],
    primary_variant: str = "ridge",
    descriptors: Sequence[str],
) -> AllMmpFitResult:
    """Fit every eligible MMP with the explicitly selected descriptor set."""

    if primary_variant not in variants:
        raise ValueError(f"primary_variant={primary_variant!r} is absent from variants.")
    descriptors = tuple(descriptors)
    atoms = _validated_atom_keys(atoms)
    audit = audit_pair_coverage(mmp, atoms, descriptors=descriptors)
    atom_lookup = atoms.set_index(["graph_id", "atom_idx"]).sort_index()
    metric_records: list[dict[str, object]] = []
    atom_prediction_blocks: list[pd.DataFrame] = []
    coefficient_blocks: list[pd.DataFrame] = []
    concept_blocks: list[pd.DataFrame] = []
    feature_records: list[dict[str, object]] = []
    eligible_ids = set(audit.loc[audit["fit_eligible"], "pair_id"].astype(int))

    for pair in mmp.to_dict("records"):
        pair = pd.Series(pair)
        pair_id = int(pair["pair_id"])
        if pair_id not in eligible_ids:
            continue
        pair_atoms = _pair_atom_rows(pair, atom_lookup)
        feature_records.append(
            _raw_pair_feature_row(pair, pair_atoms, descriptors=descriptors)
        )
        fit_key = pd.Series(
            {
                "context_id": f"pair_{pair_id}",
                "pair_id": pair_id,
                "molecule_a": str(pair["molecule_1_id"]),
                "molecule_b": str(pair["molecule_2_id"]),
            }
        )
        for variant, parameters in variants.items():
            result = fit_pair(fit_key, pair_atoms, features=descriptors, **dict(parameters))
            atom_metric = result.metrics.loc[
                result.metrics["level"].eq("A+B diff-region atoms")
            ].iloc[0]
            delta_metric = result.metrics.loc[
                result.metrics["level"].eq("B-A fragment sum")
            ].iloc[0]
            target_sd = float(np.std(result.atoms["atom_contribution"].to_numpy(dtype=float)))
            metric_records.append(
                {
                    "pair_id": pair_id,
                    "document_id": str(pair["document_id"]),
                    "mmp_class": str(pair["mmp_class"]),
                    "molecule_A": str(pair["molecule_1_id"]),
                    "molecule_B": str(pair["molecule_2_id"]),
                    "variant": str(variant),
                    "model_kind": str(atom_metric["model_kind"]),
                    "n_original_atoms": int(atom_metric["n_original_atoms"]),
                    "n_training_rows": int(atom_metric["n_training_rows"]),
                    "atom_r2_in_sample": float(atom_metric["r2"]),
                    "atom_rmse_in_sample": float(atom_metric["rmse"]),
                    "atom_mae_in_sample": float(atom_metric["mae"]),
                    "atom_nrmse_by_pair_sd": (
                        float(atom_metric["rmse"]) / target_sd if target_sd > 1.0e-12 else np.nan
                    ),
                    "atom_pearson_r_in_sample": float(atom_metric["pearson_r"]),
                    "atom_spearman_r_in_sample": float(atom_metric["spearman_r"]),
                    "teacher_delta_B_minus_A": float(delta_metric["observed_delta"]),
                    "fitted_delta_B_minus_A": float(delta_metric["predicted_delta"]),
                    "delta_residual": float(
                        delta_metric["observed_delta"] - delta_metric["predicted_delta"]
                    ),
                    "abs_delta_error": float(delta_metric["mae"]),
                    "delta_direction_match": bool(
                        np.sign(delta_metric["observed_delta"])
                        == np.sign(delta_metric["predicted_delta"])
                    ),
                    "experimental_delta_pIC50_B_minus_A": float(
                        pair["pIC50_j"] - pair["pIC50_i"]
                    ),
                }
            )

            # Retain atom-level teacher and fitted values for direct c_i fidelity plots.
            atom_predictions = result.atoms[
                ["molecule_id", "atom_idx", "endpoint", "atom_contribution", "prediction", "residual"]
            ].copy()
            atom_predictions.insert(0, "variant", str(variant))
            atom_predictions.insert(0, "mmp_class", str(pair["mmp_class"]))
            atom_predictions.insert(0, "document_id", str(pair["document_id"]))
            atom_predictions.insert(0, "pair_id", pair_id)
            atom_prediction_blocks.append(atom_predictions)

            coefficients = result.coefficients.copy()
            coefficients.insert(0, "variant", str(variant))
            coefficients.insert(0, "document_id", str(pair["document_id"]))
            coefficients.insert(0, "pair_id", pair_id)
            coefficient_blocks.append(coefficients)

            if variant == primary_variant:
                concepts = result.concept_totals.drop_duplicates("concept")[["concept", "B_minus_A"]]
                concepts = concepts.rename(columns={"concept": "descriptor", "B_minus_A": "delta_effect"})
                concepts.insert(0, "molecule_B", str(pair["molecule_2_id"]))
                concepts.insert(0, "molecule_A", str(pair["molecule_1_id"]))
                concepts.insert(0, "mmp_class", str(pair["mmp_class"]))
                concepts.insert(0, "document_id", str(pair["document_id"]))
                concepts.insert(0, "pair_id", pair_id)
                concept_blocks.append(concepts)

    metrics = pd.DataFrame.from_records(metric_records)
    atom_predictions = pd.concat(atom_prediction_blocks, ignore_index=True)
    coefficients = pd.concat(coefficient_blocks, ignore_index=True)
    concepts = pd.concat(concept_blocks, ignore_index=True)
    pair_features = pd.DataFrame.from_records(feature_records)
    primary_delta = metrics.loc[
        metrics["variant"].eq(primary_variant),
        ["pair_id", "fitted_delta_B_minus_A"],
    ]
    reconstructed = concepts.groupby("pair_id", as_index=False)["delta_effect"].sum()
    checked = primary_delta.merge(reconstructed, on="pair_id", validate="one_to_one")
    if not np.allclose(
        checked["fitted_delta_B_minus_A"],
        checked["delta_effect"],
        atol=1.0e-10,
    ):
        raise AssertionError("Primary descriptor effects do not reconstruct fitted pair deltas.")
    return AllMmpFitResult(
        audit=audit,
        pair_features=pair_features,
        fit_metrics=metrics,
        atom_predictions=atom_predictions,
        coefficients=coefficients,
        concept_deltas=concepts,
    )
