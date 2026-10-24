"""Training for notebook-based atom-contribution experiments."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Sequence
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from .scaling import AffineLabelScaler
from .dataset import CacheBackedGraphDataset
from .dataset import exclude_rows
from .scaling import fit_label_scaler
import importlib.metadata
import platform
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence
from torch_geometric.data import Batch
from torch_geometric.nn import global_add_pool
from .dataset import BOND_FEATURE_DIM
from .scaling import AffineLabelScaler
from .dataset import CacheBackedGraphDataset
from .dataset import representation_cache_excluded_row_indices
from .dataset import split_fold
from .fragment_loss import FragmentFrequencySpec
from .fragment_loss import FragmentOccurrence
from .mmp_loss import MMPPairRecord
from .fragment_loss import build_fragment_frequency_spec
from .fragment_loss import compute_fragment_frequency_loss
from .mmp_loss import compute_mmp_pair_loss
from .mmp_loss import iter_pair_batches
from .fragment_loss import load_fragment_frequency_spec
from .mmp_loss import load_mmp_pair_records
from .model import BackboneConfig
from .model import CachedRepresentationPredictor
from .model import reconstruct_affine
from .model import validate_affine_additivity
from .io import write_json
from .io import exact_integer
from .config import PRODUCTION_RUN_PROFILES
from .config import TrainingConfig
from .config import serialize_output_path
from .mmp_loss import load_mmp_atom_mapping_sources
from .mmp_loss import reconstruct_mmp_graph_records
from .mmp_loss import validate_mmp_source_chemistry


@dataclass(frozen=True)
class RegressionMetrics:
    """Raw-scale regression errors and the two explicit R-squared measures."""

    count: int
    mse: float
    rmse: float
    mae: float
    r2_score: float
    pearson_r_squared: float


@dataclass(frozen=True)
class ScaledRegressionSplit:
    """One fold partition with labels scaled from training rows only."""

    dataset: CacheBackedGraphDataset
    scaler: AffineLabelScaler
    train_rows: tuple[int, ...]
    validation_rows: tuple[int, ...]


def seed_training_runtime(seed: int, *, deterministic_cudnn: bool) -> None:
    """Seed initialization and require repeatable Torch forward/backward kernels."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Repeated MMP atom indexing accumulates CPU gradients in parallel.
    # Seeding alone does not fix that reduction order; unsupported kernels fail.
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_cudnn and hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def source_row_data_loader(
    dataset: CacheBackedGraphDataset,
    source_rows: Sequence[int],
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    """Load declared source rows while preserving graph-to-source identity."""

    positions = [
        dataset.position_for_source_row(row) for row in source_rows
    ]
    return DataLoader(
        Subset(dataset, positions),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )


def prepare_scaled_regression_split(
    raw_dataset: CacheBackedGraphDataset,
    *,
    train_rows: Sequence[int],
    validation_rows: Sequence[int],
    excluded_rows: Sequence[int],
    target_column: str,
    scaling_mode: str,
) -> ScaledRegressionSplit:
    """Exclude ineligible rows and fit label scaling on training rows only."""

    active_train_rows = tuple(exclude_rows(train_rows, excluded_rows))
    active_validation_rows = tuple(
        exclude_rows(validation_rows, excluded_rows)
    )
    targets = pd.to_numeric(
        raw_dataset.frame[target_column],
        errors="raise",
    ).to_numpy(dtype=np.float64)
    scaler = fit_label_scaler(
        targets,
        active_train_rows,
        mode=scaling_mode,
    )
    return ScaledRegressionSplit(
        dataset=raw_dataset.with_scaler(scaler),
        scaler=scaler,
        train_rows=active_train_rows,
        validation_rows=active_validation_rows,
    )


def compute_regression_metrics(
    target: Sequence[float] | np.ndarray,
    prediction: Sequence[float] | np.ndarray,
) -> RegressionMetrics:
    """Compute error, regression R2, and squared Pearson correlation once.

    The coefficient of determination is

    ``1 - sum((y - y_hat)^2) / sum((y - mean(y))^2)``.
    Undefined R-squared values are represented by ``NaN`` so callers can
    retain their existing reporting or checkpoint-selection policy.
    """

    # Preserve the caller's floating precision so refactoring does not alter
    # epoch metrics produced from float32 model tensors.
    target_array = np.asarray(target).reshape(-1)
    prediction_array = np.asarray(prediction).reshape(-1)
    residual = target_array - prediction_array
    squared_residual = np.square(residual)
    total_sum_squares = float(
        np.square(target_array - target_array.mean()).sum()
    )
    r2_score = (
        float("nan")
        if total_sum_squares <= 0.0
        else 1.0 - float(squared_residual.sum()) / total_sum_squares
    )
    target_std = float(np.std(target_array))
    prediction_std = float(np.std(prediction_array))
    pearson_r_squared = float("nan")
    if target_std > 0.0 and prediction_std > 0.0:
        pearson = float(np.corrcoef(target_array, prediction_array)[0, 1])
        pearson_r_squared = pearson * pearson
    mse = float(squared_residual.mean())
    return RegressionMetrics(
        count=int(target_array.size),
        mse=mse,
        rmse=math.sqrt(mse),
        mae=float(np.abs(residual).mean()),
        r2_score=r2_score,
        pearson_r_squared=pearson_r_squared,
    )


def compute_prediction_loss(
    prediction: Tensor,
    target: Tensor,
    *,
    mode: str,
    huber_delta: float,
) -> Tensor:
    """Apply the declared MSE or Huber prediction objective in scaled space."""

    if mode == "mse":
        return F.mse_loss(prediction, target, reduction="mean")
    if mode == "huber":
        return F.huber_loss(
            prediction,
            target,
            reduction="mean",
            delta=huber_delta,
        )
    raise AssertionError(f"Unreachable prediction loss: {mode}")


CHECKPOINT_SCHEMA_VERSION = 4


RESULT_SCHEMA_VERSION = 4


RESULT_METADATA_FILENAME = "result_metadata.json"


RESULT_RECORDS_KEY = "results"


RESULT_SCOPE_KEY = "result_scope"


MMP_OUTPUT_CHECK_KEY = "output"


MMP_DY_ABSOLUTE_TOLERANCE = 1.0e-8


ADDITIVITY_ABSOLUTE_TOLERANCE = 1.0e-6


ADDITIVITY_RELATIVE_TOLERANCE = 1.0e-6


_FINAL_EPOCH_CRITERION = "fixed_final_epoch"


_VALIDATION_MSE_CRITERION = "validation_mse_scaled_minimum"


_VALIDATION_R2_CRITERION = "validation_r2_raw_maximum"


_CHECKPOINT_CRITERION_DESCRIPTIONS = {
    _FINAL_EPOCH_CRITERION: "fixed final epoch",
    _VALIDATION_MSE_CRITERION: (
        "minimum validation MSE in scaled target space"
    ),
    _VALIDATION_R2_CRITERION: (
        "maximum validation regression R2 in raw target space"
    ),
}


PREDICTION_COLUMNS = (
    "source_row_index",
    "dataset_position",
    "molecule_id",
    "sample_id",
    "smiles",
    "split",
    "n_atoms",
    "target_scaled",
    "prediction_scaled",
    "additivity_residual_scaled",
    "target_raw",
    "prediction_raw",
    "affine_baseline_raw",
    "atom_sum_raw",
    "additivity_residual_raw",
    "absolute_error_raw",
    "squared_error_raw",
)


ATOM_CONTRIBUTION_COLUMNS = (
    "source_row_index",
    "dataset_position",
    "molecule_id",
    "sample_id",
    "smiles",
    "split",
    "graph_atom_index",
    "smiles_atom_index",
    "atomic_number",
    "atom_contribution_scaled",
    "atom_contribution_raw",
)


HISTORY_COLUMNS = (
    "epoch",
    "optimization_prediction_loss_scaled",
    "optimization_prediction_mse_scaled",
    "optimization_mmp_total",
    "optimization_mmp_weighted",
    "optimization_mmp_core",
    "optimization_mmp_difference",
    "optimization_fragment_total",
    "optimization_fragment_weighted",
    "optimization_fragment_positive",
    "optimization_fragment_consistency",
    "optimization_fragment_inconsistency",
    "optimization_fragment_active_fragments",
    "optimization_fragment_active_occurrences",
    "optimization_total",
    "training_mae_raw",
    "training_rmse_raw",
    "training_r2_raw",
    "validation_mse_scaled",
    "validation_mae_raw",
    "validation_rmse_raw",
    "validation_r2_raw",
    "is_best",
)


@dataclass(frozen=True)
class EvaluationResult:
    """Evaluation tables in original dataset-row order."""

    predictions: pd.DataFrame
    atom_contributions: pd.DataFrame
    metrics: Mapping[str, Mapping[str, float]]


@dataclass(frozen=True)
class LoadedFoldCheckpoint:
    """Strictly reconstructed model, scaler, and checkpoint metadata."""

    model: CachedRepresentationPredictor
    scaler: AffineLabelScaler
    metadata: Mapping[str, Any]
    device: torch.device


@dataclass(frozen=True)
class TrainingResults:
    """Stable file contract returned by :func:`run_fold_training`.

    ``output_dir`` contains exactly these completed-run results:

    - ``training_config.json``
    - ``provenance.json``
    - ``history.csv``
    - ``best_checkpoint.pt``
    - ``predictions.csv``
    - ``atom_contributions.csv``
    - ``result_metadata.json``
    - ``COMPLETE``
    """

    output_dir: Path
    training_config: Path
    provenance: Path
    history: Path
    checkpoint: Path
    predictions: Path
    atom_contributions: Path
    result_metadata: Path
    complete: Path
    best_epoch: int
    best_validation_mse_scaled: float


@dataclass(frozen=True)
class _PreparedFold:
    config: TrainingConfig
    dataset: CacheBackedGraphDataset
    scaler: AffineLabelScaler
    fold_indices: Mapping[int, tuple[int, ...]]
    train_rows: tuple[int, ...]
    validation_rows: tuple[int, ...]
    excluded_rows: tuple[int, ...]
    all_mmp_records: tuple[MMPPairRecord, ...]
    train_mmp_records: tuple[MMPPairRecord, ...]
    mmp_records_by_i: Mapping[int, tuple[MMPPairRecord, ...]]
    hydrogen_excluded_pair_ids: tuple[int, ...]
    input_excluded_pair_ids: tuple[int, ...]
    validation_excluded_pair_ids: tuple[int, ...]
    mmp_mapping_check: Mapping[str, Any]
    cache_quality_check: Mapping[str, Any]
    fragment_spec: FragmentFrequencySpec | None
    fragment_occurrences_by_row: Mapping[
        int, tuple[FragmentOccurrence, ...]
    ] | None
    model: CachedRepresentationPredictor
    backbone_config: BackboneConfig
    device: torch.device


@dataclass(frozen=True)
class _FullObjectiveTerms:
    prediction_loss: Tensor
    prediction_mse: Tensor
    mmp_objective: Tensor
    mmp_total: float
    mmp_core: float
    mmp_difference: float
    fragment_total: Tensor
    fragment_positive: Tensor
    fragment_consistency: Tensor
    fragment_inconsistency: Tensor
    fragment_active_fragments: int
    fragment_active_occurrences: int


@dataclass(frozen=True)
class _BatchLossTerms:
    """Raw loss components from one molecule batch and one shared forward."""

    moved: Batch
    prediction_loss: Tensor
    prediction_mse: Tensor
    mmp_core: Tensor
    mmp_difference: Tensor
    mmp_pair_count: int
    fragment_positive: Tensor
    fragment_consistency: Tensor
    fragment_inconsistency: Tensor
    fragment_active_fragments: int
    fragment_active_occurrences: int


def _checkpoint_selection_criterion(config: TrainingConfig) -> str:
    """Resolve the existing checkpoint rule without changing its metric."""

    # NOTE: The Notebook supplies one of the two declared selection modes.
    if config.checkpoint_selection != "best_validation":
        return _FINAL_EPOCH_CRITERION
    if config.early_stopping_patience is not None:
        return _VALIDATION_R2_CRITERION
    return _VALIDATION_MSE_CRITERION


def _resolve_device(device_text: str) -> torch.device:
    # NOTE: CONTRACT in=device_text: strict torch device specification
    device = torch.device(device_text)
    if device.type == "cpu":
        if device_text != "cpu":
            raise ValueError("CPU device must be declared exactly as 'cpu'")
        return device
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA device was explicitly requested but unavailable: {device_text}"
            )
        index = 0 if device.index is None else device.index
        if index < 0 or index >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA device index is unavailable: {device_text}")
        return torch.device(f"cuda:{index}")
    if device.type == "mps":
        if device_text != "mps":
            raise ValueError("MPS device must be declared exactly as 'mps'")
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS device was explicitly requested but unavailable")
        return device
    raise ValueError("device must explicitly name cpu, cuda, cuda:N, or mps")


def _package_version(name: str) -> str:
    # NOTE: CONTRACT in=name: installed package must expose metadata; absence fails-fast
    return importlib.metadata.version(name)


def _validate_output_target(path: Path) -> None:
    if path.exists():
        raise FileExistsError(
            f"output_dir already exists; refusing to mix or overwrite runs: {path}"
        )
    parent = path.parent
    if not parent.is_dir():
        raise NotADirectoryError(
            f"output_dir parent must already exist: {parent}"
        )


def _validate_checked_workspace_paths(config: TrainingConfig) -> None:
    if config.run_profile not in PRODUCTION_RUN_PROFILES:
        return
    declared_paths = {
        "data.dataset_csv": config.data.dataset_csv,
        "data.graph_root": config.data.graph_root,
        "data.fold_csv": config.data.fold_csv,
        "data.representation_cache": config.data.representation_cache,
        "mmp_csv": config.mmp_csv,
        "output_dir": config.output_dir,
    }
    if config.fragment_cache is not None:
        declared_paths["fragment_cache"] = config.fragment_cache
    external = [
        name
        for name, path in declared_paths.items()
        if serialize_output_path(path)[1] != "XAI_CHEM_ROOT_relative"
    ]
    if external:
        raise ValueError(
            "Checked production paths must be XAI_CHEM_ROOT-relative; "
            f"external fields: {external}"
        )


def _portable_path_record(path: Path | str) -> dict[str, str]:
    serialized, reference = serialize_output_path(path)
    return {
        "path": serialized,
        "path_reference": reference,
    }


def _validate_mmp_output_metadata(
    mmp_csv: Path,
    table: pd.DataFrame,
    *,
    expected_mapping_method: str,
    required: bool,
) -> Mapping[str, Any]:
    """Record the declared MMP table without an independent source lock."""

    # NOTE: CONTRACT the caller selects the MMP CSV and atom-mapping method.
    # Git owns code history; the visible path and row count identify the input.
    return {
        "required": bool(required),
        **_portable_path_record(mmp_csv),
        "core_mapping_method": expected_mapping_method,
        "row_count": len(table),
    }


def _validate_mmp_source_rows(
    frame: pd.DataFrame,
    mmp_csv: Path,
    records: Sequence[MMPPairRecord],
    *,
    smiles_column: str,
    target_column: str,
    molecule_id_column: str,
    graph_file_template: str,
    expected_mapping_method: str,
    expected_source_pair_count: int,
    expected_symmetry_ambiguous_pair_count: int,
    expected_max_automorphism_candidates: int,
    require_versioned_metadata: bool,
) -> Mapping[str, Any]:
    table = pd.read_csv(mmp_csv)
    output_check = _validate_mmp_output_metadata(
        mmp_csv,
        table,
        expected_mapping_method=expected_mapping_method,
        required=require_versioned_metadata,
    )
    required = {
        "pair_id",
        "i_id",
        "j_id",
        "dy",
        "smiles_i",
        "smiles_j",
        "graph_i_file",
        "graph_j_file",
        "dataset_index_1",
        "dataset_index_2",
        "core_mapping_method",
        "core_pair_automorphism_candidate_count",
        "core_correspondence_unique",
        "core_correspondence_symmetric",
    }
    missing = sorted(required - set(table.columns))
    if missing:
        raise KeyError(
            "MMP endpoint provenance columns are missing: "
            + ", ".join(missing)
        )
    if len(table) != len(records):
        raise ValueError("MMP table and parsed record counts differ")
    if len(records) != expected_source_pair_count:
        raise ValueError(
            "MMP source-pair count differs from the declared run contract: "
            f"expected {expected_source_pair_count}, got {len(records)}"
        )
    rows_by_pair_id = {
        exact_integer(
            row["pair_id"],
            name=f"MMP pair_id at CSV row {row_number}",
        ): row
        for row_number, row in table.iterrows()
    }
    if len(rows_by_pair_id) != len(records):
        raise ValueError("MMP pair_id values are not unique")

    ambiguous_pair_count = 0
    maximum_candidate_count = 0
    for record in records:
        if record.pair_id not in rows_by_pair_id:
            raise ValueError(f"MMP pair_id is absent from source table: {record.pair_id}")
        row = rows_by_pair_id[record.pair_id]
        if str(row["core_mapping_method"]) != expected_mapping_method:
            raise ValueError(
                f"MMP pair {record.pair_id} core mapping method differs "
                "from the declared run contract"
            )
        candidate_count = exact_integer(
            row["core_pair_automorphism_candidate_count"],
            name=(
                f"MMP pair {record.pair_id} automorphism candidate count"
            ),
        )
        if candidate_count != len(record.core_pair_candidates):
            raise ValueError(
                f"MMP pair {record.pair_id} automorphism candidate count "
                "does not match core_pair_candidates_json"
            )
        unique_value = row["core_correspondence_unique"]
        symmetric_value = row["core_correspondence_symmetric"]
        if not isinstance(unique_value, (bool, np.bool_)):
            raise ValueError(
                f"MMP pair {record.pair_id} unique flag is not boolean"
            )
        if not isinstance(symmetric_value, (bool, np.bool_)):
            raise ValueError(
                f"MMP pair {record.pair_id} symmetric flag is not boolean"
            )
        is_unique = bool(unique_value)
        if is_unique != (candidate_count == 1):
            raise ValueError(
                f"MMP pair {record.pair_id} unique flag contradicts "
                "its automorphism candidate count"
            )
        if not bool(symmetric_value):
            raise ValueError(
                f"MMP pair {record.pair_id} is not declared "
                "reversal-symmetric"
            )
        ambiguous_pair_count += int(not is_unique)
        maximum_candidate_count = max(
            maximum_candidate_count,
            candidate_count,
        )
        row_i_id = exact_integer(
            row["i_id"],
            name=f"MMP pair {record.pair_id} i_id",
        )
        row_j_id = exact_integer(
            row["j_id"],
            name=f"MMP pair {record.pair_id} j_id",
        )
        if row_i_id != record.i_id or row_j_id != record.j_id:
            raise ValueError(f"MMP endpoint mismatch for pair {record.pair_id}")
        if min(record.i_id, record.j_id) < 0 or max(
            record.i_id, record.j_id
        ) >= len(frame):
            raise IndexError(
                f"MMP pair {record.pair_id} has an out-of-range dataset endpoint"
            )
        frame_i = frame.iloc[record.i_id]
        frame_j = frame.iloc[record.j_id]
        if str(row["smiles_i"]) != str(frame_i[smiles_column]):
            raise ValueError(f"MMP i-SMILES mismatch for pair {record.pair_id}")
        if str(row["smiles_j"]) != str(frame_j[smiles_column]):
            raise ValueError(f"MMP j-SMILES mismatch for pair {record.pair_id}")
        molecule_id_i = exact_integer(
            frame_i[molecule_id_column],
            name=f"dataset molecule id at row {record.i_id}",
        )
        molecule_id_j = exact_integer(
            frame_j[molecule_id_column],
            name=f"dataset molecule id at row {record.j_id}",
        )
        if exact_integer(
            row["dataset_index_1"],
            name=f"MMP pair {record.pair_id} dataset_index_1",
        ) != molecule_id_i:
            raise ValueError(
                f"MMP i dataset identifier mismatch for pair {record.pair_id}"
            )
        if exact_integer(
            row["dataset_index_2"],
            name=f"MMP pair {record.pair_id} dataset_index_2",
        ) != molecule_id_j:
            raise ValueError(
                f"MMP j dataset identifier mismatch for pair {record.pair_id}"
            )
        expected_graph_i = graph_file_template.format(
            molecule_id=molecule_id_i
        )
        expected_graph_j = graph_file_template.format(
            molecule_id=molecule_id_j
        )
        if str(row["graph_i_file"]) != expected_graph_i:
            raise ValueError(
                f"MMP i graph filename mismatch for pair {record.pair_id}"
            )
        if str(row["graph_j_file"]) != expected_graph_j:
            raise ValueError(
                f"MMP j graph filename mismatch for pair {record.pair_id}"
            )
        expected_dy = float(frame_i[target_column]) - float(
            frame_j[target_column]
        )
        if not math.isclose(
            record.dy,
            expected_dy,
            rel_tol=0.0,
            abs_tol=MMP_DY_ABSOLUTE_TOLERANCE,
        ):
            raise ValueError(
                f"MMP dy mismatch for pair {record.pair_id}: "
                f"declared={record.dy}, expected={expected_dy}"
            )
    if ambiguous_pair_count != expected_symmetry_ambiguous_pair_count:
        raise ValueError(
            "MMP symmetry-ambiguous pair count differs from the declared "
            f"run contract: expected "
            f"{expected_symmetry_ambiguous_pair_count}, "
            f"got {ambiguous_pair_count}"
        )
    if maximum_candidate_count != expected_max_automorphism_candidates:
        raise ValueError(
            "MMP maximum automorphism candidate count differs from the "
            f"declared run contract: expected "
            f"{expected_max_automorphism_candidates}, "
            f"got {maximum_candidate_count}"
        )
    return {
        "method": expected_mapping_method,
        MMP_OUTPUT_CHECK_KEY: output_check,
        "source_pair_count": len(records),
        "unique_correspondence_pair_count": (
            len(records) - ambiguous_pair_count
        ),
        "symmetry_ambiguous_pair_count": ambiguous_pair_count,
        "maximum_automorphism_candidate_count": maximum_candidate_count,
        "core_objective": (
            "min over valid shared-core automorphisms of mean atomwise "
            "SmoothL1"
        ),
        "raw_core_pairs_policy": (
            "strict canonical representative and membership declaration; "
            "optimization uses every metadata-declared automorphism candidate"
        ),
    }


def _validate_cache_geometry_quality(
    dataset: CacheBackedGraphDataset,
) -> Mapping[str, Any]:
    """Fail closed for MMFF-derived 3D caches without converged geometries."""

    provider = dict(dataset.cache.metadata["provider"])
    conformer_method = provider.get("conformer_method")
    mmff_provider = (
        isinstance(conformer_method, str)
        and "mmff" in conformer_method.lower()
    )
    convergence_entries: list[tuple[int, bool, int]] = []
    missing_metadata_rows: list[int] = []
    for row in dataset.cache.molecule_rows:
        metadata = row["provider_metadata"]
        has_mmff_metadata = (
            "mmff_converged" in metadata
            or "mmff_optimization_status" in metadata
        )
        if not mmff_provider and not has_mmff_metadata:
            continue
        graph_idx = exact_integer(
            row["graph_idx"],
            name="representation cache graph_idx",
        )
        converged = metadata.get("mmff_converged")
        status = metadata.get("mmff_optimization_status")
        if not isinstance(converged, bool) or isinstance(status, bool) or not isinstance(
            status, int
        ):
            missing_metadata_rows.append(graph_idx)
            continue
        convergence_entries.append((graph_idx, converged, status))

    if (
        not mmff_provider
        and not convergence_entries
        and not missing_metadata_rows
    ):
        return {
            "policy": "provider_declared_geometry_quality_v1",
            "applicable": False,
            "reason": "cache provider does not declare MMFF-derived coordinates",
        }
    if provider.get("mmff_require_convergence") is not True:
        raise ValueError(
            "MMFF-derived representation cache is scientifically ineligible: "
            "cache metadata must declare mmff_require_convergence=true"
        )
    if missing_metadata_rows:
        raise ValueError(
            "MMFF-derived representation cache is missing typed convergence "
            f"metadata for graph_idx values {missing_metadata_rows[:10]}"
        )
    nonconverged = [
        graph_idx
        for graph_idx, converged, status in convergence_entries
        if not converged or status != 0
    ]
    if nonconverged:
        raise ValueError(
            "MMFF-derived representation cache contains non-converged "
            f"geometries: count={len(nonconverged)}, "
            f"sample_graph_idx={nonconverged[:10]}"
        )
    return {
        "policy": "provider_declared_geometry_quality_v1",
        "applicable": True,
        "conformer_method": conformer_method,
        "mmff_require_convergence": True,
        "molecule_count": len(convergence_entries),
        "converged_count": sum(
            converged for _, converged, _ in convergence_entries
        ),
        "optimization_status_zero_count": sum(
            status == 0 for _, _, status in convergence_entries
        ),
    }


def _prepare_fold(
    config: TrainingConfig,
    raw_dataset: CacheBackedGraphDataset | None = None,
) -> _PreparedFold:
    """Bind one fixed split, fit the training scaler and prepare train-only AC constraints."""

    _validate_checked_workspace_paths(config)
    _validate_output_target(Path(config.output_dir))
    config.data.validate_paths()
    mmp_csv = Path(config.mmp_csv)
    if not mmp_csv.is_file():
        raise FileNotFoundError(f"mmp_csv does not exist: {mmp_csv}")
    device = _resolve_device(config.device)

    # One trial shares the raw frame, fold assignment, and graph tensors.
    if raw_dataset is None:
        raw_dataset = CacheBackedGraphDataset(
            config.data,
            validate_all=True,
        )
    frame = raw_dataset.frame
    fold_indices = raw_dataset.fold_indices
    if config.training_scope == "cross_validation_fold":
        train_rows_all, validation_rows_all = split_fold(
            fold_indices,
            config.validation_fold,
        )
    elif config.training_scope == "full_dataset":
        declared_rows = sorted(
            row
            for rows in fold_indices.values()
            for row in rows
        )
        if declared_rows != list(range(len(frame))):
            raise ValueError(
                "Full-dataset training requires the fold metadata to cover "
                "every source row exactly once"
            )
        train_rows_all = tuple(declared_rows)
        validation_rows_all = ()
    else:
        raise AssertionError(
            f"Unreachable training_scope: {config.training_scope}"
        )
    explicit_hydrogen_rows = raw_dataset.explicit_hydrogen_rows
    if (
        len(explicit_hydrogen_rows)
        != config.expected_explicit_hydrogen_exclusions
    ):
        raise ValueError(
            "Explicit-H/D/T exclusion count differs from the declared run "
            f"contract: expected "
            f"{config.expected_explicit_hydrogen_exclusions}, "
            f"got {len(explicit_hydrogen_rows)}"
        )
    cache_bundle = raw_dataset.cache
    excluded_rows = representation_cache_excluded_row_indices(
        frame,
        cache_bundle.molecule_rows,
        sample_id_column=config.data.sample_id_column,
    )
    if not set(explicit_hydrogen_rows).issubset(excluded_rows):
        raise ValueError(
            "Representation cache includes rows containing explicit H/D/T"
        )
    expected_input_exclusions = (
        config.expected_explicit_hydrogen_exclusions
        if config.expected_input_exclusions is None
        else config.expected_input_exclusions
    )
    if len(excluded_rows) != expected_input_exclusions:
        raise ValueError(
            "Representation-cache input exclusion count differs from the "
            "declared run contract: expected "
            f"{expected_input_exclusions}, got {len(excluded_rows)}"
        )
    scaled_split = prepare_scaled_regression_split(
        raw_dataset,
        train_rows=train_rows_all,
        validation_rows=validation_rows_all,
        excluded_rows=excluded_rows,
        target_column=config.data.target_column,
        scaling_mode=config.scaling_mode,
    )
    train_rows = scaled_split.train_rows
    validation_rows = scaled_split.validation_rows
    if not train_rows or (
        config.training_scope == "cross_validation_fold"
        and not validation_rows
    ):
        raise ValueError(
            "Both active train and validation folds must remain non-empty"
        )
    if (
        config.expected_train_rows is not None
        and len(train_rows) != config.expected_train_rows
    ):
        raise ValueError(
            "Active train-row count differs from the declared fold contract: "
            f"expected {config.expected_train_rows}, got {len(train_rows)}"
        )
    validation_excluded_hydrogen_count = (
        len(set(validation_rows_all) & set(explicit_hydrogen_rows))
        if config.training_scope == "cross_validation_fold"
        else 0
    )
    if (
        config.expected_validation_explicit_hydrogen_exclusions is not None
        and validation_excluded_hydrogen_count
        != config.expected_validation_explicit_hydrogen_exclusions
    ):
        raise ValueError(
            "Validation-fold explicit-H/D/T exclusion count differs from "
            "the declared fold contract: expected "
            f"{config.expected_validation_explicit_hydrogen_exclusions}, "
            f"got {validation_excluded_hydrogen_count}"
        )
    validation_excluded_input_count = (
        len(set(validation_rows_all) & set(excluded_rows))
        if config.training_scope == "cross_validation_fold"
        else 0
    )
    expected_validation_input_exclusions = (
        config.expected_validation_explicit_hydrogen_exclusions
        if config.expected_validation_input_exclusions is None
        else config.expected_validation_input_exclusions
    )
    if (
        expected_validation_input_exclusions is not None
        and validation_excluded_input_count
        != expected_validation_input_exclusions
    ):
        raise ValueError(
            "Validation-fold representation-cache input exclusion count "
            "differs from the declared fold contract: expected "
            f"{expected_validation_input_exclusions}, "
            f"got {validation_excluded_input_count}"
        )

    scaler = scaled_split.scaler
    dataset = scaled_split.dataset
    dataset_excluded = tuple(
        exclusion.source_row_index for exclusion in dataset.exclusions
    )
    if dataset_excluded != excluded_rows:
        raise ValueError(
            "Modeling eligibility and representation-cache exclusions differ"
        )
    expected_active_rows = tuple(
        row for row in range(len(frame)) if row not in set(excluded_rows)
    )
    if dataset.source_row_indices != expected_active_rows:
        raise ValueError(
            "Cache-backed dataset row order does not equal the active source rows"
        )
    cache_quality_check = _validate_cache_geometry_quality(dataset)
    if config.fragment_frequency_weight > 0.0:
        fragment_spec = (
            load_fragment_frequency_spec(
                config.fragment_cache,
                dataset,
                train_rows,
                min_heavy_atoms=config.fragment_min_heavy_atoms,
                min_count=config.fragment_min_count,
                min_frequency=config.fragment_min_frequency,
            )
            if config.fragment_cache is not None
            else build_fragment_frequency_spec(
                dataset,
                train_rows,
                min_heavy_atoms=config.fragment_min_heavy_atoms,
                min_count=config.fragment_min_count,
                min_frequency=config.fragment_min_frequency,
            )
        )
    else:
        fragment_spec = None

    all_mmp_records = load_mmp_pair_records(mmp_csv)
    mmp_table = pd.read_csv(mmp_csv)
    mmp_mapping_check = _validate_mmp_source_rows(
        frame,
        mmp_csv,
        all_mmp_records,
        smiles_column=config.data.smiles_column,
        target_column=config.data.target_column,
        molecule_id_column=config.data.molecule_id_column,
        graph_file_template=dataset.graph_file_template,
        expected_mapping_method=(
            config.expected_mmp_core_mapping_method
        ),
        expected_source_pair_count=config.expected_mmp_source_pairs,
        expected_symmetry_ambiguous_pair_count=(
            config.expected_mmp_symmetry_ambiguous_pairs
        ),
        expected_max_automorphism_candidates=(
            config.expected_mmp_max_automorphism_candidates
        ),
        require_versioned_metadata=(
            config.run_profile in PRODUCTION_RUN_PROFILES
        ),
    )
    mapping_sources = load_mmp_atom_mapping_sources(
        mmp_table,
        all_mmp_records,
        expected_mapping_method=config.expected_mmp_core_mapping_method,
    )
    source_chemistry_check = validate_mmp_source_chemistry(
        all_mmp_records,
        mapping_sources,
    )
    if (
        source_chemistry_check["symmetry_ambiguous_pair_count"]
        != config.expected_mmp_symmetry_ambiguous_pairs
    ):
        raise ValueError(
            "Independently reconstructed MMP ambiguity count differs from "
            "the declared run contract"
        )
    if (
        source_chemistry_check["maximum_automorphism_candidate_count"]
        != config.expected_mmp_max_automorphism_candidates
    ):
        raise ValueError(
            "Independently reconstructed MMP maximum automorphism count "
            "differs from the declared run contract"
        )
    excluded_set = set(excluded_rows)
    train_set = set(train_rows)
    hydrogen_excluded = tuple(
        record.pair_id
        for record in all_mmp_records
        if (
            record.i_id in set(explicit_hydrogen_rows)
            or record.j_id in set(explicit_hydrogen_rows)
        )
    )
    if (
        len(hydrogen_excluded)
        != config.expected_mmp_pairs_excluded_by_hydrogen
    ):
        raise ValueError(
            "Explicit-H/D/T incident MMP count differs from the declared run "
            f"contract: expected "
            f"{config.expected_mmp_pairs_excluded_by_hydrogen}, "
            f"got {len(hydrogen_excluded)}"
        )
    input_excluded = tuple(
        record.pair_id
        for record in all_mmp_records
        if record.i_id in excluded_set or record.j_id in excluded_set
    )
    expected_mmp_input_exclusions = (
        config.expected_mmp_pairs_excluded_by_hydrogen
        if config.expected_mmp_pairs_excluded_by_input is None
        else config.expected_mmp_pairs_excluded_by_input
    )
    if len(input_excluded) != expected_mmp_input_exclusions:
        raise ValueError(
            "Representation-cache-excluded incident MMP count differs from "
            "the declared run contract: expected "
            f"{expected_mmp_input_exclusions}, got {len(input_excluded)}"
        )
    input_excluded_set = set(input_excluded)
    active_mmp = tuple(
        record
        for record in all_mmp_records
        if record.pair_id not in input_excluded_set
    )
    train_mmp = tuple(
        record
        for record in active_mmp
        if record.i_id in train_set and record.j_id in train_set
    )
    if len(train_mmp) != config.expected_retained_mmp_pairs:
        raise ValueError(
            "Retained fold-training MMP count differs from the declared run "
            f"contract: expected {config.expected_retained_mmp_pairs}, "
            f"got {len(train_mmp)}"
        )
    train_pair_ids = {record.pair_id for record in train_mmp}
    validation_excluded = tuple(
        record.pair_id
        for record in active_mmp
        if record.pair_id not in train_pair_ids
    )
    if len(input_excluded) + len(active_mmp) != len(all_mmp_records):
        raise AssertionError("MMP exclusion accounting is incomplete")
    if len(train_mmp) + len(validation_excluded) != len(active_mmp):
        raise AssertionError("MMP fold accounting is incomplete")
    mapping_result = reconstruct_mmp_graph_records(
        dataset,
        train_mmp,
        mapping_sources,
    )
    train_mmp_graph_order = mapping_result.records
    mmp_mapping_check = {
        **dict(mmp_mapping_check),
        "all_source_chemical_validation": dict(source_chemistry_check),
        "retained_graph_order_reconstruction": dict(mapping_result.check),
    }

    # Index each oriented pair once by its i endpoint. A molecule batch can
    # then collect every induced MMP without rescanning the full population.
    mmp_records_by_i_lists = {row: [] for row in train_rows}
    for record in train_mmp_graph_order:
        mmp_records_by_i_lists[record.i_id].append(record)
    mmp_records_by_i = {
        row: tuple(records)
        for row, records in mmp_records_by_i_lists.items()
    }

    if fragment_spec is None:
        fragment_occurrences_by_row = None
    else:
        # Fragment frequencies and the positive margin remain fold-global;
        # only occurrence membership changes with each molecule batch.
        fragment_occurrence_lists = {row: [] for row in train_rows}
        for occurrence in fragment_spec.occurrences:
            fragment_occurrence_lists[occurrence.source_row_index].append(
                occurrence
            )
        fragment_occurrences_by_row = {
            row: tuple(occurrences)
            for row, occurrences in fragment_occurrence_lists.items()
        }

    first_graph = dataset[0]
    input_dim = int(first_graph.x.shape[1])
    if input_dim <= 0:
        raise ValueError("Cached atom representation dimension must be positive")
    backbone_config = BackboneConfig(
        input_dim=input_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
        name=config.backbone_name,
        heads=config.heads,
        use_edge_features=config.use_edge_features,
        edge_dim=BOND_FEATURE_DIM if config.use_edge_features else None,
        residual=config.backbone_residual,
        normalization=config.backbone_normalization,
    )
    model = CachedRepresentationPredictor(
        backbone_config,
        readout_hidden_dim=config.readout_hidden_dim,
        readout_num_layers=config.readout_num_layers,
        atom_bias=config.atom_bias,
    )
    if config.output_head_init_std is not None:
        model.readout.initialize_output_layer_normal(
            config.output_head_init_std
        )
    model = model.to(device)
    return _PreparedFold(
        config=config,
        dataset=dataset,
        scaler=scaler,
        fold_indices={
            fold: tuple(rows) for fold, rows in fold_indices.items()
        },
        train_rows=train_rows,
        validation_rows=validation_rows,
        excluded_rows=excluded_rows,
        all_mmp_records=all_mmp_records,
        train_mmp_records=train_mmp_graph_order,
        mmp_records_by_i=mmp_records_by_i,
        hydrogen_excluded_pair_ids=hydrogen_excluded,
        input_excluded_pair_ids=input_excluded,
        validation_excluded_pair_ids=validation_excluded,
        mmp_mapping_check=mmp_mapping_check,
        cache_quality_check=cache_quality_check,
        fragment_spec=fragment_spec,
        fragment_occurrences_by_row=fragment_occurrences_by_row,
        model=model,
        backbone_config=backbone_config,
        device=device,
    )


def _forward_batch(
    model: CachedRepresentationPredictor,
    batch: Batch,
    device: torch.device,
):
    moved = batch.to(device)
    output = model(
        cached_node_representations=moved.x,
        edge_index=moved.edge_index,
        batch_index=moved.batch,
        edge_attr=moved.edge_attr,
    )
    return moved, output


def _contributions_by_source_row(
    source_rows: Sequence[int],
    atom_offsets: Sequence[int],
    atom_contributions: Tensor,
) -> dict[int, Tensor]:
    """Expose contiguous per-molecule views without CUDA masks or syncs."""

    # NOTE: CONTRACT source rows are unique and offsets are the CPU Batch.ptr
    # captured before the graph batch is transferred to the accelerator.
    return {
        int(source_row): atom_contributions[start:stop]
        for source_row, start, stop in zip(
            source_rows,
            atom_offsets[:-1],
            atom_offsets[1:],
        )
    }


def _graph_batch_for_rows(
    dataset: CacheBackedGraphDataset,
    source_rows: Sequence[int],
) -> Batch:
    """Build the molecular batch needed for the explicitly selected source rows."""

    graphs = [
        dataset[dataset.position_for_source_row(source_row)]
        for source_row in source_rows
    ]
    return Batch.from_data_list(graphs)


def _mmp_records_for_rows(
    prepared: _PreparedFold,
    source_rows: Sequence[int],
) -> tuple[MMPPairRecord, ...]:
    """Return every train-fold MMP induced by one molecule batch."""

    # NOTE: CONTRACT source_rows are unique train rows from one DataLoader batch.
    row_set = set(source_rows)
    return tuple(
        record
        for row in source_rows
        for record in prepared.mmp_records_by_i[row]
        if record.j_id in row_set
    )


def _fragment_spec_for_rows(
    prepared: _PreparedFold,
    source_rows: Sequence[int],
) -> FragmentFrequencySpec:
    """Restrict occurrences while retaining fold-global fragment statistics."""

    # NOTE: CONTRACT fragment data are prepared when the fragment weight is active.
    occurrences = tuple(
        occurrence
        for row in source_rows
        for occurrence in prepared.fragment_occurrences_by_row[row]
    )
    return replace(prepared.fragment_spec, occurrences=occurrences)


def _scaled_mmp_objective(
    config: TrainingConfig,
    core: Tensor,
    difference: Tensor,
) -> Tensor:
    """Combine fixed-gradient-scaled MMP components once for all train modes."""

    scaling = config.loss_gradient_scaling
    return (
        config.mmp_core_weight * scaling.mmp_core_scale * core
        + config.mmp_difference_weight
        * scaling.mmp_difference_scale
        * difference
    ) / (config.mmp_core_weight + config.mmp_difference_weight)


def _scaled_fragment_objective(
    config: TrainingConfig,
    positive: Tensor,
    consistency: Tensor,
) -> Tensor:
    """Combine fixed-gradient-scaled fragment components once."""

    scaling = config.loss_gradient_scaling
    return (
        config.fragment_positive_weight
        * scaling.fragment_positive_scale
        * positive
        + config.fragment_consistency_weight
        * scaling.fragment_consistency_scale
        * consistency
    )


def _batch_loss_terms(
    prepared: _PreparedFold,
    batch: Batch,
    *,
    include_mmp: bool,
    include_fragment: bool,
) -> _BatchLossTerms:
    """Evaluate raw batch losses from one forward without an optimizer update."""

    config = prepared.config
    batch_rows = tuple(
        int(row) for row in batch.graph_idx.reshape(-1).tolist()
    )
    atom_offsets = tuple(int(value) for value in batch.ptr.tolist())
    moved, output = _forward_batch(prepared.model, batch, prepared.device)
    targets = moved.y.reshape(-1)
    prediction_mse = F.mse_loss(
        output.prediction,
        targets,
        reduction="mean",
    )
    prediction_loss = compute_prediction_loss(
        output.prediction,
        targets,
        mode=config.prediction_loss_mode,
        huber_delta=config.prediction_huber_delta,
    )
    zero = prediction_loss.new_zeros(())
    mmp_core = zero
    mmp_difference = zero
    mmp_pair_count = 0
    fragment_positive = zero
    fragment_consistency = zero
    fragment_inconsistency = zero
    fragment_active_fragments = 0
    fragment_active_occurrences = 0
    if include_mmp or include_fragment:
        contributions = _contributions_by_source_row(
            batch_rows,
            atom_offsets,
            output.atom_contributions,
        )
    if include_mmp:
        batch_mmp_records = _mmp_records_for_rows(prepared, batch_rows)
        if batch_mmp_records:
            mmp = compute_mmp_pair_loss(
                contributions,
                batch_mmp_records,
                target_alpha=prepared.scaler.alpha,
                core_weight=1.0,
                difference_weight=1.0,
                huber_beta=config.mmp_huber_beta,
            )
            mmp_core = mmp.core
            mmp_difference = mmp.difference
            mmp_pair_count = mmp.pair_count
    if include_fragment:
        batch_fragment_spec = _fragment_spec_for_rows(
            prepared,
            batch_rows,
        )
        active_occurrences = tuple(
            occurrence
            for occurrence in batch_fragment_spec.occurrences
            if batch_fragment_spec.frequencies[
                occurrence.fragment_smiles
            ]
            > batch_fragment_spec.min_frequency
        )
        if active_occurrences:
            fragment = compute_fragment_frequency_loss(
                contributions,
                replace(
                    batch_fragment_spec,
                    occurrences=active_occurrences,
                ),
                positive_weight=1.0,
                consistency_weight=1.0,
            )
            fragment_positive = fragment.positive
            fragment_consistency = fragment.consistency
            fragment_inconsistency = fragment.inconsistency
            fragment_active_fragments = fragment.active_fragment_count
            fragment_active_occurrences = fragment.active_occurrence_count
    return _BatchLossTerms(
        moved=moved,
        prediction_loss=prediction_loss,
        prediction_mse=prediction_mse,
        mmp_core=mmp_core,
        mmp_difference=mmp_difference,
        mmp_pair_count=mmp_pair_count,
        fragment_positive=fragment_positive,
        fragment_consistency=fragment_consistency,
        fragment_inconsistency=fragment_inconsistency,
        fragment_active_fragments=fragment_active_fragments,
        fragment_active_occurrences=fragment_active_occurrences,
    )


def _minibatch_optimization_epoch(
    prepared: _PreparedFold,
    optimizer: torch.optim.Optimizer,
    *,
    epoch_index: int,
) -> dict[str, float]:
    """Optimize one combined objective per shuffled molecule batch."""

    config = prepared.config
    model = prepared.model
    model.train()
    loader = source_row_data_loader(
        prepared.dataset,
        prepared.train_rows,
        batch_size=config.supervised_batch_size,
        shuffle=True,
        seed=config.seed + epoch_index,
    )
    prediction_loss_total = 0.0
    prediction_mse_total = 0.0
    mmp_total_sum = 0.0
    mmp_core_sum = 0.0
    mmp_difference_sum = 0.0
    fragment_total_sum = 0.0
    fragment_positive_sum = 0.0
    fragment_consistency_sum = 0.0
    fragment_inconsistency_sum = 0.0
    fragment_active_occurrences = 0
    graph_count = 0
    batch_count = 0
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        terms = _batch_loss_terms(
            prepared,
            batch,
            include_mmp=config.mmp_weight != 0.0,
            include_fragment=config.fragment_frequency_weight != 0.0,
        )
        prediction_loss_tensor = terms.prediction_loss
        prediction_mse_tensor = terms.prediction_mse
        mmp_core_tensor = terms.mmp_core
        mmp_difference_tensor = terms.mmp_difference
        mmp_total_tensor = (
            _scaled_mmp_objective(
                config,
                mmp_core_tensor,
                mmp_difference_tensor,
            )
            if config.mmp_weight != 0.0
            else prediction_loss_tensor.new_zeros(())
        )
        fragment_positive_tensor = terms.fragment_positive
        fragment_consistency_tensor = terms.fragment_consistency
        fragment_total_tensor = (
            _scaled_fragment_objective(
                config,
                fragment_positive_tensor,
                fragment_consistency_tensor,
            )
            if config.fragment_frequency_weight != 0.0
            else prediction_loss_tensor.new_zeros(())
        )
        fragment_inconsistency_tensor = terms.fragment_inconsistency

        objective_tensor = (
            prediction_loss_tensor
            + config.mmp_weight * mmp_total_tensor
            + config.fragment_frequency_weight * fragment_total_tensor
        )
        objective_tensor.backward()
        parameter_gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad
        ]
        # Reduce per-parameter finite flags on-device before one host check.
        missing_gradient = any(
            gradient is None for gradient in parameter_gradients
        )
        finite_gradients = torch.stack(
            [
                torch.isfinite(gradient).all()
                for gradient in parameter_gradients
                if gradient is not None
            ]
        ).all()
        if missing_gradient or not bool(finite_gradients):
            raise FloatingPointError(
                "Every trainable parameter must have a finite mini-batch "
                "gradient"
            )
        optimizer.step()
        batch_graphs = int(terms.moved.num_graphs)
        batch_count += 1
        graph_count += batch_graphs
        # Transfer every scalar loss component together after the optimizer step.
        (
            prediction_loss_value,
            prediction_mse_value,
            mmp_total_value,
            mmp_core_value,
            mmp_difference_value,
            fragment_total_value,
            fragment_positive_value,
            fragment_consistency_value,
            fragment_inconsistency_value,
        ) = torch.stack(
            (
                prediction_loss_tensor,
                prediction_mse_tensor,
                mmp_total_tensor,
                mmp_core_tensor,
                mmp_difference_tensor,
                fragment_total_tensor,
                fragment_positive_tensor,
                fragment_consistency_tensor,
                fragment_inconsistency_tensor,
            )
        ).detach().cpu().tolist()
        prediction_loss_total += batch_graphs * prediction_loss_value
        prediction_mse_total += batch_graphs * prediction_mse_value
        mmp_total_sum += mmp_total_value
        mmp_core_sum += mmp_core_value
        mmp_difference_sum += mmp_difference_value
        fragment_total_sum += fragment_total_value
        fragment_positive_sum += fragment_positive_value
        fragment_consistency_sum += fragment_consistency_value
        fragment_inconsistency_sum += fragment_inconsistency_value
        fragment_active_occurrences += terms.fragment_active_occurrences
    if graph_count != len(prepared.train_rows):
        raise AssertionError(
            "Supervised mini-batches did not cover every training graph"
        )
    prediction_loss = prediction_loss_total / graph_count
    prediction_mse = prediction_mse_total / graph_count
    mmp_total = mmp_total_sum / batch_count
    mmp_core = mmp_core_sum / batch_count
    mmp_difference = mmp_difference_sum / batch_count
    fragment_total = fragment_total_sum / batch_count
    fragment_positive = fragment_positive_sum / batch_count
    fragment_consistency = fragment_consistency_sum / batch_count
    fragment_inconsistency = fragment_inconsistency_sum / batch_count
    fragment_active_fragments = (
        sum(
            frequency > prepared.fragment_spec.min_frequency
            for frequency in prepared.fragment_spec.frequencies.values()
        )
        if config.fragment_frequency_weight != 0.0
        else 0
    )
    return {
        "optimization_prediction_loss_scaled": prediction_loss,
        "optimization_prediction_mse_scaled": prediction_mse,
        "optimization_mmp_total": mmp_total,
        "optimization_mmp_weighted": config.mmp_weight * mmp_total,
        "optimization_mmp_core": mmp_core,
        "optimization_mmp_difference": mmp_difference,
        "optimization_fragment_total": fragment_total,
        "optimization_fragment_weighted": (
            config.fragment_frequency_weight * fragment_total
        ),
        "optimization_fragment_positive": fragment_positive,
        "optimization_fragment_consistency": fragment_consistency,
        "optimization_fragment_inconsistency": fragment_inconsistency,
        "optimization_fragment_active_fragments": float(
            fragment_active_fragments
        ),
        "optimization_fragment_active_occurrences": float(
            fragment_active_occurrences
        ),
        "optimization_total": (
            prediction_loss
            + config.mmp_weight * mmp_total
            + config.fragment_frequency_weight * fragment_total
        ),
    }


def _full_objective_terms(
    prepared: _PreparedFold,
    *,
    epoch_index: int,
) -> _FullObjectiveTerms:
    """Evaluate every raw full-fold objective term in one shared forward."""

    config = prepared.config
    graph_batch = _graph_batch_for_rows(
        prepared.dataset,
        prepared.train_rows,
    )
    source_rows = tuple(
        int(row) for row in graph_batch.graph_idx.reshape(-1).tolist()
    )
    atom_offsets = tuple(int(value) for value in graph_batch.ptr.tolist())
    moved, output = _forward_batch(
        prepared.model,
        graph_batch,
        prepared.device,
    )
    targets = moved.y.reshape(-1)
    if output.prediction.shape != targets.shape:
        raise ValueError("Prediction and target batch shapes differ")
    prediction_mse = F.mse_loss(
        output.prediction,
        targets,
        reduction="mean",
    )
    prediction_loss = compute_prediction_loss(
        output.prediction,
        targets,
        mode=config.prediction_loss_mode,
        huber_delta=config.prediction_huber_delta,
    )
    contributions = _contributions_by_source_row(
        source_rows,
        atom_offsets,
        output.atom_contributions,
    )

    record_count = len(prepared.train_mmp_records)
    similarity_total = sum(
        record.similarity for record in prepared.train_mmp_records
    )
    if record_count <= 0 or similarity_total <= 0.0:
        raise ValueError("Retained MMP population must be non-empty")
    core_total = 0.0
    difference_total = 0.0
    core_objective = prediction_loss.new_zeros(())
    difference_objective = prediction_loss.new_zeros(())
    for records in iter_pair_batches(
        prepared.train_mmp_records,
        batch_size=config.mmp_pair_batch_size,
        seed=config.seed,
        epoch=epoch_index,
    ):
        mmp = compute_mmp_pair_loss(
            contributions,
            records,
            target_alpha=prepared.scaler.alpha,
            core_weight=1.0,
            difference_weight=1.0,
            huber_beta=config.mmp_huber_beta,
        )
        record_fraction = len(records) / record_count
        batch_similarity = sum(record.similarity for record in records)
        similarity_fraction = batch_similarity / similarity_total
        core_objective = core_objective + record_fraction * mmp.core
        difference_objective = (
            difference_objective
            + similarity_fraction * mmp.difference
        )
        core_total += record_fraction * float(mmp.core.detach().cpu())
        difference_total += similarity_fraction * float(
            mmp.difference.detach().cpu()
        )
    mmp_objective = _scaled_mmp_objective(
        config,
        core_objective,
        difference_objective,
    )
    mmp_total = float(mmp_objective.detach().cpu())

    fragment_total = prediction_loss.new_zeros(())
    fragment_positive = prediction_loss.new_zeros(())
    fragment_consistency = prediction_loss.new_zeros(())
    fragment_inconsistency = prediction_loss.new_zeros(())
    fragment_active_fragments = 0
    fragment_active_occurrences = 0
    if config.fragment_frequency_weight > 0.0:
        if prepared.fragment_spec is None:
            raise AssertionError("Active fragment objective has no fold spec")
        fragment = compute_fragment_frequency_loss(
            contributions,
            prepared.fragment_spec,
            positive_weight=1.0,
            consistency_weight=1.0,
        )
        fragment_positive = fragment.positive
        fragment_consistency = fragment.consistency
        fragment_total = _scaled_fragment_objective(
            config,
            fragment_positive,
            fragment_consistency,
        )
        fragment_inconsistency = fragment.inconsistency
        fragment_active_fragments = fragment.active_fragment_count
        fragment_active_occurrences = fragment.active_occurrence_count
    return _FullObjectiveTerms(
        prediction_loss=prediction_loss,
        prediction_mse=prediction_mse,
        mmp_objective=mmp_objective,
        mmp_total=mmp_total,
        mmp_core=core_total,
        mmp_difference=difference_total,
        fragment_total=fragment_total,
        fragment_positive=fragment_positive,
        fragment_consistency=fragment_consistency,
        fragment_inconsistency=fragment_inconsistency,
        fragment_active_fragments=fragment_active_fragments,
        fragment_active_occurrences=fragment_active_occurrences,
    )


def _optimization_epoch(
    prepared: _PreparedFold,
    optimizer: torch.optim.Optimizer,
    *,
    epoch_index: int,
) -> dict[str, float]:
    """Apply the selected optimizer update rule to prediction and AC losses."""

    config = prepared.config
    if config.optimization_mode == "minibatch":
        return _minibatch_optimization_epoch(
            prepared,
            optimizer,
            epoch_index=epoch_index,
        )
    if config.optimization_mode != "full_objective":
        raise AssertionError(
            f"Unreachable optimization mode: {config.optimization_mode}"
        )
    model = prepared.model
    model.train()
    optimizer.zero_grad(set_to_none=True)
    terms = _full_objective_terms(
        prepared,
        epoch_index=epoch_index,
    )
    objective_tensor = (
        terms.prediction_loss
        + config.mmp_weight * terms.mmp_objective
        + config.fragment_frequency_weight * terms.fragment_total
    )
    objective_tensor.backward()
    prediction_loss = float(terms.prediction_loss.detach().cpu())
    prediction_mse = float(terms.prediction_mse.detach().cpu())
    objective_total = float(objective_tensor.detach().cpu())
    parameter_gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    if not parameter_gradients or any(
        gradient is None or not torch.isfinite(gradient).all()
        for gradient in parameter_gradients
    ):
        raise FloatingPointError(
            "Every trainable parameter must have a finite epoch gradient"
        )
    optimizer.step()
    return {
        "optimization_prediction_loss_scaled": prediction_loss,
        "optimization_prediction_mse_scaled": prediction_mse,
        "optimization_mmp_total": terms.mmp_total,
        "optimization_mmp_weighted": (
            config.mmp_weight * terms.mmp_total
        ),
        "optimization_mmp_core": terms.mmp_core,
        "optimization_mmp_difference": terms.mmp_difference,
        "optimization_fragment_total": float(
            terms.fragment_total.detach().cpu()
        ),
        "optimization_fragment_weighted": float(
            config.fragment_frequency_weight
            * terms.fragment_total.detach().cpu()
        ),
        "optimization_fragment_positive": float(
            terms.fragment_positive.detach().cpu()
        ),
        "optimization_fragment_consistency": float(
            terms.fragment_consistency.detach().cpu()
        ),
        "optimization_fragment_inconsistency": float(
            terms.fragment_inconsistency.detach().cpu()
        ),
        "optimization_fragment_active_fragments": float(
            terms.fragment_active_fragments
        ),
        "optimization_fragment_active_occurrences": float(
            terms.fragment_active_occurrences
        ),
        "optimization_total": objective_total,
    }


def _prediction_metrics(
    model: CachedRepresentationPredictor,
    dataset: CacheBackedGraphDataset,
    source_rows: Sequence[int],
    *,
    scaler: AffineLabelScaler,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    squared_scaled: list[np.ndarray] = []
    target_raw_values: list[np.ndarray] = []
    prediction_raw_values: list[np.ndarray] = []
    loader = source_row_data_loader(
        dataset,
        source_rows,
        batch_size=batch_size,
        shuffle=False,
        seed=0,
    )
    with torch.no_grad():
        for batch in loader:
            moved, output = _forward_batch(model, batch, device)
            targets_scaled = moved.y.reshape(-1)
            targets_raw = moved.y_raw.reshape(-1)
            raw = reconstruct_affine(
                output,
                scale=scaler.alpha,
                offset=scaler.beta,
            )
            residual_scaled = output.prediction - targets_scaled
            # One batch matrix preserves each metric while avoiding repeated
            # accelerator synchronization for individual arrays.
            batch_metrics = torch.stack(
                (
                    targets_raw,
                    raw.prediction,
                    residual_scaled.square(),
                ),
                dim=1,
            ).detach().cpu().numpy()
            target_raw_values.append(batch_metrics[:, 0])
            prediction_raw_values.append(batch_metrics[:, 1])
            squared_scaled.append(batch_metrics[:, 2])
    mse_scaled = float(np.concatenate(squared_scaled).mean())
    target_raw = np.concatenate(target_raw_values)
    prediction_raw = np.concatenate(prediction_raw_values)
    metrics = compute_regression_metrics(target_raw, prediction_raw)
    return {
        "mse_scaled": mse_scaled,
        "mae_raw": metrics.mae,
        "rmse_raw": metrics.rmse,
        "r2_raw": metrics.r2_score,
    }


def _local_smiles_to_graph_mapping(
    batch: Batch,
    graph_in_batch: int,
    start: int,
    stop: int,
) -> Tensor:
    mapping = batch.smiles_to_graph_atom_index[start:stop].to(dtype=torch.long)
    # PyG increments attributes whose key contains "index". Restore the
    # molecule-local permutation explicitly before exporting atom identities.
    mapping = mapping - start
    expected = torch.arange(
        stop - start,
        dtype=torch.long,
        device=mapping.device,
    )
    if not torch.equal(torch.sort(mapping).values, expected):
        raise ValueError(
            "Batched smiles_to_graph_atom_index is not a local permutation "
            f"for graph {graph_in_batch}"
        )
    return mapping


def _evaluate_tables(
    model: CachedRepresentationPredictor,
    dataset: CacheBackedGraphDataset,
    *,
    train_rows: Sequence[int],
    validation_rows: Sequence[int],
    scaler: AffineLabelScaler,
    batch_size: int,
    device: torch.device,
) -> EvaluationResult:
    """Export predictions and raw/scaled ACs in the original endpoint atom order."""

    train_set = set(train_rows)
    validation_set = set(validation_rows)
    active_rows = tuple(dataset.source_row_indices)
    full_dataset_scope = (
        not validation_set and train_set == set(active_rows)
    )
    if train_set & validation_set:
        raise ValueError("Evaluation train and validation rows overlap")
    if train_set | validation_set != set(active_rows):
        raise ValueError(
            "Evaluation split rows must partition every eligible dataset row"
        )
    split_by_row = (
        {row: "full_training" for row in train_rows}
        if full_dataset_scope
        else {
            **{row: "train" for row in train_rows},
            **{row: "validation" for row in validation_rows},
        }
    )
    prediction_rows: list[dict[str, Any]] = []
    contribution_rows: list[dict[str, Any]] = []
    loader = source_row_data_loader(
        dataset,
        active_rows,
        batch_size=batch_size,
        shuffle=False,
        seed=0,
    )
    model.eval()
    with torch.no_grad():
        for batch in loader:
            ptr_values = tuple(int(value) for value in batch.ptr.tolist())
            source_row_values = tuple(
                int(value) for value in batch.graph_idx.reshape(-1).tolist()
            )
            dataset_position_values = tuple(
                int(value) for value in batch.cache_idx.reshape(-1).tolist()
            )
            molecule_id_values = tuple(
                int(value) for value in batch.idx.reshape(-1).tolist()
            )
            target_scaled_values = batch.y.reshape(-1).numpy()
            target_raw_values = batch.y_raw.reshape(-1).numpy()
            atomic_number_values = batch.atomic_numbers.numpy()
            smiles_to_graph_values = tuple(
                _local_smiles_to_graph_mapping(
                    batch,
                    graph_in_batch,
                    ptr_values[graph_in_batch],
                    ptr_values[graph_in_batch + 1],
                ).numpy()
                for graph_in_batch in range(batch.num_graphs)
            )
            moved, output = _forward_batch(model, batch, device)
            raw = reconstruct_affine(
                output,
                scale=scaler.alpha,
                offset=scaler.beta,
            )
            scaled_atom_sum = global_add_pool(
                output.atom_contributions,
                output.batch_index,
                size=moved.num_graphs,
            )
            scaled_residual = output.prediction - scaled_atom_sum
            if not torch.allclose(
                output.prediction,
                scaled_atom_sum,
                atol=ADDITIVITY_ABSOLUTE_TOLERANCE,
                rtol=ADDITIVITY_RELATIVE_TOLERANCE,
            ):
                raise ValueError("Scaled atom-sum additivity validation failed")
            validate_affine_additivity(
                raw,
                output.batch_index,
                absolute_tolerance=ADDITIVITY_ABSOLUTE_TOLERANCE,
                relative_tolerance=ADDITIVITY_RELATIVE_TOLERANCE,
            )
            # Move graph- and atom-level report values to CPU in two batches;
            # the row construction below is then accelerator-independent.
            graph_values = torch.stack(
                (
                    output.prediction,
                    scaled_residual,
                    raw.prediction,
                    raw.baseline.expand_as(raw.prediction),
                    raw.atom_sum,
                    raw.additivity_residual,
                ),
                dim=1,
            ).detach().cpu().numpy()
            contribution_values = torch.stack(
                (
                    output.atom_contributions,
                    raw.atom_contributions,
                ),
                dim=1,
            ).detach().cpu().numpy()
            for graph_in_batch in range(moved.num_graphs):
                source_row = source_row_values[graph_in_batch]
                dataset_position = dataset_position_values[graph_in_batch]
                molecule_id = molecule_id_values[graph_in_batch]
                frame_row = dataset.frame.iloc[source_row]
                start = ptr_values[graph_in_batch]
                stop = ptr_values[graph_in_batch + 1]
                atom_count = stop - start
                split = split_by_row[source_row]
                target_raw = float(target_raw_values[graph_in_batch])
                (
                    prediction_scaled,
                    additivity_residual_scaled,
                    prediction_raw,
                    affine_baseline_raw,
                    atom_sum_raw,
                    additivity_residual_raw,
                ) = (
                    float(value) for value in graph_values[graph_in_batch]
                )
                prediction_rows.append(
                    {
                        "source_row_index": source_row,
                        "dataset_position": dataset_position,
                        "molecule_id": molecule_id,
                        "sample_id": str(
                            frame_row[dataset.config.sample_id_column]
                        ),
                        "smiles": str(
                            frame_row[dataset.config.smiles_column]
                        ),
                        "split": split,
                        "n_atoms": atom_count,
                        "target_scaled": float(
                            target_scaled_values[graph_in_batch]
                        ),
                        "prediction_scaled": prediction_scaled,
                        "additivity_residual_scaled": (
                            additivity_residual_scaled
                        ),
                        "target_raw": target_raw,
                        "prediction_raw": prediction_raw,
                        "affine_baseline_raw": affine_baseline_raw,
                        "atom_sum_raw": atom_sum_raw,
                        "additivity_residual_raw": additivity_residual_raw,
                        "absolute_error_raw": abs(
                            prediction_raw - target_raw
                        ),
                        "squared_error_raw": (
                            prediction_raw - target_raw
                        )
                        ** 2,
                    }
                )

                smiles_to_graph = smiles_to_graph_values[graph_in_batch]
                graph_to_smiles = np.empty_like(smiles_to_graph)
                graph_to_smiles[smiles_to_graph] = np.arange(
                    atom_count,
                    dtype=np.int64,
                )
                numbers = atomic_number_values[start:stop]
                scaled_contributions = contribution_values[start:stop, 0]
                raw_contributions = contribution_values[start:stop, 1]
                for graph_atom_index in range(atom_count):
                    contribution_rows.append(
                        {
                            "source_row_index": source_row,
                            "dataset_position": dataset_position,
                            "molecule_id": molecule_id,
                            "sample_id": str(
                                frame_row[dataset.config.sample_id_column]
                            ),
                            "smiles": str(
                                frame_row[dataset.config.smiles_column]
                            ),
                            "split": split,
                            "graph_atom_index": graph_atom_index,
                            "smiles_atom_index": int(
                                graph_to_smiles[graph_atom_index]
                            ),
                            "atomic_number": int(numbers[graph_atom_index]),
                            "atom_contribution_scaled": float(
                                scaled_contributions[graph_atom_index]
                            ),
                            "atom_contribution_raw": float(
                                raw_contributions[graph_atom_index]
                            ),
                        }
                    )

    predictions = pd.DataFrame(prediction_rows, columns=PREDICTION_COLUMNS)
    atom_contributions = pd.DataFrame(
        contribution_rows,
        columns=ATOM_CONTRIBUTION_COLUMNS,
    )
    predictions = predictions.sort_values(
        "source_row_index",
        kind="stable",
    ).reset_index(drop=True)
    atom_contributions = atom_contributions.sort_values(
        ["source_row_index", "graph_atom_index"],
        kind="stable",
    ).reset_index(drop=True)
    metrics: dict[str, dict[str, float]] = {}
    evaluation_splits = (
        ("full_training",)
        if full_dataset_scope
        else ("train", "validation")
    )
    for split in evaluation_splits:
        rows = predictions[predictions["split"] == split]
        if rows.empty:
            raise ValueError(f"Evaluation split is empty: {split}")
        raw_metrics = compute_regression_metrics(
            rows["target_raw"].to_numpy(),
            rows["prediction_raw"].to_numpy(),
        )
        scaled_metrics = compute_regression_metrics(
            rows["target_scaled"].to_numpy(),
            rows["prediction_scaled"].to_numpy(),
        )
        metrics[split] = {
            "mse_scaled": scaled_metrics.mse,
            "mae_raw": raw_metrics.mae,
            "rmse_raw": raw_metrics.rmse,
            "max_abs_additivity_residual_scaled": float(
                rows["additivity_residual_scaled"].abs().max()
            ),
            "max_abs_additivity_residual_raw": float(
                rows["additivity_residual_raw"].abs().max()
            ),
        }
    return EvaluationResult(
        predictions=predictions,
        atom_contributions=atom_contributions,
        metrics=metrics,
    )


def _checkpoint_payload(
    prepared: _PreparedFold,
    *,
    state_dict: Mapping[str, Tensor],
    best_epoch: int,
    best_validation_mse_scaled: float,
) -> dict[str, Any]:
    """Save the model state with its scaler and scientific reconstruction settings."""

    selection_criterion = _checkpoint_selection_criterion(prepared.config)
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_purpose": "selected_fold_inference",
        "checkpoint_selection": prepared.config.checkpoint_selection,
        "selection_uses_validation_labels": (
            selection_criterion != _FINAL_EPOCH_CRITERION
        ),
        "model_state_dict": dict(state_dict),
        "backbone_config": asdict(prepared.backbone_config),
        "readout_config": {
            "hidden_dim": prepared.config.readout_hidden_dim,
            "num_layers": prepared.config.readout_num_layers,
            "atom_bias": prepared.config.atom_bias,
            "activation_contract": (
                prepared.model.readout.activation_contract
            ),
        },
        "scaler": {
            "mode": prepared.scaler.mode,
            "beta": prepared.scaler.beta,
            "alpha": prepared.scaler.alpha,
        },
        "best_epoch": best_epoch,
        "best_validation_mse_scaled": best_validation_mse_scaled,
        "train_rows": list(prepared.train_rows),
        "validation_rows": list(prepared.validation_rows),
        "excluded_rows": list(prepared.excluded_rows),
        "training_config": prepared.config.as_dict(),
    }


def load_fold_checkpoint(
    checkpoint_path: Path | str,
    *,
    device: str,
) -> LoadedFoldCheckpoint:
    """Load a strict selected-fold checkpoint without compatibility fallback."""

    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    resolved_device = _resolve_device(device)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as exc:
        raise RuntimeError(
            "secure weights-only checkpoint loading is required; "
            "use a PyTorch version that supports weights_only=True"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint must contain a dictionary")
    required = {
        "schema_version",
        "checkpoint_purpose",
        "checkpoint_selection",
        "selection_uses_validation_labels",
        "model_state_dict",
        "backbone_config",
        "readout_config",
        "scaler",
        "best_epoch",
        "best_validation_mse_scaled",
        "train_rows",
        "validation_rows",
        "excluded_rows",
        "training_config",
    }
    if set(payload) != required:
        raise ValueError("Checkpoint fields do not match the strict schema")
    if payload["schema_version"] not in {2, 3, CHECKPOINT_SCHEMA_VERSION}:
        raise ValueError("Checkpoint schema version is unsupported")
    if payload["checkpoint_purpose"] != "selected_fold_inference":
        raise ValueError("Checkpoint purpose is invalid")
    selection = payload["checkpoint_selection"]
    if selection not in {"best_validation", "fixed_final_epoch"}:
        raise ValueError("Checkpoint selection policy is invalid")
    if payload["selection_uses_validation_labels"] is not (
        selection == "best_validation"
    ):
        raise ValueError(
            "Checkpoint validation-label selection declaration is invalid"
        )
    backbone_payload = dict(payload["backbone_config"])
    if payload["schema_version"] < CHECKPOINT_SCHEMA_VERSION:
        backbone_payload["gated_residual"] = False
        backbone_payload.setdefault("normalization", "layernorm")
    backbone = BackboneConfig(**backbone_payload)
    readout = payload["readout_config"]
    if payload["schema_version"] == 2:
        if set(readout) != {"hidden_dim", "atom_bias"}:
            raise ValueError("Checkpoint readout_config fields are invalid")
        readout_num_layers = None
        readout_activation_contract = "gelu_hidden_v1"
    elif payload["schema_version"] == 3:
        if set(readout) != {"hidden_dim", "num_layers", "atom_bias"}:
            raise ValueError("Checkpoint readout_config fields are invalid")
        readout_num_layers = readout["num_layers"]
        readout_activation_contract = "gelu_hidden_v1"
    else:
        if set(readout) != {
            "hidden_dim",
            "num_layers",
            "atom_bias",
            "activation_contract",
        }:
            raise ValueError("Checkpoint readout_config fields are invalid")
        readout_num_layers = readout["num_layers"]
        readout_activation_contract = readout["activation_contract"]
    model = CachedRepresentationPredictor(
        backbone,
        readout_hidden_dim=readout["hidden_dim"],
        readout_num_layers=readout_num_layers,
        atom_bias=readout["atom_bias"],
        readout_activation_contract=readout_activation_contract,
    )
    if payload["schema_version"] < CHECKPOINT_SCHEMA_VERSION:
        # Schema v2/v3 checkpoints were trained before the explicit
        # architecture contract and used GELU in the graph backbone.
        model.backbone.activation = torch.nn.GELU()
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(resolved_device).eval()
    scaler_payload = payload["scaler"]
    if not isinstance(scaler_payload, Mapping) or set(scaler_payload) != {
        "mode",
        "beta",
        "alpha",
    }:
        raise ValueError("Checkpoint scaler fields are invalid")
    scaler = AffineLabelScaler(**scaler_payload)
    return LoadedFoldCheckpoint(
        model=model,
        scaler=scaler,
        metadata=payload,
        device=resolved_device,
    )


def _provenance_payload(
    prepared: _PreparedFold,
    *,
    best_epoch: int,
    best_validation_mse_scaled: float,
    best_validation_r2_raw: float | None,
    epochs_completed: int,
    stop_reason: str,
    evaluation: EvaluationResult,
) -> dict[str, Any]:
    selection_criterion = _checkpoint_selection_criterion(prepared.config)
    device_payload: dict[str, Any] = {"resolved": str(prepared.device)}
    if prepared.device.type == "cuda":
        device_payload["name"] = torch.cuda.get_device_name(prepared.device)
        device_payload["capability"] = list(
            torch.cuda.get_device_capability(prepared.device)
        )
    targets = pd.to_numeric(
        prepared.dataset.frame[prepared.config.data.target_column],
        errors="raise",
    ).to_numpy(dtype=np.float64)
    excluded_target_values = targets[list(prepared.excluded_rows)]
    eligible_target_values = targets[list(prepared.dataset.source_row_indices)]

    def target_summary(
        values: np.ndarray,
    ) -> dict[str, float | int | None]:
        if values.size == 0:
            return {
                "count": 0,
                "mean": None,
                "minimum": None,
                "maximum": None,
            }
        return {
            "count": int(values.size),
            "mean": float(values.mean()),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
        }

    excluded_set = set(prepared.excluded_rows)
    per_fold_exclusion_counts = {
        str(fold): sum(row in excluded_set for row in rows)
        for fold, rows in sorted(prepared.fold_indices.items())
    }
    has_gradient_calibration = (
        prepared.config.loss_gradient_scaling.method
        == "train_only_gradient_rms_v1"
    )
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        RESULT_SCOPE_KEY: (
            "production_full_dataset_fit_no_internal_validation"
            if prepared.config.training_scope == "full_dataset"
            else (
                "production_full_analysis"
                if prepared.config.run_profile in PRODUCTION_RUN_PROFILES
                else "exploratory_or_nonfinal"
            )
        ),
        "objective": {
            "formula": (
                "L = L_prediction_scaled + "
                "mmp_weight * "
                "(mmp_core_weight * mmp_core_scale * L_core + "
                "mmp_difference_weight * mmp_difference_scale * "
                "L_difference) / "
                "(mmp_core_weight + mmp_difference_weight) + "
                "fragment_frequency_weight * "
                "(fragment_positive_weight * fragment_positive_scale * "
                "L_positive + fragment_consistency_weight * "
                "fragment_consistency_scale * L_consistency)"
            ),
            "loss_gradient_scaling": (
                prepared.config.loss_gradient_scaling.as_dict()
            ),
            "calibration_distribution_gradient_ratio_upper_bounds": {
                "scope": (
                    "train-only fixed-model distribution after warm-up"
                ),
                "meaning": (
                    "outer-weighted group bound during calibration; "
                    "not an epoch-level observed gradient ratio"
                ),
                "mmp_group": (
                    prepared.config.mmp_weight
                    if has_gradient_calibration
                    else None
                ),
                "fragment_group": (
                    prepared.config.fragment_frequency_weight
                    * (
                        prepared.config.fragment_positive_weight
                        + prepared.config.fragment_consistency_weight
                    )
                    if has_gradient_calibration
                    else None
                ),
            },
            "prediction_loss_mode": prepared.config.prediction_loss_mode,
            "prediction_huber_delta": (
                prepared.config.prediction_huber_delta
                if prepared.config.prediction_loss_mode == "huber"
                else None
            ),
            "mmp_target_difference": "dy_scaled = dy_raw / scaler.alpha",
            "fragment_frequency": {
                "weight": prepared.config.fragment_frequency_weight,
                "positive_weight": prepared.config.fragment_positive_weight,
                "consistency_weight": (
                    prepared.config.fragment_consistency_weight
                ),
                "min_heavy_atoms": (
                    prepared.config.fragment_min_heavy_atoms
                ),
                "min_count": prepared.config.fragment_min_count,
                "min_frequency": prepared.config.fragment_min_frequency,
                "reference_split": (
                    "full_training_dataset"
                    if prepared.config.training_scope == "full_dataset"
                    else "fold_train"
                ),
                "optimization_occurrence_scope": (
                    "current shuffled molecule minibatch"
                    if prepared.config.optimization_mode == "minibatch"
                    else "all active training molecules"
                ),
                "positive_margin": (
                    prepared.fragment_spec.positive_margin
                    if prepared.fragment_spec is not None
                    else None
                ),
                "retained_fragment_count": (
                    len(prepared.fragment_spec.frequencies)
                    if prepared.fragment_spec is not None
                    else 0
                ),
                "active_fragment_count": (
                    sum(
                        frequency
                        > prepared.fragment_spec.min_frequency
                        for frequency in (
                            prepared.fragment_spec.frequencies.values()
                        )
                    )
                    if prepared.fragment_spec is not None
                    else 0
                ),
            },
            "optimizer_updates_per_epoch": (
                math.ceil(
                    len(prepared.train_rows)
                    / prepared.config.supervised_batch_size
                )
                if prepared.config.optimization_mode == "minibatch"
                else 1
            ),
            "optimization_graph_forward": (
                "one PyG Batch per shuffled molecule minibatch"
                if prepared.config.optimization_mode == "minibatch"
                else "one PyG Batch containing every active train row"
            ),
            "pair_batch_partition_weighting": {
                "role": (
                    "all MMPs induced by each molecule minibatch; no pair cap "
                    "or population correction"
                    if prepared.config.optimization_mode == "minibatch"
                    else "loss-computation chunks only; all chunks reuse the "
                    "same single-forward atom contributions"
                ),
                "core": (
                    "mean over induced minibatch pairs"
                    if prepared.config.optimization_mode == "minibatch"
                    else "pair_count_in_batch / total_pair_count"
                ),
                "difference": (
                    "similarity-weighted mean over induced minibatch pairs"
                    if prepared.config.optimization_mode == "minibatch"
                    else "similarity_sum_in_batch / total_similarity_sum"
                ),
            },
        },
        "target_scaler": {
            "mode": prepared.scaler.mode,
            "fit_rows": list(prepared.train_rows),
            "beta": prepared.scaler.beta,
            "alpha": prepared.scaler.alpha,
            "formula": "z = (y - beta) / alpha",
            "raw_reconstruction": (
                "prediction_raw = beta + "
                "sum(alpha * atom_contribution_scaled)"
            ),
            "baseline_policy": (
                "beta remains one graph-level baseline and is not assigned "
                "to atoms"
            ),
        },
        "split": {
            "training_scope": prepared.config.training_scope,
            "validation_fold": (
                None
                if prepared.config.training_scope == "full_dataset"
                else prepared.config.validation_fold
            ),
            "train_rows": list(prepared.train_rows),
            "validation_rows": list(prepared.validation_rows),
            "excluded_rows": list(prepared.excluded_rows),
            "eligible_row_count": len(prepared.dataset),
            "original_row_count": len(prepared.dataset.frame),
            "estimand_warning": (
                "All fitted and evaluated results target the "
                f"{len(prepared.dataset)}-molecule eligibility estimand, "
                f"not the original {len(prepared.dataset.frame)}-molecule "
                "dataset."
            ),
            "explicit_hydrogen_exclusions_by_declared_fold": (
                per_fold_exclusion_counts
            ),
            "excluded_target_summary": target_summary(
                excluded_target_values
            ),
            "eligible_target_summary": target_summary(
                eligible_target_values
            ),
        },
        "mmp_accounting": {
            "source_pair_count": len(prepared.all_mmp_records),
            "explicit_hydrogen_incident_pair_count": len(
                prepared.hydrogen_excluded_pair_ids
            ),
            "explicit_hydrogen_incident_pair_ids": list(
                prepared.hydrogen_excluded_pair_ids
            ),
            "input_contract_incident_pair_count": len(
                prepared.input_excluded_pair_ids
            ),
            "input_contract_incident_pair_ids": list(
                prepared.input_excluded_pair_ids
            ),
            "fold_non_train_pair_count_after_input_exclusion": len(
                prepared.validation_excluded_pair_ids
            ),
            "fold_non_train_pair_ids_after_input_exclusion": list(
                prepared.validation_excluded_pair_ids
            ),
            "fold_non_train_pair_count_after_hydrogen_exclusion": len(
                prepared.validation_excluded_pair_ids
            ),
            "fold_non_train_pair_ids_after_hydrogen_exclusion": list(
                prepared.validation_excluded_pair_ids
            ),
            "retained_train_pair_count": len(prepared.train_mmp_records),
            "retained_train_pair_ids": [
                record.pair_id for record in prepared.train_mmp_records
            ],
            "chemical_atom_mapping": dict(prepared.mmp_mapping_check),
            "dy_absolute_tolerance": MMP_DY_ABSOLUTE_TOLERANCE,
            "retained_pair_validation": [
                "dataset endpoints",
                "endpoint SMILES",
                "graph filenames",
                "dy orientation and value",
                "atom-index ranges",
                "core/difference exact partition",
                "mapped-core element equality",
                (
                    "SMILES-order MMP indices mapped explicitly to graph "
                    "atom order"
                ),
            ],
        },
        "inputs": {
            "dataset_csv": _portable_path_record(
                prepared.config.data.dataset_csv
            ),
            "fold_csv": _portable_path_record(prepared.config.data.fold_csv),
            "mmp_csv": _portable_path_record(prepared.config.mmp_csv),
            "fragment_cache": (
                None
                if prepared.config.fragment_cache is None
                else {
                    **_portable_path_record(
                        prepared.config.fragment_cache
                    ),
                    "selection_policy": (
                        "reuse cached graph-aligned BRICS occurrences; "
                        "refit frequencies and positive margin on exact "
                        "atom-representation-eligible training rows"
                    ),
                }
            ),
            "representation_cache": {
                **_portable_path_record(
                    prepared.config.data.representation_cache
                ),
                "encoder": prepared.dataset.cache.metadata["provider"][
                    "provider"
                ],
                "molecule_count": prepared.dataset.cache.metadata[
                    "molecule_count"
                ],
                "embedding_dim": prepared.dataset.cache.metadata[
                    "embedding_dim"
                ],
                "geometry_quality": dict(prepared.cache_quality_check),
            },
        },
        "model": {
            "backbone": asdict(prepared.backbone_config),
            "readout_hidden_dim": prepared.config.readout_hidden_dim,
            "atom_bias": prepared.config.atom_bias,
            "trainable_parameter_count": sum(
                parameter.numel()
                for parameter in prepared.model.parameters()
                if parameter.requires_grad
            ),
        },
        "runtime": {
            "seed": prepared.config.seed,
            "device": device_payload,
            "python": platform.python_version(),
            "packages": {
                "numpy": _package_version("numpy"),
                "pandas": _package_version("pandas"),
                "torch": _package_version("torch"),
                "torch-geometric": _package_version("torch-geometric"),
                "rdkit": _package_version("rdkit"),
            },
            "data_loader_num_workers": 0,
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "torch_num_threads": torch.get_num_threads(),
        },
        "selection": {
            "checkpoint_selection": prepared.config.checkpoint_selection,
            "selection_uses_validation_labels": (
                selection_criterion != _FINAL_EPOCH_CRITERION
            ),
            "criterion": _CHECKPOINT_CRITERION_DESCRIPTIONS[
                selection_criterion
            ],
            "best_epoch": best_epoch,
            "best_validation_mse_scaled": best_validation_mse_scaled,
            "best_validation_r2_raw": best_validation_r2_raw,
            "epochs_completed": epochs_completed,
            "stop_reason": stop_reason,
            "early_stopping_patience": (
                prepared.config.early_stopping_patience
            ),
            "validation_metrics_role": (
                "in-sample diagnostic only; no held-out validation split"
                if prepared.config.training_scope == "full_dataset"
                else (
                    "monitoring only; not used to choose weights"
                    if selection_criterion == _FINAL_EPOCH_CRITERION
                    else "used to choose checkpoint weights"
                )
            ),
        },
        "additivity_validation": {
            "absolute_tolerance": ADDITIVITY_ABSOLUTE_TOLERANCE,
            "relative_tolerance": ADDITIVITY_RELATIVE_TOLERANCE,
            "raw_relative_error_scale": (
                "abs(prediction_raw) + abs(affine_baseline_raw) + "
                "sum(abs(atom_contribution_raw))"
            ),
            "metrics": {
                split: {
                    key: value
                    for key, value in values.items()
                    if "additivity_residual" in key
                }
                for split, values in evaluation.metrics.items()
            },
            "interpretation": (
                "atom contributions are predictive decomposition terms, "
                "not physical observables"
            ),
        },
        "evaluation_metrics": {
            split: dict(values)
            for split, values in evaluation.metrics.items()
        },
    }


def _result_record(path: Path, *, columns: Sequence[str] | None = None) -> dict:
    record: dict[str, Any] = {
        "filename": path.name,
    }
    if columns is not None:
        record["columns"] = list(columns)
    return record


def run_fold_training(
    config: TrainingConfig,
    *,
    epoch_observer: Callable[[Mapping[str, Any]], None] | None = None,
    raw_dataset: CacheBackedGraphDataset | None = None,
) -> TrainingResults:
    """Train one fold, restore its selected model, and export results.

    Input validation, all retained MMP relationship validation, and output
    collision checks complete before the output directory is created. There is
    no legacy checkpoint conversion, cache fallback, device fallback, or
    silent MMP-pair skip.  ``epoch_observer`` receives one immutable copy of
    the just-completed history row.  It may persist progress, report pruning,
    or raise to interrupt the run; omitting it preserves the historical
    training path. ``raw_dataset`` shares validated unscaled graph tensors
    across folds while each fold retains its own fitted label scaler.
    """

    if not isinstance(config, TrainingConfig):
        raise TypeError("config must be a TrainingConfig")
    if epoch_observer is not None and not callable(epoch_observer):
        raise TypeError("epoch_observer must be callable or None")
    seed_training_runtime(config.seed, deterministic_cudnn=True)
    prepared = _prepare_fold(config, raw_dataset=raw_dataset)
    output_dir = Path(config.output_dir)
    output_dir.mkdir()
    config_path = write_json(
        output_dir / "training_config.json",
        config.as_dict(),
    )

    optimizer = torch.optim.AdamW(
        prepared.model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    history_rows: list[dict[str, Any]] = []
    best_epoch = -1
    best_validation_mse = math.inf
    best_validation_r2 = -math.inf
    best_state: dict[str, Tensor] | None = None
    selection_criterion = _checkpoint_selection_criterion(config)
    uses_validation_for_selection = (
        selection_criterion != _FINAL_EPOCH_CRITERION
    )
    uses_validation_r2_for_selection = (
        selection_criterion == _VALIDATION_R2_CRITERION
    )
    epochs_without_improvement = 0
    stop_reason: str | None = None
    epoch_index = 0
    while epoch_index < config.epochs:
        optimization = _optimization_epoch(
            prepared,
            optimizer,
            epoch_index=epoch_index,
        )
        training = _prediction_metrics(
            prepared.model,
            prepared.dataset,
            prepared.train_rows,
            scaler=prepared.scaler,
            batch_size=config.supervised_batch_size,
            device=prepared.device,
        )
        validation = (
            dict(training)
            if config.training_scope == "full_dataset"
            else _prediction_metrics(
                prepared.model,
                prepared.dataset,
                prepared.validation_rows,
                scaler=prepared.scaler,
                batch_size=config.supervised_batch_size,
                device=prepared.device,
            )
        )
        validation_r2 = float(validation["r2_raw"])
        if uses_validation_r2_for_selection and not math.isfinite(
            validation_r2
        ):
            raise FloatingPointError(
                "Validation regression R2 must be finite for "
                "best_validation selection"
            )
        reached_epoch_limit = epoch_index + 1 == config.epochs
        if uses_validation_for_selection:
            is_best = (
                validation_r2 > best_validation_r2
                if uses_validation_r2_for_selection
                else validation["mse_scaled"] < best_validation_mse
            )
        else:
            is_best = reached_epoch_limit
        if is_best:
            best_epoch = epoch_index + 1
            best_validation_mse = validation["mse_scaled"]
            best_validation_r2 = validation_r2
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in prepared.model.state_dict().items()
            }
            epochs_without_improvement = 0
        elif uses_validation_for_selection:
            epochs_without_improvement += 1
        history_row = {
            "epoch": epoch_index + 1,
            **optimization,
            "training_mae_raw": training["mae_raw"],
            "training_rmse_raw": training["rmse_raw"],
            "training_r2_raw": training["r2_raw"],
            "validation_mse_scaled": validation["mse_scaled"],
            "validation_mae_raw": validation["mae_raw"],
            "validation_rmse_raw": validation["rmse_raw"],
            "validation_r2_raw": validation["r2_raw"],
            "is_best": is_best,
        }
        history_rows.append(history_row)
        if epoch_observer is not None:
            epoch_observer(dict(history_row))
        if (
            config.early_stopping_patience is not None
            and epochs_without_improvement
            >= config.early_stopping_patience
        ):
            stop_reason = "validation_r2_no_improvement"
            break
        if reached_epoch_limit:
            stop_reason = "epoch_limit_reached"
            break
        epoch_index += 1
    if best_state is None or best_epoch < 1 or not math.isfinite(
        best_validation_mse
    ):
        raise RuntimeError("Training produced no finite selected checkpoint")
    if stop_reason is None:
        raise RuntimeError("Training ended without a declared stop reason")

    prepared.model.load_state_dict(best_state, strict=True)
    checkpoint_path = output_dir / "best_checkpoint.pt"
    torch.save(
        _checkpoint_payload(
            prepared,
            state_dict=best_state,
            best_epoch=best_epoch,
            best_validation_mse_scaled=best_validation_mse,
        ),
        checkpoint_path,
    )
    history = pd.DataFrame(history_rows, columns=HISTORY_COLUMNS)
    history_path = output_dir / "history.csv"
    history.to_csv(history_path, index=False)

    evaluation = _evaluate_tables(
        prepared.model,
        prepared.dataset,
        train_rows=prepared.train_rows,
        validation_rows=prepared.validation_rows,
        scaler=prepared.scaler,
        batch_size=config.supervised_batch_size,
        device=prepared.device,
    )
    predictions_path = output_dir / "predictions.csv"
    atom_contributions_path = output_dir / "atom_contributions.csv"
    evaluation.predictions.to_csv(predictions_path, index=False)
    evaluation.atom_contributions.to_csv(
        atom_contributions_path,
        index=False,
    )
    provenance_path = write_json(
        output_dir / "provenance.json",
        _provenance_payload(
            prepared,
            best_epoch=best_epoch,
            best_validation_mse_scaled=best_validation_mse,
            best_validation_r2_raw=(
                best_validation_r2
                if math.isfinite(best_validation_r2)
                else None
            ),
            epochs_completed=len(history_rows),
            stop_reason=stop_reason,
            evaluation=evaluation,
        ),
    )
    result_records = {
        "training_config": _result_record(config_path),
        "provenance": _result_record(provenance_path),
        "history": _result_record(history_path, columns=HISTORY_COLUMNS),
        "checkpoint": _result_record(checkpoint_path),
        "predictions": _result_record(
            predictions_path,
            columns=PREDICTION_COLUMNS,
        ),
        "atom_contributions": _result_record(
            atom_contributions_path,
            columns=ATOM_CONTRIBUTION_COLUMNS,
        ),
    }
    result_metadata_path = write_json(
        output_dir / RESULT_METADATA_FILENAME,
        {
            "schema_version": RESULT_SCHEMA_VERSION,
            RESULT_RECORDS_KEY: result_records,
        },
    )
    complete_path = output_dir / "COMPLETE"
    complete_path.write_text("complete\n", encoding="ascii")
    return TrainingResults(
        output_dir=output_dir,
        training_config=config_path,
        provenance=provenance_path,
        history=history_path,
        checkpoint=checkpoint_path,
        predictions=predictions_path,
        atom_contributions=atom_contributions_path,
        result_metadata=result_metadata_path,
        complete=complete_path,
        best_epoch=best_epoch,
        best_validation_mse_scaled=best_validation_mse,
    )
