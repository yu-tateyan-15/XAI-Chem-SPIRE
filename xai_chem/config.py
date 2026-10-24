"""Config for notebook-based atom-contribution experiments."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from .dataset import BOND_FEATURE_DIM
from .dataset import DataConfig


XAI_CHEM_ROOT = Path(__file__).resolve().parents[1]


MMP_CORE_MAPPING_METHOD = (
    "rdkit_canonical_fragment_smiles_atom_output_order_v1"
)


DEFAULT_CHEMBL3884_260729_MMP_CSV = XAI_CHEM_ROOT / "notebooks_for_sglt2/output/sglt2/training_mmp.csv"


CHECKED_CHEMBL3884_260729_FOLD_CONTRACTS = {
    0: {
        "expected_train_rows": 1005,
        "expected_validation_explicit_hydrogen_exclusions": 0,
        "expected_validation_input_exclusions": 0,
        "expected_retained_mmp_pairs": 15186,
    },
    1: {
        "expected_train_rows": 1006,
        "expected_validation_explicit_hydrogen_exclusions": 0,
        "expected_validation_input_exclusions": 0,
        "expected_retained_mmp_pairs": 15042,
    },
    2: {
        "expected_train_rows": 1008,
        "expected_validation_explicit_hydrogen_exclusions": 0,
        "expected_validation_input_exclusions": 0,
        "expected_retained_mmp_pairs": 14507,
    },
    3: {
        "expected_train_rows": 1008,
        "expected_validation_explicit_hydrogen_exclusions": 0,
        "expected_validation_input_exclusions": 0,
        "expected_retained_mmp_pairs": 14596,
    },
    4: {
        "expected_train_rows": 1009,
        "expected_validation_explicit_hydrogen_exclusions": 0,
        "expected_validation_input_exclusions": 0,
        "expected_retained_mmp_pairs": 14262,
    },
}


CHECKED_FULL_OBJECTIVE_SETTINGS = {
    "optimization_mode": "full_objective",
    "checkpoint_selection": "fixed_final_epoch",
    "epochs": 200,
    "early_stopping_patience": None,
    "supervised_batch_size": 128,
    "mmp_pair_batch_size": 128,
    "learning_rate": 2.5e-4,
    "weight_decay": 1.0e-4,
    "scaling_mode": "standard",
    "prediction_loss_mode": "huber",
    "prediction_huber_delta": 0.5,
    "mmp_weight": 5.0,
    "mmp_core_weight": 3.0,
    "mmp_difference_weight": 1.0,
    "fragment_frequency_weight": 0.01,
    "fragment_positive_weight": 1.0,
    "fragment_consistency_weight": 1.0,
    "fragment_min_heavy_atoms": 1,
    "fragment_min_count": 3,
    "fragment_min_frequency": 0.1,
    "backbone_name": "gatv2",
    "hidden_dim": 384,
    "num_layers": 2,
    "dropout": 0.20,
    "heads": 4,
    "use_edge_features": True,
    "backbone_residual": False,
    "backbone_normalization": "layernorm",
    "readout_hidden_dim": 128,
    "readout_num_layers": None,
    "atom_bias": True,
    "output_head_init_std": None,
}


PRODUCTION_RUN_PROFILES = frozenset(
    {"checked_chembl3884_260729_full_objective"}
)


@dataclass(frozen=True)
class LossGradientScaling:
    """Fixed train-only gradient scales for one fold's loss components."""

    method: str = "identity"
    parameter_scope: str = "all_trainable_parameters"
    absent_component_batch_policy: str = "zero_gradient"
    warmup_epochs: int = 0
    batch_seeds: tuple[int, ...] = ()
    training_row_count: int = 0
    trainable_parameter_count: int = 0
    batch_count: int = 0
    mmp_active_batch_count: int = 0
    fragment_active_batch_count: int = 0
    prediction_gradient_rms: float = 1.0
    mmp_core_gradient_rms: float = 1.0
    mmp_difference_gradient_rms: float = 1.0
    fragment_positive_gradient_rms: float = 1.0
    fragment_consistency_gradient_rms: float = 1.0
    mmp_core_scale: float = 1.0
    mmp_difference_scale: float = 1.0
    fragment_positive_scale: float = 1.0
    fragment_consistency_scale: float = 1.0

    def __post_init__(self) -> None:
        """Reject non-finite or non-positive scales that break optimization."""

        object.__setattr__(self, "batch_seeds", tuple(self.batch_seeds))
        gradient_values = (
            self.prediction_gradient_rms,
            self.mmp_core_gradient_rms,
            self.mmp_difference_gradient_rms,
            self.fragment_positive_gradient_rms,
            self.fragment_consistency_gradient_rms,
        )
        scale_values = (
            self.mmp_core_scale,
            self.mmp_difference_scale,
            self.fragment_positive_scale,
            self.fragment_consistency_scale,
        )
        if not all(math.isfinite(float(value)) for value in gradient_values):
            raise ValueError("Loss gradient RMS values must be finite")
        if not all(
            math.isfinite(float(value)) and float(value) > 0.0
            for value in scale_values
        ):
            raise ValueError("Loss gradient scales must be finite and positive")

    def as_dict(self) -> dict[str, Any]:
        """Return the complete JSON-compatible calibration record."""

        payload = asdict(self)
        payload["batch_seeds"] = list(self.batch_seeds)
        return payload


def serialize_output_path(path: Path | str) -> tuple[str, str]:
    """Serialize a workspace path portably or label an external absolute."""

    resolved = Path(path).expanduser().resolve()
    root = XAI_CHEM_ROOT.resolve()
    # NOTE: CONTRACT in=path: absolute path under XAI_CHEM_ROOT is canonicalized
    #       fail-fast if path is outside project root
    if resolved.is_relative_to(root):
        return resolved.relative_to(root).as_posix(), "XAI_CHEM_ROOT_relative"
    return str(resolved), "external_absolute"


@dataclass(frozen=True)
class TrainingConfig:
    """All scientific and runtime choices for one fold training run.

    In ``full_objective`` mode, the optimized epoch objective is

    ``L = L_prediction + mmp_weight * L_mmp
    + fragment_frequency_weight * L_fragment``,

    where ``L_prediction`` is the explicitly selected MSE or Huber loss in
    the scaled target space. ``L_mmp`` is the normalized weighted sum of the
    fixed-gradient-scaled core and difference components. ``L_fragment`` is
    the weighted sum of the fixed-gradient-scaled positive and consistency
    components; Stage B supplies complementary component weights.
    ``minibatch`` mode evaluates the same normalized loss components on each
    shuffled molecule batch and performs one combined optimizer update.
    """

    data: DataConfig
    mmp_csv: Path
    output_dir: Path
    device: str
    fragment_cache: Path | None = None
    run_profile: str = "strict_lightweight"
    training_scope: str = "cross_validation_fold"
    optimization_mode: str = "full_objective"
    checkpoint_selection: str = "best_validation"
    validation_fold: int = 0
    seed: int = 42
    epochs: int = 200
    early_stopping_patience: int | None = None
    supervised_batch_size: int = 64
    mmp_pair_batch_size: int = 64
    learning_rate: float = 1.0e-3
    weight_decay: float = 0.0
    scaling_mode: str = "standard"
    prediction_loss_mode: str = "mse"
    prediction_huber_delta: float = 0.5
    mmp_weight: float = 1.0
    mmp_core_weight: float = 5.0
    mmp_difference_weight: float = 1.0
    mmp_huber_beta: float = 0.25
    fragment_frequency_weight: float = 0.0
    fragment_positive_weight: float = 1.0
    fragment_consistency_weight: float = 1.0
    loss_gradient_scaling: LossGradientScaling = field(
        default_factory=LossGradientScaling
    )
    fragment_min_heavy_atoms: int = 1
    fragment_min_count: int = 3
    fragment_min_frequency: float = 0.1
    backbone_name: str = "gatv2"
    hidden_dim: int = 128
    num_layers: int = 3
    dropout: float = 0.0
    heads: int | None = None
    use_edge_features: bool = True
    backbone_residual: bool = False
    backbone_normalization: str = "layernorm"
    architecture_contract: str = "gated_residual_tanh_v1"
    readout_hidden_dim: int | None = None
    readout_num_layers: int | None = None
    atom_bias: bool = True
    output_head_init_std: float | None = None
    expected_explicit_hydrogen_exclusions: int = 0
    expected_mmp_pairs_excluded_by_hydrogen: int = 0
    expected_input_exclusions: int | None = None
    expected_mmp_pairs_excluded_by_input: int | None = None
    expected_retained_mmp_pairs: int = 15186
    expected_mmp_source_pairs: int = 23029
    expected_mmp_symmetry_ambiguous_pairs: int = 4845
    expected_mmp_max_automorphism_candidates: int = 16
    expected_mmp_core_mapping_method: str = MMP_CORE_MAPPING_METHOD
    expected_train_rows: int | None = 1005
    expected_validation_explicit_hydrogen_exclusions: int | None = 0
    expected_validation_input_exclusions: int | None = None

    def __post_init__(self) -> None:
        """Reject only settings that would silently change or invalidate the objective."""
        # NOTE: The Notebook caller owns type, range, path, and schema contracts.
        finite_values = (
            self.learning_rate,
            self.weight_decay,
            self.dropout,
            self.prediction_huber_delta,
            self.mmp_weight,
            self.mmp_core_weight,
            self.mmp_difference_weight,
            self.mmp_huber_beta,
            self.fragment_frequency_weight,
            self.fragment_positive_weight,
            self.fragment_consistency_weight,
            self.fragment_min_frequency,
        )
        if not all(math.isfinite(float(value)) for value in finite_values):
            raise ValueError("Training hyperparameters must be finite")
        if self.output_head_init_std is not None and not math.isfinite(
            self.output_head_init_std
        ):
            raise ValueError("Output-head initialization must be finite")

        if (
            self.training_scope == "full_dataset"
            and self.checkpoint_selection != "fixed_final_epoch"
        ):
            raise ValueError(
                "full_dataset training requires fixed_final_epoch checkpoint selection"
            )
        if self.fragment_frequency_weight > 0.0 and self.scaling_mode != "standard":
            raise ValueError(
                "fragment-frequency loss requires standard target scaling"
            )
        if (
            self.mmp_weight > 0.0
            and self.mmp_core_weight + self.mmp_difference_weight <= 0.0
        ):
            raise ValueError(
                "active MMP loss requires a positive core or difference component"
            )
        if (
            self.fragment_frequency_weight > 0.0
            and self.fragment_positive_weight
            + self.fragment_consistency_weight
            <= 0.0
        ):
            raise ValueError(
                "active fragment loss requires a positive component weight"
            )

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible, complete experiment configuration."""

        payload = asdict(self)
        payload["loss_gradient_scaling"]["batch_seeds"] = list(
            self.loss_gradient_scaling.batch_seeds
        )
        data_payload = payload["data"]
        path_references: dict[str, str] = {}
        for key in (
            "dataset_csv",
            "graph_root",
            "fold_csv",
            "representation_cache",
        ):
            serialized, reference = serialize_output_path(
                data_payload[key]
            )
            data_payload[key] = serialized
            path_references[f"data.{key}"] = reference
        for key in ("mmp_csv", "output_dir"):
            serialized, reference = serialize_output_path(payload[key])
            payload[key] = serialized
            path_references[key] = reference
        if payload["fragment_cache"] is not None:
            serialized, reference = serialize_output_path(
                payload["fragment_cache"]
            )
            payload["fragment_cache"] = serialized
            path_references["fragment_cache"] = reference
        payload["path_base"] = "XAI_CHEM_ROOT"
        payload["path_references"] = path_references
        payload["bond_feature_dim"] = (
            BOND_FEATURE_DIM if self.use_edge_features else None
        )
        return payload

    @classmethod
    def checked_chembl3884_260729_full_objective(
        cls,
        *,
        data: DataConfig,
        fragment_cache: Path,
        mmp_csv: Path = DEFAULT_CHEMBL3884_260729_MMP_CSV,
        output_dir: Path,
        device: str,
        seed: int = 42,
        validation_fold: int = 0,
    ) -> "TrainingConfig":
        """Build the checked full objective for the 2026-07-29 dataset."""

        # NOTE: CONTRACT in=validation_fold: 0-4 only for checked profile
        fold_contract = CHECKED_CHEMBL3884_260729_FOLD_CONTRACTS[
            validation_fold
        ]
        return cls(
            data=data,
            mmp_csv=mmp_csv,
            fragment_cache=fragment_cache,
            output_dir=output_dir,
            device=device,
            run_profile="checked_chembl3884_260729_full_objective",
            validation_fold=validation_fold,
            seed=seed,
            expected_explicit_hydrogen_exclusions=0,
            expected_mmp_pairs_excluded_by_hydrogen=0,
            expected_input_exclusions=0,
            expected_mmp_pairs_excluded_by_input=0,
            expected_retained_mmp_pairs=fold_contract[
                "expected_retained_mmp_pairs"
            ],
            expected_mmp_source_pairs=23029,
            expected_mmp_symmetry_ambiguous_pairs=4845,
            expected_mmp_max_automorphism_candidates=16,
            expected_train_rows=fold_contract["expected_train_rows"],
            expected_validation_explicit_hydrogen_exclusions=fold_contract[
                "expected_validation_explicit_hydrogen_exclusions"
            ],
            expected_validation_input_exclusions=fold_contract[
                "expected_validation_input_exclusions"
            ],
            **CHECKED_FULL_OBJECTIVE_SETTINGS,
        )
