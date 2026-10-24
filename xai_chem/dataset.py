"""Dataset for notebook-based atom-contribution experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Any
import pandas as pd
from rdkit import Chem
from .io import exact_integer
import copy
import numpy as np
import torch
from torch_geometric.data import Data
from .io import load_pyg_data
from .representations import ExcludedMolecule
from .representations import RepresentationBundle
from .chemistry import mapped_graph_chemistry_from_smiles
from .representations import open_representation_cache
from .scaling import AffineLabelScaler


@dataclass(frozen=True)
class DataConfig:
    """All external inputs required to construct a modeling dataset."""

    dataset_csv: Path
    graph_root: Path
    fold_csv: Path
    representation_cache: Path
    molecule_id_column: str = "index"
    sample_id_column: str = "index"
    smiles_column: str = "Smiles"
    target_column: str = "pIC50"
    graph_file_template: str = "graph_{molecule_id:03d}.pt"
    expected_folds: int = 5

    def validate_paths(self) -> None:
        """Fail before training if any declared input is unavailable."""

        for name in ("dataset_csv", "fold_csv"):
            path = Path(getattr(self, name))
            if not path.is_file():
                raise FileNotFoundError(f"{name} does not exist: {path}")
        for name in ("graph_root", "representation_cache"):
            path = Path(getattr(self, name))
            if not path.is_dir():
                raise NotADirectoryError(f"{name} does not exist: {path}")

    def resolved_graph_file_template(self) -> str:
        """Return the explicitly declared graph filename template."""

        template = self.graph_file_template
        # NOTE: CONTRACT in=template:str with {molecule_id} placeholder
        template.format(molecule_id=1)
        return template


def explicit_hydrogen_row_indices(
    frame: pd.DataFrame,
    *,
    smiles_column: str = "Smiles",
) -> tuple[int, ...]:
    """Return rows containing explicit H/D/T atoms absent from model nodes."""

    if smiles_column not in frame.columns:
        raise KeyError(f"Dataset is missing SMILES column: {smiles_column}")
    parser = Chem.SmilesParserParams()
    parser.removeHs = False
    unsupported: list[int] = []
    for row_index, smiles_value in enumerate(frame[smiles_column].tolist()):
        smiles = str(smiles_value)
        molecule = Chem.MolFromSmiles(smiles, parser)
        if molecule is None:
            raise ValueError(
                f"RDKit could not parse SMILES at row {row_index}: {smiles!r}"
            )
        if any(atom.GetAtomicNum() == 1 for atom in molecule.GetAtoms()):
            unsupported.append(row_index)
    return tuple(unsupported)


def representation_cache_excluded_row_indices(
    frame: pd.DataFrame,
    molecule_rows: Sequence[Mapping[str, Any]],
    *,
    sample_id_column: str = "index",
) -> tuple[int, ...]:
    """Return source rows absent from an immutable representation cache."""

    if sample_id_column not in frame.columns:
        raise KeyError(
            f"Dataset is missing sample ID column: {sample_id_column}"
        )
    sample_ids = frame[sample_id_column]
    if (
        sample_ids.isna().any()
        or sample_ids.duplicated().any()
        or (sample_ids.astype(str).str.strip() == "").any()
    ):
        raise ValueError(
            "Dataset sample identifiers must be non-null and unique"
        )
    dataset_ids = tuple(str(value) for value in sample_ids.tolist())
    cached_ids = tuple(str(row["sample_id"]) for row in molecule_rows)
    if len(cached_ids) != len(set(cached_ids)):
        raise ValueError("Representation cache sample identifiers are not unique")
    unknown = sorted(set(cached_ids) - set(dataset_ids))
    if unknown:
        raise ValueError(
            "Representation cache contains sample IDs absent from the dataset: "
            f"{unknown[:10]}"
        )
    cached_set = set(cached_ids)
    return tuple(
        row_index
        for row_index, sample_id in enumerate(dataset_ids)
        if sample_id not in cached_set
    )


def exclude_rows(
    row_indices: Sequence[int],
    excluded: Sequence[int],
) -> list[int]:
    """Apply one recorded exclusion set without altering remaining order."""

    excluded_set = set(excluded)
    return [int(row) for row in row_indices if int(row) not in excluded_set]


def load_fold_indices(
    dataset: pd.DataFrame,
    fold_csv: Path | str,
    *,
    molecule_id_column: str = "index",
    expected_folds: int = 5,
) -> dict[int, list[int]]:
    """Map declared molecule identifiers to dataset row positions.

    No zero/one-based inference or row-order fallback is performed.
    """

    if molecule_id_column not in dataset.columns:
        raise KeyError(f"Dataset is missing id column: {molecule_id_column}")
    dataset_id_series = dataset[molecule_id_column]
    if dataset_id_series.isna().any() or dataset_id_series.duplicated().any():
        raise ValueError("Dataset molecule identifiers must be non-null and unique")
    dataset_ids = [
        exact_integer(value, name=f"dataset molecule id at row {row_index}")
        for row_index, value in enumerate(dataset_id_series.tolist())
    ]
    if len(dataset_ids) != len(set(dataset_ids)):
        raise ValueError(
            "Dataset molecule identifiers are not unique as integers"
        )

    folds = pd.read_csv(fold_csv)
    required = {molecule_id_column, "fold"}
    missing_columns = sorted(required - set(folds.columns))
    if missing_columns:
        raise KeyError(f"Fold CSV is missing columns: {missing_columns}")
    fold_id_series = folds[molecule_id_column]
    if fold_id_series.isna().any() or fold_id_series.duplicated().any():
        raise ValueError("Fold molecule identifiers must be non-null and unique")
    fold_ids = [
        exact_integer(value, name=f"fold molecule id at row {row_index}")
        for row_index, value in enumerate(fold_id_series.tolist())
    ]
    if len(fold_ids) != len(set(fold_ids)):
        raise ValueError("Fold molecule identifiers are not unique as integers")

    dataset_id_set = set(dataset_ids)
    fold_id_set = set(fold_ids)
    if dataset_id_set != fold_id_set:
        missing = sorted(dataset_id_set - fold_id_set)[:10]
        unexpected = sorted(fold_id_set - dataset_id_set)[:10]
        raise ValueError(
            "Dataset/fold identifiers do not match exactly: "
            f"missing={missing}, unexpected={unexpected}"
        )

    fold_values = [
        exact_integer(value, name=f"fold assignment at row {row_index}")
        for row_index, value in enumerate(folds["fold"].tolist())
    ]
    expected_fold_count = exact_integer(
        expected_folds,
        name="expected_folds",
    )
    if expected_fold_count < 1:
        raise ValueError("expected_folds must be positive")
    expected = set(range(expected_fold_count))
    observed = set(fold_values)
    if observed != expected:
        raise ValueError(
            f"Fold ids must be exactly {sorted(expected)}, got {sorted(observed)}"
        )

    fold_by_id = dict(zip(fold_ids, fold_values))
    out = {fold: [] for fold in sorted(expected)}
    for row_index, molecule_id in enumerate(dataset_ids):
        out[fold_by_id[molecule_id]].append(row_index)
    return out


def split_fold(
    fold_indices: dict[int, list[int]],
    validation_fold: int,
) -> tuple[list[int], list[int]]:
    """Return train and validation rows for a declared fold."""

    if validation_fold not in fold_indices:
        raise KeyError(
            f"validation_fold={validation_fold} is not in {sorted(fold_indices)}"
        )
    validation = list(fold_indices[validation_fold])
    train = [
        row
        for fold, rows in sorted(fold_indices.items())
        if fold != validation_fold
        for row in rows
    ]
    if not train or not validation:
        raise ValueError("Both train and validation splits must be non-empty")
    if set(train) & set(validation):
        raise ValueError("Train and validation splits overlap")
    return train, validation


BOND_FEATURE_DIM = 6


def _graph_order_cache_slice(
    bundle: RepresentationBundle,
    graph_idx: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Copy one mmap slice while checking its numerical and mapping contract."""

    start = int(bundle.offsets[graph_idx])
    stop = int(bundle.offsets[graph_idx + 1])
    encoder_values = np.array(
        bundle.values[start:stop],
        dtype=np.float32,
        copy=True,
    )
    if not np.isfinite(encoder_values).all():
        raise ValueError(
            f"Representation contains non-finite values at cache row {graph_idx}"
        )
    encoder_atomic_numbers = np.array(
        bundle.atomic_numbers[start:stop],
        dtype=np.int64,
        copy=True,
    )
    encoder_atom_index = np.array(
        bundle.encoder_atom_index[start:stop],
        dtype=np.int64,
        copy=True,
    )
    graph_atom_index = np.array(
        bundle.graph_atom_index[start:stop],
        dtype=np.int64,
        copy=True,
    )
    expected = np.arange(stop - start, dtype=np.int64)
    if not np.array_equal(np.sort(encoder_atom_index), expected):
        raise ValueError(
            f"Encoder atom index is not bijective at cache row {graph_idx}"
        )
    if not np.array_equal(np.sort(graph_atom_index), expected):
        raise ValueError(
            f"Graph atom index is not bijective at cache row {graph_idx}"
        )
    values = np.empty_like(encoder_values)
    atomic_numbers = np.empty_like(encoder_atomic_numbers)
    values[graph_atom_index] = encoder_values
    atomic_numbers[graph_atom_index] = encoder_atomic_numbers
    return values, atomic_numbers, encoder_atom_index, graph_atom_index


class CacheBackedGraphDataset(torch.utils.data.Dataset[Data]):
    """Load graphs only after dataset, cache, target, and atom identity agree."""

    def __init__(
        self,
        config: DataConfig,
        *,
        scaler: AffineLabelScaler | None = None,
        validate_all: bool = True,
    ) -> None:
        config.validate_paths()
        self.config = config
        self.scaler = scaler
        self.graph_file_template = config.resolved_graph_file_template()
        self.frame = pd.read_csv(config.dataset_csv)
        self._validate_frame()
        self.fold_indices = load_fold_indices(
            self.frame,
            config.fold_csv,
            molecule_id_column=config.molecule_id_column,
            expected_folds=config.expected_folds,
        )
        self.explicit_hydrogen_rows = explicit_hydrogen_row_indices(
            self.frame,
            smiles_column=config.smiles_column,
        )
        self.cache = open_representation_cache(config.representation_cache)
        excluded_rows = representation_cache_excluded_row_indices(
            self.frame,
            self.cache.molecule_rows,
            sample_id_column=config.sample_id_column,
        )
        self._index_cache_rows(excluded_rows)
        self._validated_graphs: tuple[Data, ...] | None = None
        if validate_all:
            self._validated_graphs = tuple(
                self._load_graph(dataset_position)
                for dataset_position in range(len(self))
            )

    def _validate_frame(self) -> None:
        required = {
            self.config.molecule_id_column,
            self.config.sample_id_column,
            self.config.smiles_column,
            self.config.target_column,
        }
        missing = sorted(required - set(self.frame.columns))
        if missing:
            raise KeyError(f"Dataset CSV is missing columns: {missing}")
        identifiers = self.frame[self.config.molecule_id_column]
        if identifiers.isna().any() or identifiers.duplicated().any():
            raise ValueError("Dataset molecule identifiers must be non-null and unique")
        numeric_identifiers = [
            exact_integer(value, name=f"dataset molecule id at row {row_index}")
            for row_index, value in enumerate(identifiers.tolist())
        ]
        if len(numeric_identifiers) != len(set(numeric_identifiers)):
            raise ValueError(
                "Dataset molecule identifiers are not unique as integers"
            )
        sample_ids = self.frame[self.config.sample_id_column]
        if (
            sample_ids.isna().any()
            or sample_ids.duplicated().any()
            or (sample_ids.astype(str).str.strip() == "").any()
        ):
            raise ValueError("Dataset sample identifiers must be non-null and unique")
        smiles = self.frame[self.config.smiles_column]
        if smiles.isna().any() or (smiles.astype(str).str.strip() == "").any():
            raise ValueError("Dataset SMILES values must be non-empty")
        targets = pd.to_numeric(
            self.frame[self.config.target_column],
            errors="raise",
        ).to_numpy(dtype=np.float64)
        if not np.isfinite(targets).all():
            raise ValueError("Dataset targets contain non-finite values")

    def _index_cache_rows(self, excluded_rows: tuple[int, ...]) -> None:
        """Bind cache order to source rows without deserializing graph files."""

        excluded_set = set(excluded_rows)
        source_rows = tuple(
            row_index
            for row_index in range(len(self.frame))
            if row_index not in excluded_set
        )
        if len(self.cache.molecule_rows) != len(source_rows):
            raise ValueError(
                "Eligible dataset and representation cache row counts differ"
            )
        for cache_idx, (source_row, cached) in enumerate(
            zip(source_rows, self.cache.molecule_rows)
        ):
            row = self.frame.iloc[source_row]
            if (
                cached["graph_idx"] != cache_idx
                or cached["sample_id"]
                != str(row[self.config.sample_id_column])
                or cached["smiles"] != str(row[self.config.smiles_column])
            ):
                raise ValueError(
                    f"Dataset/cache identity mismatch at cache row {cache_idx}"
                )
        self.source_row_indices = source_rows
        self.exclusions = tuple(
            ExcludedMolecule(
                source_row_index=source_row,
                molecule_id=exact_integer(
                    self.frame.iloc[source_row][self.config.molecule_id_column],
                    name=f"dataset molecule id at row {source_row}",
                ),
                sample_id=str(
                    self.frame.iloc[source_row][self.config.sample_id_column]
                ),
                smiles=str(self.frame.iloc[source_row][self.config.smiles_column]),
                reason="excluded_by_representation_cache_input_contract",
            )
            for source_row in excluded_rows
        )
        self._position_by_source_row = {
            source_row: position
            for position, source_row in enumerate(self.source_row_indices)
        }

    def __len__(self) -> int:
        return len(self.source_row_indices)

    def with_scaler(
        self,
        scaler: AffineLabelScaler,
    ) -> CacheBackedGraphDataset:
        """Share validated graph features while replacing fold-scaled labels."""

        # NOTE: caller supplies a validate_all=True dataset. Graph feature tensors
        # are immutable; only the shallow Data container and its y tensor differ.
        scaled_targets = scaler.transform(
            self.frame.iloc[list(self.source_row_indices)][
                self.config.target_column
            ].to_numpy(dtype=np.float64)
        )
        view = copy.copy(self)
        view.scaler = scaler
        scaled_graphs = []
        for graph, scaled_target in zip(
            self._validated_graphs,
            scaled_targets,
        ):
            scaled_graph = copy.copy(graph)
            scaled_graph.y = torch.tensor(
                [scaled_target],
                dtype=torch.float32,
            )
            scaled_graphs.append(scaled_graph)
        view._validated_graphs = tuple(scaled_graphs)
        return view

    def position_for_source_row(self, source_row_index: int) -> int:
        """Map an original CSV row to its eligible dataset position."""

        if isinstance(source_row_index, bool) or not isinstance(
            source_row_index, int
        ):
            raise TypeError("source_row_index must be an integer")
        # NOTE: CONTRACT in=source_row_index:int, out=position:int
        #       fail-fast: KeyError if source_row_index has no modeling position
        return self._position_by_source_row[source_row_index]

    def _graph_path(self, molecule_id: object) -> Path:
        numeric_id = exact_integer(molecule_id, name="molecule id")
        filename = self.graph_file_template.format(
            molecule_id=numeric_id
        )
        path = Path(self.config.graph_root) / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing graph for molecule {molecule_id}: {path}")
        return path

    def _load_graph(self, dataset_position: int) -> Data:
        """Attach cached raw atom states and the training-fitted target scaler to one graph."""

        source_row_index = self.source_row_indices[dataset_position]
        row = self.frame.iloc[source_row_index]
        molecule_id = row[self.config.molecule_id_column]
        sample_id = str(row[self.config.sample_id_column])
        smiles = str(row[self.config.smiles_column])
        path = self._graph_path(molecule_id)
        graph = load_pyg_data(path, map_location="cpu")
        numeric_id = exact_integer(molecule_id, name="molecule id")
        if exact_integer(
            getattr(graph, "idx", -1),
            name=f"{path} graph.idx",
        ) != numeric_id:
            raise ValueError(f"Graph molecule id mismatch: {path}")
        if exact_integer(
            getattr(graph, "graph_idx", -1),
            name=f"{path} graph.graph_idx",
        ) != source_row_index:
            raise ValueError(f"Graph source row mismatch: {path}")
        if str(getattr(graph, "sample_id", "")) != sample_id:
            raise ValueError(f"Graph sample id mismatch: {path}")
        if str(getattr(graph, "smiles", "")) != smiles:
            raise ValueError(f"Graph SMILES mismatch: {path}")

        (
            values,
            cached_atomic_numbers,
            encoder_atom_index,
            graph_atom_index,
        ) = _graph_order_cache_slice(
            self.cache,
            dataset_position,
        )
        atom_mapping = getattr(graph, "smiles_to_graph_atom_index", None)
        if atom_mapping is None:
            raise ValueError(
                f"Graph has no explicit smiles_to_graph_atom_index: {path}"
            )
        atom_mapping = atom_mapping.to(dtype=torch.long)
        expected_mapping = torch.arange(graph.num_nodes, dtype=torch.long)
        if not torch.equal(torch.sort(atom_mapping).values, expected_mapping):
            raise ValueError(f"Graph atom mapping is not bijective: {path}")
        expected_representation_mapping = atom_mapping.cpu().numpy()[
            encoder_atom_index
        ]
        if not np.array_equal(
            graph_atom_index,
            expected_representation_mapping,
        ):
            raise ValueError(f"Cache/graph atom mapping mismatch: {path}")

        edge_index = graph.edge_index.to(dtype=torch.long)
        expected_atomic_numbers, expected_edge_attr_array = (
            mapped_graph_chemistry_from_smiles(
                smiles,
                edge_index,
                atom_mapping,
            )
        )
        if not np.array_equal(cached_atomic_numbers, expected_atomic_numbers):
            raise ValueError(f"Cache/SMILES atom-order mismatch: {path}")
        if graph.num_nodes != len(cached_atomic_numbers):
            raise ValueError(f"Graph/cache atom-count mismatch: {path}")
        graph_atomic_numbers = getattr(graph, "atomic_numbers", None)
        if graph_atomic_numbers is None:
            raise ValueError(f"Graph has no explicit atomic_numbers: {path}")
        graph_atomic_numbers = graph_atomic_numbers.to(dtype=torch.long)
        if not np.array_equal(
            graph_atomic_numbers.cpu().numpy(),
            cached_atomic_numbers,
        ):
            raise ValueError(f"Graph/cache atomic-number mismatch: {path}")

        expected_edge_attr = torch.from_numpy(expected_edge_attr_array)
        stored_edge_attr = getattr(graph, "edge_attr", None)
        if stored_edge_attr is None:
            raise ValueError(f"Graph has no explicit edge_attr: {path}")
        stored_edge_attr = stored_edge_attr.to(dtype=torch.float32)
        if not torch.equal(stored_edge_attr, expected_edge_attr):
            raise ValueError(f"Graph bond features do not match SMILES: {path}")
        target = float(row[self.config.target_column])
        scaled_target = (
            target
            if self.scaler is None
            else float(self.scaler.transform([target])[0])
        )
        graph.x = torch.from_numpy(np.array(values, dtype=np.float32, copy=True))
        graph.edge_index = edge_index
        graph.edge_attr = stored_edge_attr
        graph.atomic_numbers = graph_atomic_numbers
        graph.smiles_to_graph_atom_index = atom_mapping
        graph.y = torch.tensor([scaled_target], dtype=torch.float32)
        graph.y_raw = torch.tensor([target], dtype=torch.float32)
        graph.graph_idx = int(source_row_index)
        graph.cache_idx = int(dataset_position)
        graph.sample_id = sample_id
        return graph

    def __getitem__(self, dataset_position: int) -> Data:
        if isinstance(dataset_position, bool) or not isinstance(
            dataset_position, int
        ):
            raise TypeError("dataset_position must be an integer")
        if dataset_position < 0 or dataset_position >= len(self):
            raise IndexError(
                f"dataset_position out of range: {dataset_position}"
            )
        if self._validated_graphs is not None:
            # Validated graph objects are immutable inputs. PyG batching creates
            # a separate Batch, so cloning every sample only duplicates tensors.
            return self._validated_graphs[dataset_position]
        return self._load_graph(dataset_position)
