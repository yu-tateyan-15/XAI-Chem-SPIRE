"""Graphs for notebook-based atom-contribution experiments."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from .chemistry import ATOM_MAPPING_VERSION
from .chemistry import heavy_atom_edges_from_smiles
from .chemistry import heavy_atom_numbers_from_smiles
from .chemistry import mapped_bond_features_from_smiles
from .representations import sha256_file
from .io import write_json
from .io import exact_integer


def _source_id_column(frame: pd.DataFrame) -> str:
    """Resolve the human-readable molecule identifier without changing legacy data."""

    if "number" in frame.columns:
        return "number"
    if "cid" in frame.columns:
        return "cid"
    raise KeyError("Dataset CSV requires either a number or cid column")


def _filename_width(molecule_ids: list[int]) -> int:
    """Use at least three digits and expand to the largest molecule id."""

    return max(3, len(str(max(molecule_ids))))


def _graph_filename(molecule_id: int, filename_width: int) -> str:
    return f"graph_{molecule_id:0{filename_width}d}.pt"


def _validate_frame(frame: pd.DataFrame, target_column: str = "pIC50") -> tuple[list[int], str, int]:
    required = {"index", "Smiles", target_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"Dataset CSV is missing columns: {missing}")
    source_id_column = _source_id_column(frame)
    if frame["index"].isna().any() or frame["index"].duplicated().any():
        raise ValueError("Dataset index values must be non-null and unique")
    if (
        frame[source_id_column].isna().any()
        or (frame[source_id_column].astype(str).str.strip() == "").any()
    ):
        raise ValueError(
            f"Dataset {source_id_column} values must be non-null and non-empty"
        )
    molecule_ids = [
        exact_integer(
            value,
            name=f"dataset row {row_index} molecule id",
        )
        for row_index, value in enumerate(frame["index"].tolist())
    ]
    if len(molecule_ids) != len(set(molecule_ids)):
        raise ValueError("Dataset molecule identifiers are not unique as integers")
    filename_width = _filename_width(molecule_ids)
    filenames = [
        _graph_filename(molecule_id, filename_width)
        for molecule_id in molecule_ids
    ]
    if len(filenames) != len(set(filenames)):
        raise ValueError("Dataset graph filenames are not unique")
    targets = pd.to_numeric(frame[target_column], errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(targets).all():
        raise ValueError(f"Dataset {target_column} targets contain non-finite values")
    return molecule_ids, source_id_column, filename_width


def build_graphs(dataset_csv: Path, output_root: Path, *, target_column: str = "pIC50") -> Path:
    """Build topology and raw labels using one explicitly named regression target."""
    if not dataset_csv.is_file():
        raise FileNotFoundError(f"Dataset CSV does not exist: {dataset_csv}")
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite graph output: {output_root}")
    frame = pd.read_csv(dataset_csv)
    molecule_ids, source_id_column, filename_width = _validate_frame(frame, target_column)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.tmp-",
            dir=output_root.parent,
        )
    )
    try:
        graph_records: list[dict[str, object]] = []
        for (graph_idx, row), molecule_id in zip(
            frame.iterrows(),
            molecule_ids,
            strict=True,
        ):
            sample_id = str(molecule_id)
            source_number = str(row[source_id_column])
            smiles = str(row["Smiles"])
            atomic_numbers = torch.from_numpy(
                heavy_atom_numbers_from_smiles(smiles).copy()
            )
            edge_index = torch.from_numpy(
                heavy_atom_edges_from_smiles(smiles).copy()
            )
            atom_mapping = torch.arange(len(atomic_numbers), dtype=torch.long)
            edge_attr = torch.from_numpy(
                mapped_bond_features_from_smiles(
                    smiles,
                    edge_index,
                    atom_mapping,
                )
            )
            graph = Data(
                x=torch.empty((len(atomic_numbers), 0), dtype=torch.float32),
                edge_index=edge_index,
                edge_attr=edge_attr,
                atomic_numbers=atomic_numbers,
                smiles_to_graph_atom_index=atom_mapping,
                y_raw=torch.tensor([float(row[target_column])], dtype=torch.float32),
                smiles=smiles,
                idx=molecule_id,
                index=source_number,
                sample_id=sample_id,
                graph_idx=int(graph_idx),
            )
            filename = _graph_filename(molecule_id, filename_width)
            destination = staging_root / filename
            temporary = destination.with_suffix(".pt.tmp")
            torch.save(graph, temporary)
            temporary.replace(destination)
            graph_records.append(
                {
                    "graph_idx": int(graph_idx),
                    "molecule_id": molecule_id,
                    "sample_id": sample_id,
                    "source_number": source_number,
                    "filename": filename,
                    "node_count": int(graph.num_nodes),
                    "edge_count": int(graph.num_edges),
                    "sha256": sha256_file(destination),
                }
            )

        manifest = {
            "schema_version": 3,
            "builder": "xai_chem/graphs.py",
            "dataset_filename": dataset_csv.name,
            "dataset_sha256": sha256_file(dataset_csv),
            "source_id_column": source_id_column,
            "graph_filename_width": filename_width,
            "graph_file_template": (
                f"graph_{{molecule_id:0{filename_width}d}}.pt"
            ),
            "atom_mapping_version": ATOM_MAPPING_VERSION,
            "hydrogen_policy": "remove_all_atomic_number_1_isotopes",
            "formal_charge_source": "SMILES",
            "bond_features": [
                "single",
                "double",
                "triple",
                "aromatic",
                "conjugated",
                "in_ring",
            ],
            "graph_count": len(graph_records),
            "graphs": graph_records,
        }
        write_json(staging_root / "manifest.json", manifest)
        if output_root.exists():
            raise FileExistsError(
                f"Graph output appeared during build: {output_root}"
            )
        staging_root.rename(output_root)
        return output_root / "manifest.json"
    finally:
        if staging_root.exists():
            shutil.rmtree(staging_root)


def assign_activity_balanced_folds(
    frame: pd.DataFrame,
    *,
    n_folds: int = 5,
    n_bins: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Assign folds by shuffled round-robin allocation within pIC50 bins.

    This preserves the historical SGLT2 protocol: pIC50 is divided into
    quantile bins, rows are shuffled deterministically within each bin, and
    consecutive rows are allocated to folds 0..n_folds-1.
    """

    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")
    if len(frame) < n_folds:
        raise ValueError("Dataset has fewer rows than requested folds")
    missing = sorted({"index", "pIC50"} - set(frame.columns))
    if missing:
        raise KeyError(f"Dataset frame is missing columns: {missing}")

    rng = np.random.default_rng(int(seed))
    folds = frame[["index", "pIC50"]].copy()
    folds["pIC50"] = pd.to_numeric(folds["pIC50"], errors="raise")
    effective_bins = min(
        int(n_bins),
        max(2, int(len(folds) // max(1, int(n_folds)))),
    )
    folds["bin"] = pd.qcut(
        folds["pIC50"],
        q=effective_bins,
        duplicates="drop",
    ).astype(str)
    folds["fold"] = -1
    for _, group in folds.groupby("bin", sort=True):
        shuffled_rows = group.index.to_numpy()
        rng.shuffle(shuffled_rows)
        for offset, row_index in enumerate(shuffled_rows):
            folds.loc[row_index, "fold"] = int(offset % n_folds)
    if (folds["fold"] < 0).any():
        raise RuntimeError("Fold assignment failed for at least one row")
    return folds[["index", "pIC50", "fold", "bin"]].sort_values(
        "index"
    ).reset_index(drop=True)
