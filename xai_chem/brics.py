"""Brics for notebook-based atom-contribution experiments."""

from __future__ import annotations

import json
import pickle
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import numpy as np
import pandas as pd
import torch
from .fragment_loss import BRICS_CACHE_SCHEMA_VERSION
from .fragment_loss import brics_fragments_for_smiles


@dataclass(frozen=True)
class GraphFragmentInput:
    """One molecule with the atom-order contract needed by BRICS."""

    source_row: int
    molecule_id: int
    cid: str
    smiles: str
    pIC50: float
    num_nodes: int
    smiles_to_graph_atom_index: tuple[int, ...]


def load_graph_fragment_inputs(
    dataset_csv: Path,
    graph_root: Path,
) -> tuple[GraphFragmentInput, ...]:
    """Load dataset rows and fail if graph identity or atom mapping differs."""

    dataset_path = Path(dataset_csv)
    root = Path(graph_root)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset CSV does not exist: {dataset_path}")
    if not root.is_dir():
        raise NotADirectoryError(f"Graph directory does not exist: {root}")
    frame = pd.read_csv(dataset_path)
    required = {"index", "cid", "Smiles", "pIC50"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"Dataset CSV is missing columns: {missing}")
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Graph manifest does not exist: {manifest_path}")
    graph_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    filename_width = int(
        graph_manifest.get(
            "graph_filename_width",
            max(3, len(str(int(frame["index"].max())))),
        )
    )

    items = []
    for source_row, row in frame.iterrows():
        molecule_id = int(row["index"])
        graph_path = root / f"graph_{molecule_id:0{filename_width}d}.pt"
        if not graph_path.is_file():
            raise FileNotFoundError(f"Missing graph file: {graph_path}")
        graph = torch.load(
            graph_path,
            map_location="cpu",
            weights_only=False,
        )
        mapping = tuple(
            int(value)
            for value in graph.smiles_to_graph_atom_index.detach()
            .cpu()
            .tolist()
        )
        if tuple(sorted(mapping)) != tuple(range(int(graph.num_nodes))):
            raise ValueError(
                f"Graph atom mapping is not bijective: {graph_path}"
            )
        if int(graph.idx) != molecule_id:
            raise ValueError(f"Graph molecule id mismatch: {graph_path}")
        if int(graph.graph_idx) != int(source_row):
            raise ValueError(f"Graph source-row mismatch: {graph_path}")
        if str(graph.sample_id) != str(molecule_id):
            raise ValueError(f"Graph sample id mismatch: {graph_path}")
        if str(graph.smiles) != str(row["Smiles"]):
            raise ValueError(f"Graph SMILES mismatch: {graph_path}")
        items.append(
            GraphFragmentInput(
                source_row=int(source_row),
                molecule_id=molecule_id,
                cid=str(row["cid"]),
                smiles=str(row["Smiles"]),
                pIC50=float(row["pIC50"]),
                num_nodes=int(graph.num_nodes),
                smiles_to_graph_atom_index=mapping,
            )
        )
    return tuple(items)


def compute_positive_margin(
    items: Sequence[GraphFragmentInput],
    activity_values: Sequence[float],
) -> float:
    """Fit m = median(abs(z_u) / n_u) in the supplied training subset."""

    values = np.asarray(activity_values, dtype=float).reshape(-1)
    if values.size != len(items):
        raise ValueError("Activity length does not match cache items")
    if not np.isfinite(values).all():
        raise ValueError("Activity values must be finite")
    standard_deviation = float(np.std(values, ddof=0))
    if standard_deviation <= 0.0:
        raise ValueError("Cannot fit positive margin from constant activity")
    activity_z = (values - float(np.mean(values))) / standard_deviation
    node_counts = np.asarray(
        [item.num_nodes for item in items],
        dtype=float,
    )
    if (node_counts <= 0).any():
        raise ValueError("All cache molecules require at least one graph atom")
    margin = float(np.median(np.abs(activity_z) / node_counts))
    if not np.isfinite(margin) or margin <= 0.0:
        raise ValueError("Positive margin must be finite and positive")
    return margin


def build_brics_cache_payload(
    items: Sequence[GraphFragmentInput],
    *,
    min_heavy_atoms: int = 1,
    min_count: int = 3,
    activity_values: Sequence[float] | None = None,
) -> dict:
    """Build BRICS occurrences in graph atom order and document frequencies."""

    if min_heavy_atoms < 1:
        raise ValueError("min_heavy_atoms must be positive")
    if min_count < 1:
        raise ValueError("min_count must be positive")
    if not items:
        raise ValueError("Cannot build a BRICS cache from zero molecules")
    source_rows = [item.source_row for item in items]
    if len(source_rows) != len(set(source_rows)):
        raise ValueError("BRICS cache source rows must be unique")

    raw_cache: dict[int, tuple[list[str], list[set[int]]]] = {}
    document_counts: Counter[str] = Counter()
    for item in items:
        occurrences = brics_fragments_for_smiles(
            item.smiles,
            min_heavy_atoms=int(min_heavy_atoms),
        )
        fragment_smiles: list[str] = []
        graph_atom_sets: list[set[int]] = []
        for fragment, smiles_atom_indices in occurrences:
            graph_indices = {
                item.smiles_to_graph_atom_index[int(atom_index)]
                for atom_index in smiles_atom_indices
            }
            if not graph_indices:
                raise ValueError(
                    f"BRICS fragment has no graph atoms: {item.cid}"
                )
            if min(graph_indices) < 0 or max(graph_indices) >= item.num_nodes:
                raise ValueError(
                    f"BRICS atom index is outside graph range: {item.cid}"
                )
            fragment_smiles.append(fragment)
            graph_atom_sets.append(graph_indices)
        raw_cache[item.source_row] = (fragment_smiles, graph_atom_sets)
        document_counts.update(set(fragment_smiles))

    retained = {
        fragment
        for fragment, count in document_counts.items()
        if count >= int(min_count)
    }
    global_frag_freqs = {
        fragment: document_counts[fragment] / float(len(items))
        for fragment in sorted(retained)
    }
    frag_cache: dict[int, tuple[list[str], list[set[int]]]] = {}
    for source_row, (fragments, atom_sets) in raw_cache.items():
        kept = [
            (fragment, atoms)
            for fragment, atoms in zip(fragments, atom_sets, strict=True)
            if fragment in retained
        ]
        frag_cache[source_row] = (
            [fragment for fragment, _ in kept],
            [atoms for _, atoms in kept],
        )

    positive_margin = (
        None
        if activity_values is None
        else compute_positive_margin(items, activity_values)
    )
    payload = {
        "schema_version": BRICS_CACHE_SCHEMA_VERSION,
        "global_frag_freqs": global_frag_freqs,
        "positive_margin": positive_margin,
        "frag_cache": frag_cache,
    }
    validate_brics_cache_payload(payload, items, min_count=min_count)
    return payload


def validate_brics_cache_payload(
    payload: dict,
    items: Sequence[GraphFragmentInput],
    *,
    min_count: int,
) -> None:
    """Fail when cache keys, frequencies, or atom indices break the contract."""

    required = {
        "schema_version",
        "global_frag_freqs",
        "positive_margin",
        "frag_cache",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise KeyError(f"BRICS cache is missing fields: {missing}")
    if payload["schema_version"] != BRICS_CACHE_SCHEMA_VERSION:
        raise ValueError("BRICS cache schema version differs")
    frag_cache = payload["frag_cache"]
    expected_rows = {item.source_row for item in items}
    if set(frag_cache) != expected_rows:
        raise ValueError("BRICS cache source-row coverage differs")
    item_by_row = {item.source_row: item for item in items}
    document_counts: Counter[str] = Counter()
    for source_row, (fragments, atom_sets) in frag_cache.items():
        if len(fragments) != len(atom_sets):
            raise ValueError("BRICS fragment and atom-set lengths differ")
        item = item_by_row[source_row]
        document_counts.update(set(fragments))
        for atoms in atom_sets:
            if not atoms:
                raise ValueError("BRICS cache contains an empty atom set")
            if min(atoms) < 0 or max(atoms) >= item.num_nodes:
                raise ValueError("BRICS cache atom index is outside graph range")
    expected_frequencies = {
        fragment: count / float(len(items))
        for fragment, count in sorted(document_counts.items())
    }
    if payload["global_frag_freqs"] != expected_frequencies:
        raise ValueError("BRICS frequencies disagree with cached occurrences")
    if any(count < int(min_count) for count in document_counts.values()):
        raise ValueError("BRICS cache retained a fragment below min_count")
    margin = payload["positive_margin"]
    if margin is not None and (
        not np.isfinite(float(margin)) or float(margin) <= 0.0
    ):
        raise ValueError("BRICS positive margin must be finite and positive")


def write_or_reuse_brics_cache(payload: dict, cache_path: Path) -> str:
    """Write a cache once, or reuse it only when the payload is identical."""

    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with path.open("rb") as handle:
            existing = pickle.load(handle)
        if existing != payload:
            raise FileExistsError(
                f"Refusing to replace non-identical BRICS cache: {path}"
            )
        return "reused"
    with path.open("wb") as handle:
        pickle.dump(payload, handle)
    return "written"


def fragment_summary_frame(payload: dict, n_molecules: int) -> pd.DataFrame:
    """Return retained fragment counts and frequencies for inspection."""

    occurrence_counts: Counter[str] = Counter()
    molecule_counts: Counter[str] = Counter()
    for fragments, _ in payload["frag_cache"].values():
        occurrence_counts.update(fragments)
        molecule_counts.update(set(fragments))
    rows = [
        {
            "fragment": fragment,
            "molecule_count": int(molecule_counts[fragment]),
            "occurrence_count": int(occurrence_counts[fragment]),
            "frequency": float(frequency),
            "n_molecules": int(n_molecules),
        }
        for fragment, frequency in payload["global_frag_freqs"].items()
    ]
    if not rows:
        return pd.DataFrame(
            columns=[
                "fragment",
                "molecule_count",
                "occurrence_count",
                "frequency",
                "n_molecules",
            ]
        )
    return pd.DataFrame(rows).sort_values(
        ["molecule_count", "fragment"],
        ascending=[False, True],
    ).reset_index(drop=True)
