"""Fragment loss for notebook-based atom-contribution experiments."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Mapping, Sequence
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import BRICS
from torch import Tensor


BRICS_CACHE_SCHEMA_VERSION = "brics_positive_consistency_v5"


@dataclass(frozen=True)
class FragmentOccurrence:
    """One BRICS fragment occurrence in graph atom order."""

    source_row_index: int
    fragment_smiles: str
    graph_atom_indices: tuple[int, ...]


@dataclass(frozen=True)
class FragmentFrequencySpec:
    """Train-fold fragment population and positive-reference scale."""

    occurrences: tuple[FragmentOccurrence, ...]
    frequencies: Mapping[str, float]
    positive_margin: float
    train_graph_count: int
    min_heavy_atoms: int
    min_count: int
    min_frequency: float


@dataclass(frozen=True)
class FragmentFrequencyLoss:
    """Differentiable fragment objective and its named components."""

    total: Tensor
    positive: Tensor
    consistency: Tensor
    inconsistency: Tensor
    active_fragment_count: int
    active_occurrence_count: int


def brics_fragments_for_smiles(
    smiles: str,
    *,
    min_heavy_atoms: int,
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    """Return frozen-definition BRICS fragments in heavy-atom SMILES order."""

    if isinstance(min_heavy_atoms, bool) or not isinstance(
        min_heavy_atoms, int
    ):
        raise TypeError("min_heavy_atoms must be an integer")
    if min_heavy_atoms < 1:
        raise ValueError("min_heavy_atoms must be positive")
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")
    heavy_atom_indices = {
        int(atom.GetIdx()): heavy_index
        for heavy_index, atom in enumerate(
            atom
            for atom in molecule.GetAtoms()
            if int(atom.GetAtomicNum()) > 1
        )
    }
    molecule_with_hydrogen = Chem.AddHs(molecule)
    broken = BRICS.BreakBRICSBonds(molecule_with_hydrogen)
    atom_mappings: list[tuple[int, ...]] = []
    fragments = Chem.GetMolFrags(
        broken,
        asMols=True,
        fragsMolAtomMapping=atom_mappings,
    )
    results: list[tuple[str, tuple[int, ...]]] = []
    for fragment, raw_atom_indices in zip(
        fragments,
        atom_mappings,
        strict=True,
    ):
        heavy_indices = tuple(
            sorted(
                {
                    heavy_atom_indices[int(atom_index)]
                    for atom_index in raw_atom_indices
                    if int(atom_index) in heavy_atom_indices
                }
            )
        )
        if len(heavy_indices) < min_heavy_atoms:
            continue
        results.append((Chem.MolToSmiles(fragment), heavy_indices))
    return tuple(results)


def build_fragment_frequency_spec(
    dataset,
    train_source_rows: Sequence[int],
    *,
    min_heavy_atoms: int = 1,
    min_count: int = 3,
    min_frequency: float = 0.1,
) -> FragmentFrequencySpec:
    """Build the frozen train-fold fragment population without disk fallback."""

    if isinstance(min_count, bool) or not isinstance(min_count, int):
        raise TypeError("min_count must be an integer")
    if min_count < 1:
        raise ValueError("min_count must be positive")
    frequency_threshold = float(min_frequency)
    if not np.isfinite(frequency_threshold) or not (
        0.0 <= frequency_threshold < 1.0
    ):
        raise ValueError("min_frequency must satisfy 0 <= value < 1")
    rows = tuple(int(row) for row in train_source_rows)
    if not rows or len(set(rows)) != len(rows):
        raise ValueError("train_source_rows must contain unique rows")

    raw_occurrences: list[FragmentOccurrence] = []
    molecule_fragments: list[set[str]] = []
    scaled_activity_per_atom: list[float] = []
    for source_row in rows:
        graph = dataset[dataset.position_for_source_row(source_row)]
        if int(graph.graph_idx) != source_row:
            raise ValueError("Dataset/source-row identity mismatch")
        mapping = graph.smiles_to_graph_atom_index.detach().cpu().numpy()
        if not np.array_equal(
            np.sort(mapping),
            np.arange(int(graph.num_nodes)),
        ):
            raise ValueError("SMILES-to-graph atom mapping is not bijective")
        fragments = brics_fragments_for_smiles(
            str(graph.smiles),
            min_heavy_atoms=min_heavy_atoms,
        )
        molecule_fragments.append(
            {fragment_smiles for fragment_smiles, _ in fragments}
        )
        for fragment_smiles, smiles_atom_indices in fragments:
            graph_atom_indices = tuple(
                sorted(int(mapping[index]) for index in smiles_atom_indices)
            )
            if not graph_atom_indices:
                raise ValueError("BRICS occurrence has no graph atoms")
            raw_occurrences.append(
                FragmentOccurrence(
                    source_row_index=source_row,
                    fragment_smiles=fragment_smiles,
                    graph_atom_indices=graph_atom_indices,
                )
            )
        scaled_target = float(graph.y.detach().cpu().reshape(-1)[0])
        if not np.isfinite(scaled_target):
            raise ValueError("Scaled training target is non-finite")
        scaled_activity_per_atom.append(
            abs(scaled_target) / float(graph.num_nodes)
        )

    document_counts = Counter(
        fragment
        for fragments in molecule_fragments
        for fragment in fragments
    )
    graph_count = len(rows)
    retained = {
        fragment
        for fragment, count in document_counts.items()
        if count >= min_count
    }
    frequencies = {
        fragment: document_counts[fragment] / float(graph_count)
        for fragment in sorted(retained)
    }
    occurrences = tuple(
        occurrence
        for occurrence in raw_occurrences
        if occurrence.fragment_smiles in retained
    )
    positive_margin = float(np.median(scaled_activity_per_atom))
    if not np.isfinite(positive_margin) or positive_margin <= 0.0:
        raise ValueError(
            "Train-fold fragment positive margin must be finite and positive"
        )
    if not any(
        frequency > frequency_threshold
        for frequency in frequencies.values()
    ):
        raise ValueError(
            "No retained train-fold fragment exceeds min_frequency"
        )
    return FragmentFrequencySpec(
        occurrences=occurrences,
        frequencies=frequencies,
        positive_margin=positive_margin,
        train_graph_count=graph_count,
        min_heavy_atoms=min_heavy_atoms,
        min_count=min_count,
        min_frequency=frequency_threshold,
    )


def load_fragment_frequency_spec(
    cache_path: Path | str,
    dataset,
    train_source_rows: Sequence[int],
    *,
    min_heavy_atoms: int = 1,
    min_count: int = 3,
    min_frequency: float = 0.1,
) -> FragmentFrequencySpec:
    """Load cached BRICS occurrences and refit fold-local statistics.

    The Input_dataset cache may include molecules later excluded by the
    atom-representation eligibility contract. Fragment occurrence generation
    is reused, while document frequencies and the standardized positive
    margin are recomputed on the exact active training rows.
    """

    path = Path(cache_path)
    if not path.is_file():
        raise FileNotFoundError(f"BRICS cache does not exist: {path}")
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise TypeError("BRICS cache payload must be a dictionary")
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
        raise ValueError(
            "BRICS cache schema differs from the active fragment contract"
        )
    cached_occurrences = payload["frag_cache"]
    if not isinstance(cached_occurrences, dict):
        raise TypeError("BRICS frag_cache must be a dictionary")

    if isinstance(min_count, bool) or not isinstance(min_count, int):
        raise TypeError("min_count must be an integer")
    if min_count < 1:
        raise ValueError("min_count must be positive")
    frequency_threshold = float(min_frequency)
    if not np.isfinite(frequency_threshold) or not (
        0.0 <= frequency_threshold < 1.0
    ):
        raise ValueError("min_frequency must satisfy 0 <= value < 1")
    rows = tuple(int(row) for row in train_source_rows)
    if not rows or len(set(rows)) != len(rows):
        raise ValueError("train_source_rows must contain unique rows")
    missing_rows = sorted(set(rows) - set(cached_occurrences))
    if missing_rows:
        raise ValueError(
            "BRICS cache does not cover active training rows: "
            f"{missing_rows[:10]}"
        )

    raw_occurrences: list[FragmentOccurrence] = []
    molecule_fragments: list[set[str]] = []
    scaled_activity_per_atom: list[float] = []
    for source_row in rows:
        graph = dataset[dataset.position_for_source_row(source_row)]
        if int(graph.graph_idx) != source_row:
            raise ValueError("Dataset/source-row identity mismatch")
        fragments, atom_sets = cached_occurrences[source_row]
        if len(fragments) != len(atom_sets):
            raise ValueError(
                "BRICS cache fragment and atom-set lengths differ"
            )
        molecule_fragments.append(set(str(item) for item in fragments))
        for fragment_smiles, atom_indices in zip(
            fragments,
            atom_sets,
            strict=True,
        ):
            graph_atom_indices = tuple(
                sorted(int(index) for index in atom_indices)
            )
            if not graph_atom_indices:
                raise ValueError("BRICS occurrence has no graph atoms")
            if (
                graph_atom_indices[0] < 0
                or graph_atom_indices[-1] >= int(graph.num_nodes)
            ):
                raise ValueError(
                    "BRICS cache atom index is outside the graph"
                )
            raw_occurrences.append(
                FragmentOccurrence(
                    source_row_index=source_row,
                    fragment_smiles=str(fragment_smiles),
                    graph_atom_indices=graph_atom_indices,
                )
            )
        scaled_target = float(graph.y.detach().cpu().reshape(-1)[0])
        if not np.isfinite(scaled_target):
            raise ValueError("Scaled training target is non-finite")
        scaled_activity_per_atom.append(
            abs(scaled_target) / float(graph.num_nodes)
        )

    document_counts = Counter(
        fragment
        for fragments in molecule_fragments
        for fragment in fragments
    )
    graph_count = len(rows)
    retained = {
        fragment
        for fragment, count in document_counts.items()
        if count >= min_count
    }
    frequencies = {
        fragment: document_counts[fragment] / float(graph_count)
        for fragment in sorted(retained)
    }
    occurrences = tuple(
        occurrence
        for occurrence in raw_occurrences
        if occurrence.fragment_smiles in retained
    )
    positive_margin = float(np.median(scaled_activity_per_atom))
    if not np.isfinite(positive_margin) or positive_margin <= 0.0:
        raise ValueError(
            "Train-fold fragment positive margin must be finite and positive"
        )
    if not any(
        frequency > frequency_threshold
        for frequency in frequencies.values()
    ):
        raise ValueError(
            "No retained train-fold fragment exceeds min_frequency"
        )
    return FragmentFrequencySpec(
        occurrences=occurrences,
        frequencies=frequencies,
        positive_margin=positive_margin,
        train_graph_count=graph_count,
        min_heavy_atoms=min_heavy_atoms,
        min_count=min_count,
        min_frequency=frequency_threshold,
    )


def compute_fragment_frequency_loss(
    contributions: Mapping[int, Tensor],
    spec: FragmentFrequencySpec,
    *,
    positive_weight: float = 1.0,
    consistency_weight: float = 1.0,
    eps: float = 1.0e-12,
) -> FragmentFrequencyLoss:
    """Compute the frozen positive/consistency fragment objective exactly."""

    positive_coefficient = float(positive_weight)
    consistency_coefficient = float(consistency_weight)
    epsilon = float(eps)
    for name, value in {
        "positive_weight": positive_coefficient,
        "consistency_weight": consistency_coefficient,
        "eps": epsilon,
    }.items():
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if positive_coefficient < 0.0 or consistency_coefficient < 0.0:
        raise ValueError("Fragment component weights cannot be negative")
    if positive_coefficient + consistency_coefficient <= 0.0:
        raise ValueError(
            "At least one fragment component weight must be positive"
        )
    if epsilon <= 0.0:
        raise ValueError("eps must be positive")

    scores: dict[str, list[Tensor]] = defaultdict(list)
    reference: Tensor | None = None
    for occurrence in spec.occurrences:
        frequency = spec.frequencies[occurrence.fragment_smiles]
        if frequency <= spec.min_frequency:
            continue
        # NOTE: CONTRACT in=contributions Mapping[int, Tensor], keyed by source row
        #       fail-fast: KeyError if source row key is absent
        values = contributions[occurrence.source_row_index]
        if values.ndim != 1 or values.numel() == 0:
            raise ValueError(
                "Fragment atom contributions must be non-empty vectors"
            )
        if not torch.is_floating_point(values):
            raise TypeError("Fragment atom contributions must be floating point")
        if not torch.isfinite(values).all():
            raise ValueError("Fragment atom contributions are non-finite")
        index = torch.tensor(
            occurrence.graph_atom_indices,
            dtype=torch.long,
            device=values.device,
        )
        if int(index.min()) < 0 or int(index.max()) >= values.numel():
            raise IndexError("Fragment occurrence has an out-of-range atom")
        if reference is None:
            reference = values
        elif values.device != reference.device or values.dtype != reference.dtype:
            raise ValueError(
                "All fragment contributions must share device and dtype"
            )
        scores[occurrence.fragment_smiles].append(values[index].mean())
    if reference is None or not scores:
        raise ValueError("No active fragment occurrences are available")

    margin = reference.new_tensor(spec.positive_margin)
    positive_sum = reference.new_zeros(())
    positive_frequency_sum = 0.0
    consistency_sum = reference.new_zeros(())
    repeated_occurrences = 0
    active_occurrences = 0
    for fragment, fragment_scores in scores.items():
        values = torch.stack(fragment_scores)
        frequency = float(spec.frequencies[fragment])
        positive_sum = positive_sum + frequency * torch.relu(
            1.0 - values / margin
        ).square().mean()
        positive_frequency_sum += frequency
        active_occurrences += int(values.numel())
        if values.numel() >= 2:
            count = int(values.numel())
            consistency_sum = (
                consistency_sum
                + count * torch.var(values, unbiased=False)
            )
            repeated_occurrences += count

    positive = positive_sum / (positive_frequency_sum + epsilon)
    if repeated_occurrences:
        inconsistency = consistency_sum / float(repeated_occurrences)
        consistency = consistency_sum / (
            margin.square() * float(repeated_occurrences) + epsilon
        )
    else:
        inconsistency = reference.new_zeros(())
        consistency = reference.new_zeros(())
    total = (
        positive_coefficient * positive
        + consistency_coefficient * consistency
    )
    return FragmentFrequencyLoss(
        total=total,
        positive=positive,
        consistency=consistency,
        inconsistency=inconsistency,
        active_fragment_count=len(scores),
        active_occurrence_count=active_occurrences,
    )
