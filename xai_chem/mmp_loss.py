"""Mmp loss for notebook-based atom-contribution experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence
import numpy as np
import pandas as pd
from .io import exact_integer
from typing import Mapping, Sequence
import torch
from torch import Tensor
from torch.nn import functional as F
from typing import Any, Mapping, Protocol, Sequence
from rdkit import Chem
from torch_geometric.data import Data


@dataclass(frozen=True)
class MMPPairRecord:
    """One oriented MMP constraint in zero-based dataset-row space."""

    pair_id: int
    i_id: int
    j_id: int
    dy: float
    similarity: float
    core_pairs: tuple[tuple[int, int], ...]
    core_pair_candidates: tuple[tuple[tuple[int, int], ...], ...]
    diff_i: tuple[int, ...]
    diff_j: tuple[int, ...]


def _records_json_indices(value: object, *, name: str) -> tuple[int, ...]:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a JSON string")
    # NOTE: CONTRACT in=value: JSON list encoded as text
    #       fail-fast: json.JSONDecodeError for malformed JSON is propagated
    parsed = json.loads(value)
    if not isinstance(parsed, list) or not parsed:
        raise ValueError(f"{name} must be a non-empty JSON list")
    indices = tuple(
        exact_integer(item, name=f"{name}[{position}]")
        for position, item in enumerate(parsed)
    )
    if any(index < 0 for index in indices):
        raise ValueError(f"{name} contains a negative atom index")
    if len(indices) != len(set(indices)):
        raise ValueError(f"{name} contains duplicate atom indices")
    return indices


def _json_core_pairs(value: object) -> tuple[tuple[int, int], ...]:
    if not isinstance(value, str):
        raise ValueError("core_pairs_json must be a JSON string")
    # NOTE: CONTRACT in=value: JSON list encoded as text
    parsed = json.loads(value)
    if not isinstance(parsed, list) or not parsed:
        raise ValueError("core_pairs_json must be a non-empty JSON list")
    pairs: list[tuple[int, int]] = []
    for position, item in enumerate(parsed):
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError(
                f"core_pairs_json[{position}] must contain two indices"
            )
        pair = (
            exact_integer(item[0], name=f"core_pairs_json[{position}][0]"),
            exact_integer(item[1], name=f"core_pairs_json[{position}][1]"),
        )
        if min(pair) < 0:
            raise ValueError("core_pairs_json contains a negative atom index")
        pairs.append(pair)
    core_i = [pair[0] for pair in pairs]
    core_j = [pair[1] for pair in pairs]
    if len(core_i) != len(set(core_i)) or len(core_j) != len(set(core_j)):
        raise ValueError("core_pairs_json is not a one-to-one atom mapping")
    return tuple(pairs)


def _json_core_pair_candidates(
    value: object,
    *,
    canonical: tuple[tuple[int, int], ...],
) -> tuple[tuple[tuple[int, int], ...], ...]:
    if not isinstance(value, str):
        raise ValueError("core_pair_candidates_json must be a JSON string")
    # NOTE: CONTRACT in=value: JSON list encoded as text
    parsed = json.loads(value)
    if not isinstance(parsed, list) or not parsed:
        raise ValueError(
            "core_pair_candidates_json must be a non-empty JSON list"
        )

    canonical_i = tuple(atom_i for atom_i, _atom_j in canonical)
    canonical_j = {atom_j for _atom_i, atom_j in canonical}
    candidates: list[tuple[tuple[int, int], ...]] = []
    for candidate_position, raw_candidate in enumerate(parsed):
        candidate = _json_core_pairs(json.dumps(raw_candidate))
        candidate_i = tuple(atom_i for atom_i, _atom_j in candidate)
        candidate_j = {atom_j for _atom_i, atom_j in candidate}
        if candidate_i != canonical_i:
            raise ValueError(
                "core_pair_candidates_json candidate "
                f"{candidate_position} changes canonical i atom order"
            )
        if candidate_j != canonical_j:
            raise ValueError(
                "core_pair_candidates_json candidate "
                f"{candidate_position} changes j core membership"
            )
        candidates.append(candidate)
    if len(candidates) != len(set(candidates)):
        raise ValueError("core_pair_candidates_json contains duplicates")
    if canonical not in candidates:
        raise ValueError(
            "core_pairs_json is absent from core_pair_candidates_json"
        )
    return tuple(candidates)


def load_mmp_pair_records(
    path: Path | str,
    *,
    train_row_ids: Iterable[int] | None = None,
) -> tuple[MMPPairRecord, ...]:
    """Parse every row, then retain pairs whose two endpoints are train rows."""

    table = pd.read_csv(path)
    required = {
        "pair_id",
        "i_id",
        "j_id",
        "dy",
        "fp_similarity",
        "core_pairs_json",
        "core_pair_candidates_json",
        "diff_i_json",
        "diff_j_json",
    }
    missing = sorted(required - set(table.columns))
    if missing:
        raise KeyError(f"MMP CSV is missing columns: {missing}")

    records: list[MMPPairRecord] = []
    pair_ids: set[int] = set()
    endpoint_pairs: set[tuple[int, int]] = set()
    for row_number, row in table.iterrows():
        pair_id = exact_integer(row["pair_id"], name=f"row {row_number} pair_id")
        i_id = exact_integer(row["i_id"], name=f"row {row_number} i_id")
        j_id = exact_integer(row["j_id"], name=f"row {row_number} j_id")
        if i_id < 0 or j_id < 0 or i_id == j_id:
            raise ValueError(f"row {row_number} has invalid MMP endpoints")
        dy = float(row["dy"])
        similarity = float(row["fp_similarity"])
        if not np.isfinite(dy):
            raise ValueError(f"row {row_number} dy must be finite")
        if not np.isfinite(similarity) or not 0.0 < similarity <= 1.0:
            raise ValueError(
                f"row {row_number} fp_similarity must satisfy 0 < value <= 1"
            )
        core_pairs = _json_core_pairs(row["core_pairs_json"])
        core_pair_candidates = _json_core_pair_candidates(
            row["core_pair_candidates_json"],
            canonical=core_pairs,
        )
        diff_i = _records_json_indices(row["diff_i_json"], name="diff_i_json")
        diff_j = _records_json_indices(row["diff_j_json"], name="diff_j_json")
        if set(diff_i) & {pair[0] for pair in core_pairs}:
            raise ValueError(f"row {row_number} i core/difference atoms overlap")
        if set(diff_j) & {pair[1] for pair in core_pairs}:
            raise ValueError(f"row {row_number} j core/difference atoms overlap")
        if pair_id in pair_ids:
            raise ValueError(f"Duplicate pair_id: {pair_id}")
        endpoints = (i_id, j_id)
        if endpoints in endpoint_pairs or endpoints[::-1] in endpoint_pairs:
            raise ValueError(f"Duplicate molecule pair: {endpoints}")
        pair_ids.add(pair_id)
        endpoint_pairs.add(endpoints)
        records.append(
            MMPPairRecord(
                pair_id=pair_id,
                i_id=i_id,
                j_id=j_id,
                dy=dy,
                similarity=similarity,
                core_pairs=core_pairs,
                core_pair_candidates=core_pair_candidates,
                diff_i=diff_i,
                diff_j=diff_j,
            )
        )

    if train_row_ids is not None:
        train_ids = {
            exact_integer(value, name="train_row_ids") for value in train_row_ids
        }
        records = [
            record
            for record in records
            if record.i_id in train_ids and record.j_id in train_ids
        ]
    return tuple(records)


def iter_pair_batches(
    records: Sequence[MMPPairRecord],
    *,
    batch_size: int,
    seed: int,
    epoch: int,
) -> Iterator[tuple[MMPPairRecord, ...]]:
    """Yield each declared pair once in a deterministic epoch-specific order."""

    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError("batch_size must be an integer")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise ValueError("epoch must be a non-negative integer")
    generator = np.random.default_rng(np.random.SeedSequence([seed, epoch]))
    order = generator.permutation(len(records))
    for start in range(0, len(order), batch_size):
        yield tuple(records[int(index)] for index in order[start : start + batch_size])


@dataclass(frozen=True)
class MMPPairLoss:
    """Differentiable MMP objective and its named components."""

    total: Tensor
    core: Tensor
    difference: Tensor
    pair_count: int


@dataclass(frozen=True)
class _PackedMMPBatch:
    """GPU-ready indices and constants for one explicit relationship batch."""

    atom_values: Tensor
    core_i: Tensor
    core_j: Tensor
    core_candidate: Tensor
    core_candidate_count: Tensor
    core_candidate_pair: Tensor
    diff_i: Tensor
    diff_j: Tensor
    diff_i_pair: Tensor
    diff_j_pair: Tensor
    difference_target: Tensor
    difference_scale: Tensor
    similarity: Tensor
    pair_count: int


def _pack_mmp_batch(
    contributions: Mapping[int, Tensor],
    records: Sequence[MMPPairRecord],
    *,
    target_alpha: float,
) -> _PackedMMPBatch:
    """Pack ragged atom mappings into a small set of batch GPU tensors."""

    # NOTE: CONTRACT records and scalar settings were validated during fold
    # preparation; contribution vectors share the model output device and dtype.
    atom_offsets: dict[int, int] = {}
    atom_parts = []
    atom_offset = 0
    for graph_id, values in contributions.items():
        atom_offsets[graph_id] = atom_offset
        atom_parts.append(values)
        atom_offset += values.numel()
    atom_values = torch.cat(atom_parts)

    core_i: list[int] = []
    core_j: list[int] = []
    core_candidate: list[int] = []
    core_candidate_count: list[float] = []
    core_candidate_pair: list[int] = []
    diff_i: list[int] = []
    diff_j: list[int] = []
    diff_i_pair: list[int] = []
    diff_j_pair: list[int] = []
    difference_target: list[float] = []
    difference_scale: list[float] = []
    similarity: list[float] = []
    candidate_index = 0
    for pair_index, record in enumerate(records):
        offset_i = atom_offsets[record.i_id]
        offset_j = atom_offsets[record.j_id]
        for candidate in record.core_pair_candidates:
            core_i.extend(offset_i + left for left, _ in candidate)
            core_j.extend(offset_j + right for _, right in candidate)
            core_candidate.extend([candidate_index] * len(candidate))
            core_candidate_count.append(float(len(candidate)))
            core_candidate_pair.append(pair_index)
            candidate_index += 1

        diff_i.extend(offset_i + index for index in record.diff_i)
        diff_j.extend(offset_j + index for index in record.diff_j)
        diff_i_pair.extend([pair_index] * len(record.diff_i))
        diff_j_pair.extend([pair_index] * len(record.diff_j))
        difference_target.append(record.dy / float(target_alpha))
        difference_scale.append(
            float(len(record.diff_i) + len(record.diff_j)) ** -0.5
        )
        similarity.append(record.similarity)

    device = atom_values.device
    return _PackedMMPBatch(
        atom_values=atom_values,
        core_i=torch.tensor(core_i, dtype=torch.long, device=device),
        core_j=torch.tensor(core_j, dtype=torch.long, device=device),
        core_candidate=torch.tensor(
            core_candidate,
            dtype=torch.long,
            device=device,
        ),
        core_candidate_count=atom_values.new_tensor(core_candidate_count),
        core_candidate_pair=torch.tensor(
            core_candidate_pair,
            dtype=torch.long,
            device=device,
        ),
        diff_i=torch.tensor(diff_i, dtype=torch.long, device=device),
        diff_j=torch.tensor(diff_j, dtype=torch.long, device=device),
        diff_i_pair=torch.tensor(diff_i_pair, dtype=torch.long, device=device),
        diff_j_pair=torch.tensor(diff_j_pair, dtype=torch.long, device=device),
        difference_target=atom_values.new_tensor(difference_target),
        difference_scale=atom_values.new_tensor(difference_scale),
        similarity=atom_values.new_tensor(similarity),
        pair_count=len(records),
    )


def compute_mmp_pair_loss(
    contributions: Mapping[int, Tensor],
    records: Sequence[MMPPairRecord],
    *,
    target_alpha: float,
    core_weight: float = 5.0,
    difference_weight: float = 1.0,
    huber_beta: float = 0.25,
) -> MMPPairLoss:
    """Compute the automorphism-aware explicit-pair objective.

    Atom contributions are in the scaled target space
    ``z=(y-beta)/target_alpha``. Each raw MMP activity difference is therefore
    divided by ``target_alpha``; the affine offset cancels exactly. The shared-
    core loss is the minimum atomwise SmoothL1 over all declared valid
    automorphisms. Packing only changes execution structure: all reductions and
    objective weights retain the historical definitions.
    """

    # NOTE: CONTRACT callers provide a non-empty validated record batch and
    # positive finite scalar settings from TrainingConfig.
    packed = _pack_mmp_batch(
        contributions,
        records,
        target_alpha=target_alpha,
    )

    core_delta = (
        packed.atom_values[packed.core_i]
        - packed.atom_values[packed.core_j]
    )
    core_atom_loss = F.smooth_l1_loss(
        core_delta,
        torch.zeros_like(core_delta),
        beta=float(huber_beta),
        reduction="none",
    )
    candidate_sum = packed.atom_values.new_zeros(
        packed.core_candidate_count.shape
    ).scatter_add_(0, packed.core_candidate, core_atom_loss)
    candidate_mean = candidate_sum / packed.core_candidate_count
    pair_core = packed.atom_values.new_full(
        (packed.pair_count,),
        torch.inf,
    ).scatter_reduce_(
        0,
        packed.core_candidate_pair,
        candidate_mean,
        reduce="amin",
        include_self=True,
    )
    core = pair_core.mean()

    difference_i = packed.atom_values.new_zeros(
        packed.pair_count
    ).scatter_add_(
        0,
        packed.diff_i_pair,
        packed.atom_values[packed.diff_i],
    )
    difference_j = packed.atom_values.new_zeros(
        packed.pair_count
    ).scatter_add_(
        0,
        packed.diff_j_pair,
        packed.atom_values[packed.diff_j],
    )
    normalized_residual = (
        difference_i - difference_j - packed.difference_target
    ) * packed.difference_scale
    difference_terms = F.smooth_l1_loss(
        normalized_residual,
        torch.zeros_like(normalized_residual),
        beta=float(huber_beta),
        reduction="none",
    )
    difference = (
        packed.similarity * difference_terms
    ).sum() / packed.similarity.sum()
    total = (
        float(core_weight) * core + float(difference_weight) * difference
    ) / float(core_weight + difference_weight)
    return MMPPairLoss(
        total=total,
        core=core,
        difference=difference,
        pair_count=packed.pair_count,
    )


_MAX_SUBSTRUCTURE_MATCHES = 10_001


class _GraphDataset(Protocol):
    """Minimal cache-backed graph dataset interface needed for MMP mapping."""

    def __getitem__(self, position: int) -> Data: ...

    def position_for_source_row(self, source_row_index: int) -> int: ...


@dataclass(frozen=True)
class MMPAtomMappingSource:
    """Source fields needed to reconstruct one common-core correspondence."""

    core_smiles: str
    smiles_i: str
    smiles_j: str
    core_i: tuple[int, ...]
    core_j: tuple[int, ...]
    attachment_i: tuple[int, ...]
    attachment_j: tuple[int, ...]
    attachment_variable_i: tuple[int, ...]
    attachment_variable_j: tuple[int, ...]


@dataclass(frozen=True)
class MMPAtomMappingResult:
    """Graph-order records and a JSON-compatible reconstruction check."""

    records: tuple[MMPPairRecord, ...]
    check: Mapping[str, Any]


@dataclass(frozen=True)
class _ChemicalMapping:
    candidates: tuple[tuple[tuple[int, int], ...], ...]
    valid_match_count_i: int
    valid_match_count_j: int


def _mmp_mapping_json_indices(
    value: object,
    *,
    field: str,
    pair_id: int,
) -> tuple[int, ...]:
    if not isinstance(value, str):
        raise TypeError(f"MMP pair {pair_id} {field} must be JSON text")
    # NOTE: CONTRACT in=value: JSON list encoded as text
    payload = json.loads(value)
    if not isinstance(payload, list):
        raise TypeError(f"MMP pair {pair_id} {field} must decode to a list")
    indices: list[int] = []
    for position, item in enumerate(payload):
        if isinstance(item, bool) or not isinstance(item, int):
            raise TypeError(
                f"MMP pair {pair_id} {field}[{position}] must be an integer"
            )
        indices.append(item)
    if len(indices) != len(set(indices)):
        raise ValueError(f"MMP pair {pair_id} {field} contains duplicates")
    return tuple(indices)


def load_mmp_atom_mapping_sources(
    table: pd.DataFrame,
    records: Sequence[MMPPairRecord],
    *,
    expected_mapping_method: str,
) -> Mapping[int, MMPAtomMappingSource]:
    """Load and validate source declarations for all parsed MMP records."""

    required = {
        "pair_id",
        "core_smiles",
        "smiles_i",
        "smiles_j",
        "core_i_json",
        "core_j_json",
        "attachment_i_json",
        "attachment_j_json",
        "attachment_variable_i_json",
        "attachment_variable_j_json",
        "core_mapping_method",
    }
    missing = sorted(required - set(table.columns))
    if missing:
        raise KeyError(
            "MMP chemical-mapping columns are missing: " + ", ".join(missing)
        )
    rows_by_pair_id: dict[int, pd.Series] = {}
    for _, row in table.iterrows():
        raw_pair_id = row["pair_id"]
        if isinstance(raw_pair_id, bool):
            raise TypeError("MMP pair_id must be an integer")
        pair_id = int(raw_pair_id)
        if float(raw_pair_id) != float(pair_id):
            raise TypeError("MMP pair_id must be an exact integer")
        if pair_id in rows_by_pair_id:
            raise ValueError("MMP pair_id values are not unique")
        rows_by_pair_id[pair_id] = row

    record_ids = {record.pair_id for record in records}
    if record_ids != set(rows_by_pair_id):
        raise ValueError(
            "MMP parsed records and chemical source rows have different pair IDs"
        )

    sources: dict[int, MMPAtomMappingSource] = {}
    for record in records:
        row = rows_by_pair_id[record.pair_id]
        method = row["core_mapping_method"]
        if not isinstance(method, str) or method != expected_mapping_method:
            raise ValueError(
                f"MMP pair {record.pair_id} core mapping method differs "
                "from the declared run contract"
            )
        core_smiles = row["core_smiles"]
        smiles_i = row["smiles_i"]
        smiles_j = row["smiles_j"]
        if not isinstance(core_smiles, str) or not core_smiles.strip():
            raise ValueError(
                f"MMP pair {record.pair_id} core_smiles must be non-empty"
            )
        if not isinstance(smiles_i, str) or not smiles_i.strip():
            raise ValueError(
                f"MMP pair {record.pair_id} smiles_i must be non-empty"
            )
        if not isinstance(smiles_j, str) or not smiles_j.strip():
            raise ValueError(
                f"MMP pair {record.pair_id} smiles_j must be non-empty"
            )

        core_i = _mmp_mapping_json_indices(
            row["core_i_json"],
            field="core_i_json",
            pair_id=record.pair_id,
        )
        core_j = _mmp_mapping_json_indices(
            row["core_j_json"],
            field="core_j_json",
            pair_id=record.pair_id,
        )
        record_core_i = tuple(atom_i for atom_i, _ in record.core_pairs)
        record_core_j = tuple(atom_j for _, atom_j in record.core_pairs)
        if set(core_i) != set(record_core_i):
            raise ValueError(
                f"MMP pair {record.pair_id} core_i disagrees with "
                "core_pairs_json membership"
            )
        if set(core_j) != set(record_core_j):
            raise ValueError(
                f"MMP pair {record.pair_id} core_j disagrees with "
                "core_pairs_json membership"
            )

        attachment_i = _mmp_mapping_json_indices(
            row["attachment_i_json"],
            field="attachment_i_json",
            pair_id=record.pair_id,
        )
        attachment_j = _mmp_mapping_json_indices(
            row["attachment_j_json"],
            field="attachment_j_json",
            pair_id=record.pair_id,
        )
        attachment_variable_i = _mmp_mapping_json_indices(
            row["attachment_variable_i_json"],
            field="attachment_variable_i_json",
            pair_id=record.pair_id,
        )
        attachment_variable_j = _mmp_mapping_json_indices(
            row["attachment_variable_j_json"],
            field="attachment_variable_j_json",
            pair_id=record.pair_id,
        )
        for endpoint, core, difference, core_attachment, variable_attachment in (
            (
                "i",
                set(core_i),
                set(record.diff_i),
                attachment_i,
                attachment_variable_i,
            ),
            (
                "j",
                set(core_j),
                set(record.diff_j),
                attachment_j,
                attachment_variable_j,
            ),
        ):
            if not core_attachment or len(core_attachment) != len(
                variable_attachment
            ):
                raise ValueError(
                    f"MMP pair {record.pair_id} {endpoint} attachment "
                    "declarations must be non-empty and one-to-one"
                )
            if not set(core_attachment).issubset(core):
                raise ValueError(
                    f"MMP pair {record.pair_id} {endpoint} core attachment "
                    "is outside the shared core"
                )
            if not set(variable_attachment).issubset(difference):
                raise ValueError(
                    f"MMP pair {record.pair_id} {endpoint} variable "
                    "attachment is outside difference atoms"
                )
        sources[record.pair_id] = MMPAtomMappingSource(
            core_smiles=core_smiles,
            smiles_i=smiles_i,
            smiles_j=smiles_j,
            core_i=core_i,
            core_j=core_j,
            attachment_i=attachment_i,
            attachment_j=attachment_j,
            attachment_variable_i=attachment_variable_i,
            attachment_variable_j=attachment_variable_j,
        )
    return sources


def _common_core_queries(
    core_smiles: str,
    *,
    pair_id: int,
) -> tuple[Chem.Mol, Chem.Mol, tuple[int, ...], tuple[int, ...]]:
    """Construct shared core queries without changing endpoint atom identities."""

    original = Chem.MolFromSmiles(core_smiles)
    if original is None:
        raise ValueError(
            f"MMP pair {pair_id} core_smiles cannot be parsed by RDKit"
        )
    dummy_indices = tuple(
        atom.GetIdx()
        for atom in original.GetAtoms()
        if atom.GetAtomicNum() == 0
    )
    core_query_indices = tuple(
        atom.GetIdx()
        for atom in original.GetAtoms()
        if atom.GetAtomicNum() != 0
    )
    if not dummy_indices:
        raise ValueError(
            f"MMP pair {pair_id} core_smiles has no attachment dummy atom"
        )
    editable = Chem.RWMol(original)
    for atom_index in dummy_indices:
        editable.ReplaceAtom(
            atom_index,
            Chem.AtomFromSmarts("*"),
            preserveProps=False,
        )
    return original, editable.GetMol(), core_query_indices, dummy_indices


def _canonical_valid_match(
    molecule: Chem.Mol,
    query: Chem.Mol,
    *,
    core_query_indices: tuple[int, ...],
    dummy_indices: tuple[int, ...],
    declared_core: frozenset[int],
    declared_attachment: frozenset[int],
    pair_id: int,
    endpoint: str,
) -> tuple[tuple[int, ...], int]:
    """Select the canonical chemically valid match among core automorphisms."""

    raw_matches = molecule.GetSubstructMatches(
        query,
        uniquify=False,
        useChirality=True,
        maxMatches=_MAX_SUBSTRUCTURE_MATCHES,
    )
    if len(raw_matches) >= _MAX_SUBSTRUCTURE_MATCHES:
        raise ValueError(
            f"MMP pair {pair_id} endpoint {endpoint} exceeded the explicit "
            "substructure-match cap"
        )
    valid = {
        tuple(int(atom) for atom in match)
        for match in raw_matches
        if frozenset(match[index] for index in core_query_indices)
        == declared_core
        and frozenset(match[index] for index in dummy_indices)
        == declared_attachment
    }
    if not valid:
        raise ValueError(
            f"MMP pair {pair_id} endpoint {endpoint} has no exact shared-core "
            "substructure match consistent with its declared partition and "
            "variable attachment"
        )
    ranks = tuple(
        int(rank)
        for rank in Chem.CanonicalRankAtoms(
            molecule,
            breakTies=True,
            includeChirality=True,
            includeIsotopes=True,
        )
    )
    canonical = min(
        valid,
        key=lambda match: (
            tuple(ranks[match[index]] for index in core_query_indices),
            match,
        ),
    )
    return canonical, len(valid)


def _validate_partition(
    record: MMPPairRecord,
    *,
    atom_count_i: int,
    atom_count_j: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Keep core and changed regions disjoint and exhaustive."""

    core_i = tuple(atom_i for atom_i, _ in record.core_pairs)
    core_j = tuple(atom_j for _, atom_j in record.core_pairs)
    for endpoint, core, difference, atom_count in (
        ("i", core_i, record.diff_i, atom_count_i),
        ("j", core_j, record.diff_j, atom_count_j),
    ):
        values = core + difference
        if not core or not difference:
            raise ValueError(
                f"MMP pair {record.pair_id} endpoint {endpoint} has an "
                "empty core or difference partition"
            )
        if any(atom < 0 or atom >= atom_count for atom in values):
            raise IndexError(
                f"MMP pair {record.pair_id} endpoint {endpoint} has an "
                "out-of-range atom index"
            )
        if len(values) != len(set(values)):
            raise ValueError(
                f"MMP pair {record.pair_id} endpoint {endpoint} core and "
                "difference atoms overlap or contain duplicates"
            )
        if set(values) != set(range(atom_count)):
            raise ValueError(
                f"MMP pair {record.pair_id} endpoint {endpoint} core and "
                "difference atoms are not an exact molecular partition"
            )
    return core_i, core_j


def _bond_signature(bond: Chem.Bond | None) -> tuple[float, bool] | None:
    if bond is None:
        return None
    return float(bond.GetBondTypeAsDouble()), bool(bond.GetIsAromatic())


def _normalized_candidate(
    candidate: Sequence[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    return tuple(
        sorted((int(atom_i), int(atom_j)) for atom_i, atom_j in candidate)
    )


def _validate_record_chemical_contract(
    record: MMPPairRecord,
    source: MMPAtomMappingSource,
) -> _ChemicalMapping:
    """Check atom elements and bonds before comparing endpoint ACs."""

    mol_i = Chem.MolFromSmiles(source.smiles_i)
    mol_j = Chem.MolFromSmiles(source.smiles_j)
    if mol_i is None or mol_j is None:
        raise ValueError(
            f"MMP pair {record.pair_id} endpoint SMILES cannot be parsed by RDKit"
        )
    core_i, core_j = _validate_partition(
        record,
        atom_count_i=mol_i.GetNumAtoms(),
        atom_count_j=mol_j.GetNumAtoms(),
    )
    if set(source.core_i) != set(core_i) or set(source.core_j) != set(core_j):
        raise ValueError(
            f"MMP pair {record.pair_id} source core membership is inconsistent"
        )
    for endpoint, molecule, core_attachment, variable_attachment in (
        (
            "i",
            mol_i,
            source.attachment_i,
            source.attachment_variable_i,
        ),
        (
            "j",
            mol_j,
            source.attachment_j,
            source.attachment_variable_j,
        ),
    ):
        attachment_bonds = {
            (core_atom, variable_atom)
            for core_atom in core_attachment
            for variable_atom in variable_attachment
            if molecule.GetBondBetweenAtoms(core_atom, variable_atom)
            is not None
        }
        attached_core = {core_atom for core_atom, _ in attachment_bonds}
        attached_variable = {
            variable_atom for _, variable_atom in attachment_bonds
        }
        if (
            len(attachment_bonds) != len(core_attachment)
            or attached_core != set(core_attachment)
            or attached_variable != set(variable_attachment)
        ):
            raise ValueError(
                f"MMP pair {record.pair_id} endpoint {endpoint} cut "
                "attachments are not one-to-one"
            )

    original_query, search_query, query_core, query_dummy = (
        _common_core_queries(source.core_smiles, pair_id=record.pair_id)
    )
    match_i, count_i = _canonical_valid_match(
        mol_i,
        search_query,
        core_query_indices=query_core,
        dummy_indices=query_dummy,
        declared_core=frozenset(core_i),
        declared_attachment=frozenset(source.attachment_variable_i),
        pair_id=record.pair_id,
        endpoint="i",
    )
    match_j, count_j = _canonical_valid_match(
        mol_j,
        search_query,
        core_query_indices=query_core,
        dummy_indices=query_dummy,
        declared_core=frozenset(core_j),
        declared_attachment=frozenset(source.attachment_variable_j),
        pair_id=record.pair_id,
        endpoint="j",
    )
    raw_automorphisms = original_query.GetSubstructMatches(
        original_query,
        uniquify=False,
        useChirality=True,
        maxMatches=_MAX_SUBSTRUCTURE_MATCHES,
    )
    if len(raw_automorphisms) >= _MAX_SUBSTRUCTURE_MATCHES:
        raise ValueError(
            f"MMP pair {record.pair_id} core automorphisms exceeded the "
            "explicit match cap"
        )
    query_core_set = set(query_core)
    expected_candidates = {
        _normalized_candidate(
            tuple(
                (match_i[query_index], match_j[int(automorphism[query_index])])
                for query_index in query_core
            )
        )
        for automorphism in raw_automorphisms
        if all(
            int(automorphism[query_index]) in query_core_set
            for query_index in query_core
        )
    }
    if not expected_candidates:
        raise ValueError(
            f"MMP pair {record.pair_id} shared core has no valid automorphism"
        )
    declared_candidates = {
        _normalized_candidate(candidate)
        for candidate in record.core_pair_candidates
    }
    if declared_candidates != expected_candidates:
        raise ValueError(
            f"MMP pair {record.pair_id} declared automorphism candidates do "
            "not equal the independently reconstructed shared-core orbit"
        )
    if _normalized_candidate(record.core_pairs) not in declared_candidates:
        raise ValueError(
            f"MMP pair {record.pair_id} canonical core mapping is not a "
            "declared automorphism candidate"
        )

    for candidate in declared_candidates:
        for atom_i, atom_j in candidate:
            if (
                mol_i.GetAtomWithIdx(atom_i).GetAtomicNum()
                != mol_j.GetAtomWithIdx(atom_j).GetAtomicNum()
            ):
                raise ValueError(
                    f"MMP pair {record.pair_id} maps unequal core elements "
                    f"at ({atom_i}, {atom_j})"
                )
        mapping = dict(candidate)
        i_atoms = tuple(mapping)
        for left_position, atom_i_a in enumerate(i_atoms):
            for atom_i_b in i_atoms[left_position + 1 :]:
                if _bond_signature(
                    mol_i.GetBondBetweenAtoms(atom_i_a, atom_i_b)
                ) != _bond_signature(
                    mol_j.GetBondBetweenAtoms(
                        mapping[atom_i_a],
                        mapping[atom_i_b],
                    )
                ):
                    raise ValueError(
                        f"MMP pair {record.pair_id} candidate changes shared-"
                        "core bond topology"
                    )
    return _ChemicalMapping(
        candidates=tuple(sorted(declared_candidates)),
        valid_match_count_i=count_i,
        valid_match_count_j=count_j,
    )


def validate_mmp_source_chemistry(
    records: Sequence[MMPPairRecord],
    sources: Mapping[int, MMPAtomMappingSource],
) -> Mapping[str, Any]:
    """Validate every source MMP row, including pairs later excluded by folds."""

    if {record.pair_id for record in records} != set(sources):
        raise ValueError("MMP records and mapping sources have different pair IDs")
    candidate_total = 0
    ambiguous_pairs = 0
    maximum_candidates = 0
    ambiguous_i = 0
    ambiguous_j = 0
    for record in records:
        chemical = _validate_record_chemical_contract(
            record,
            sources[record.pair_id],
        )
        count = len(chemical.candidates)
        candidate_total += count
        ambiguous_pairs += int(count > 1)
        maximum_candidates = max(maximum_candidates, count)
        ambiguous_i += int(chemical.valid_match_count_i > 1)
        ambiguous_j += int(chemical.valid_match_count_j > 1)
    return {
        "validated_source_pair_count": len(records),
        "declared_candidate_mapping_count": candidate_total,
        "symmetry_ambiguous_pair_count": ambiguous_pairs,
        "maximum_automorphism_candidate_count": maximum_candidates,
        "ambiguous_i_endpoint_count": ambiguous_i,
        "ambiguous_j_endpoint_count": ambiguous_j,
        "element_mismatch_count": 0,
        "bond_topology_mismatch_count": 0,
        "attachment_bond_mismatch_count": 0,
        "interpretation": (
            "candidate mappings form the finite shared-core automorphism orbit; "
            "the core objective is a quotient distance that treats atom "
            "correspondence as a latent nuisance variable, not evidence that "
            "symmetry-related atoms have physically identical contributions"
        ),
    }


def _bijective_graph_mapping(
    graph: Data,
    *,
    molecule: Chem.Mol,
    pair_id: int,
    endpoint: str,
) -> torch.Tensor:
    """Translate SMILES-local indices into a bijective graph atom order."""

    atom_count = molecule.GetNumAtoms()
    mapping = graph.smiles_to_graph_atom_index.to(
        dtype=torch.long,
        device="cpu",
    )
    if mapping.ndim != 1 or mapping.numel() != atom_count:
        raise ValueError(
            f"MMP pair {pair_id} endpoint {endpoint} graph atom mapping has "
            "the wrong shape"
        )
    if not torch.equal(
        torch.sort(mapping).values,
        torch.arange(atom_count, dtype=torch.long),
    ):
        raise ValueError(
            f"MMP pair {pair_id} endpoint {endpoint} graph atom mapping is "
            "not bijective"
        )
    atomic_numbers = graph.atomic_numbers.to(dtype=torch.long, device="cpu")
    if atomic_numbers.ndim != 1 or atomic_numbers.numel() != atom_count:
        raise ValueError(
            f"MMP pair {pair_id} endpoint {endpoint} atomic_numbers has the "
            "wrong shape"
        )
    for smiles_atom, graph_atom in enumerate(mapping.tolist()):
        expected = molecule.GetAtomWithIdx(smiles_atom).GetAtomicNum()
        if int(atomic_numbers[graph_atom]) != expected:
            raise ValueError(
                f"MMP pair {pair_id} endpoint {endpoint} graph mapping "
                "does not preserve atomic numbers"
            )
    return mapping


def reconstruct_mmp_graph_records(
    dataset: _GraphDataset,
    records: Sequence[MMPPairRecord],
    sources: Mapping[int, MMPAtomMappingSource],
) -> MMPAtomMappingResult:
    """Validate retained pairs and convert every atom index to graph order."""

    endpoints = sorted(
        {
            endpoint
            for record in records
            for endpoint in (record.i_id, record.j_id)
        }
    )
    graphs = {
        source_row: dataset[dataset.position_for_source_row(source_row)]
        for source_row in endpoints
    }
    mapped_records: list[MMPPairRecord] = []
    ambiguous_pairs = 0
    maximum_candidates = 0
    for record in records:
        source = sources[record.pair_id]
        chemical = _validate_record_chemical_contract(record, source)
        ambiguous_pairs += int(len(chemical.candidates) > 1)
        maximum_candidates = max(
            maximum_candidates,
            len(chemical.candidates),
        )
        mol_i = Chem.MolFromSmiles(source.smiles_i)
        mol_j = Chem.MolFromSmiles(source.smiles_j)
        if mol_i is None or mol_j is None:
            raise AssertionError("SMILES parsed during chemical validation")
        mapping_i = _bijective_graph_mapping(
            graphs[record.i_id],
            molecule=mol_i,
            pair_id=record.pair_id,
            endpoint="i",
        )
        mapping_j = _bijective_graph_mapping(
            graphs[record.j_id],
            molecule=mol_j,
            pair_id=record.pair_id,
            endpoint="j",
        )

        def map_candidate(
            candidate: Sequence[tuple[int, int]],
        ) -> tuple[tuple[int, int], ...]:
            return tuple(
                (
                    int(mapping_i[atom_i]),
                    int(mapping_j[atom_j]),
                )
                for atom_i, atom_j in candidate
            )

        mapped_candidates = tuple(
            map_candidate(candidate)
            for candidate in record.core_pair_candidates
        )
        if len(mapped_candidates) != len(set(mapped_candidates)):
            raise ValueError(
                f"MMP pair {record.pair_id} graph-order automorphism "
                "candidates are not unique"
            )
        mapped_canonical = map_candidate(record.core_pairs)
        if mapped_canonical not in mapped_candidates:
            raise ValueError(
                f"MMP pair {record.pair_id} graph-order canonical mapping "
                "is absent from its candidate set"
            )
        mapped_records.append(
            MMPPairRecord(
                pair_id=record.pair_id,
                i_id=record.i_id,
                j_id=record.j_id,
                dy=record.dy,
                similarity=record.similarity,
                core_pairs=mapped_canonical,
                core_pair_candidates=mapped_candidates,
                diff_i=tuple(int(mapping_i[atom]) for atom in record.diff_i),
                diff_j=tuple(int(mapping_j[atom]) for atom in record.diff_j),
            )
        )
    return MMPAtomMappingResult(
        records=tuple(mapped_records),
        check={
            "method": "independent_shared_core_automorphism_reconstruction_v1",
            "retained_pair_count": len(mapped_records),
            "symmetry_ambiguous_pair_count": ambiguous_pairs,
            "maximum_automorphism_candidate_count": maximum_candidates,
            "graph_atom_mapping": (
                "explicit bijective smiles_to_graph_atom_index with atomic-"
                "number preservation"
            ),
            "symmetry_policy": (
                "all chemically valid shared-core automorphisms retained; "
                "downstream core loss takes their minimum"
            ),
            "scientific_limitation": (
                "automorphism quotienting removes an arbitrary atom-label "
                "choice; it does not assert physically identical atomic "
                "contributions across symmetry-related sites"
            ),
            "substructure_match_fail_closed_cap": _MAX_SUBSTRUCTURE_MATCHES,
        },
    )
