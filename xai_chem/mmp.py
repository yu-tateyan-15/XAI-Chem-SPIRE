"""Mmp for notebook-based atom-contribution experiments."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Iterable, Sequence
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from .io import exact_integer
from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence
from rdkit import Chem


DEFAULT_RDMMPA_PATTERN = "[#6+0;!$(*=,#[!#6])]!@!=!#[*]"


FRAGMENTATION_METHOD = "rdkit_fragment_on_bonds_with_rdmmpa_cut_smarts_v1"


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none"} else text


def load_graph_meta(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as fp:
        meta = json.load(fp)
    if isinstance(meta, dict):
        records = meta.get("graphs")
        if not isinstance(records, list):
            raise TypeError(
                f"Expected graph manifest records at {path}:graphs"
            )
        return [
            {
                **record,
                "file": record.get("file", record.get("filename", "")),
            }
            for record in records
        ]
    if isinstance(meta, list):
        return meta
    raise TypeError(f"Expected graph meta list or manifest, got {type(meta)}")


def build_pair_metric_key(i_id: int, j_id: int) -> str:
    return f"{int(i_id)}::{int(j_id)}"


def build_fragment_pair_key(
    i_id: int,
    j_id: int,
    *,
    cut_count: int,
    core_smiles: str,
    variable_i_smiles: str,
    variable_j_smiles: str,
) -> str:
    payload = json.dumps(
        {
            "cut_count": int(cut_count),
            "core_smiles": str(core_smiles),
            "variable_i_smiles": str(variable_i_smiles),
            "variable_j_smiles": str(variable_j_smiles),
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    suffix = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
    return f"{int(i_id)}::{int(j_id)}::frag::{suffix}"


def build_dataset_entries(
    dataset_rows: Sequence[dict],
    graph_meta: Sequence[dict],
    *,
    morgan_radius: int,
    morgan_nbits: int,
) -> tuple[list[dict], Counter]:
    """Bind molecular graphs, target values and document groups before fragmentation."""

    graph_meta_by_filename: dict[str, dict] = {}
    graph_meta_by_molecule_id: dict[int, dict] = {}
    for meta_index, graph_row in enumerate(graph_meta):
        if not isinstance(graph_row, dict):
            raise TypeError(f"graph_meta[{meta_index}] must be a dictionary")
        graph_file = normalize_text(
            graph_row.get("file", graph_row.get("filename"))
        )
        if not graph_file:
            raise ValueError(f"graph_meta[{meta_index}] is missing required file")
        if graph_file in graph_meta_by_filename:
            raise ValueError(
                f"Duplicate graph metadata filename: {graph_file}"
            )
        graph_meta_by_filename[graph_file] = graph_row
        raw_molecule_id = graph_row.get("molecule_id")
        if raw_molecule_id is None:
            stem_digits = Path(graph_file).stem.removeprefix("graph_")
            raw_molecule_id = stem_digits if stem_digits.isdigit() else None
        if raw_molecule_id is not None:
            molecule_id = exact_integer(
                raw_molecule_id,
                name=f"graph_meta[{meta_index}] molecule_id",
            )
            if molecule_id in graph_meta_by_molecule_id:
                raise ValueError(
                    f"Duplicate graph metadata molecule_id: {molecule_id}"
                )
            graph_meta_by_molecule_id[molecule_id] = graph_row

    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=int(morgan_radius),
        fpSize=int(morgan_nbits),
    )
    entries: list[dict] = []
    stats: Counter = Counter()
    dataset_indices: set[int] = set()
    used_graph_files: set[str] = set()
    for row_idx, row in enumerate(dataset_rows):
        stats["rows_seen"] += 1
        context = f"dataset_rows[{row_idx}]"
        if not isinstance(row, dict):
            raise TypeError(f"{context} must be a dictionary")
        if "index" not in row or row["index"] is None:
            raise ValueError(f"{context} is missing required index")
        dataset_index = exact_integer(
            row["index"],
            name=f"{context} index",
        )
        if dataset_index in dataset_indices:
            raise ValueError(f"{context} duplicates dataset index {dataset_index}")
        dataset_indices.add(dataset_index)

        molecule_id = normalize_text(row.get("cid") or row.get("number"))
        document_text = normalize_text(
            row.get("document_chembl_ids")
            or row.get("Document ChEMBL ID")
        )
        document_ids = tuple(
            dict.fromkeys(
                part.strip()
                for part in document_text.split("|")
                if part.strip()
            )
        )
        smiles = normalize_text(row.get("Smiles"))
        if isinstance(row.get("pIC50"), bool):
            raise ValueError(f"{context} pIC50 must be a finite number")
        try:
            pic50 = float(row.get("pIC50"))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{context} pIC50 must be a finite number"
            ) from exc
        if not math.isfinite(pic50):
            raise ValueError(f"{context} pIC50 must be a finite number")
        if not molecule_id:
            raise ValueError(f"{context} is missing required number")
        if not document_ids:
            raise ValueError(
                f"{context} is missing required Document ChEMBL ID/"
                "document_chembl_ids"
            )
        if not smiles:
            raise ValueError(f"{context} is missing required Smiles")

        graph_row = graph_meta_by_molecule_id.get(dataset_index)
        if graph_row is None:
            raise ValueError(
                f"{context} has no graph metadata for molecule_id "
                f"{dataset_index}"
            )
        graph_file = normalize_text(
            graph_row.get("file", graph_row.get("filename"))
        )
        used_graph_files.add(graph_file)
        graph_source_number = normalize_text(graph_row.get("source_number"))
        if graph_source_number and graph_source_number != molecule_id:
            raise ValueError(
                f"{context} molecule id {molecule_id!r} differs from "
                f"graph source_number {graph_source_number!r}"
            )

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"{context} contains invalid Smiles")
        canonical_isomeric_smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        canonical_nonisomeric_smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)

        base_entry = {
            "row_idx": int(row_idx),
            "dataset_index": dataset_index,
            "molecule_id": molecule_id,
            "smiles": smiles,
            "canonical_isomeric_smiles": canonical_isomeric_smiles,
            "canonical_nonisomeric_smiles": canonical_nonisomeric_smiles,
            "pIC50": float(pic50),
            "graph_file": graph_file,
            "mol": mol,
            "fingerprint": generator.GetFingerprint(mol),
            "n_atoms": int(mol.GetNumAtoms()),
        }
        for document_id in document_ids:
            entries.append(
                {
                    **base_entry,
                    "document_id": document_id,
                }
            )
            stats["kept_entries"] += 1
        stats["kept_molecules"] += 1
        stats["document_memberships"] += len(document_ids)
        if len(document_ids) > 1:
            stats["multi_document_molecules"] += 1
    unused_graph_files = sorted(set(graph_meta_by_filename) - used_graph_files)
    if unused_graph_files:
        raise ValueError(
            "Graph metadata contains no matching dataset row: "
            + ", ".join(unused_graph_files[:5])
        )
    return entries, stats


def cuttable_bonds(mol: Chem.Mol, pattern: str, max_cut_bonds: int) -> list[tuple[int, int, int]]:
    """Select bonds matching the RDKit MMPA cut SMARTS."""

    query = Chem.MolFromSmarts(pattern)
    if query is None:
        raise ValueError(f"Invalid RDKit MMPA cut SMARTS: {pattern}")
    bond_rows: set[tuple[int, int, int]] = set()
    for atom_i, atom_j in mol.GetSubstructMatches(query):
        bond = mol.GetBondBetweenAtoms(int(atom_i), int(atom_j))
        if bond is not None:
            left, right = sorted([int(atom_i), int(atom_j)])
            bond_rows.add((int(bond.GetIdx()), left, right))
    out = sorted(bond_rows)
    return out[: int(max_cut_bonds)]


CORE_MAPPING_METHOD = "rdkit_canonical_fragment_smiles_atom_output_order_v1"


def _component_smiles_and_atom_order(
    frag_mol: Chem.Mol,
    frag_atom_indices: Iterable[int],
) -> tuple[str, list[int]]:
    """Retain original atom identity through canonical fragment traversal."""

    atom_indices = [int(idx) for idx in frag_atom_indices]
    smiles = Chem.MolFragmentToSmiles(
        frag_mol,
        atomsToUse=atom_indices,
        canonical=True,
        isomericSmiles=True,
    )
    if not frag_mol.HasProp("_smilesAtomOutputOrder"):
        raise ValueError("RDKit did not expose canonical SMILES atom output order")
    try:
        output_order = [
            int(idx)
            for idx in json.loads(
                frag_mol.GetProp("_smilesAtomOutputOrder")
            )
        ]
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid RDKit canonical SMILES atom output order") from exc
    if len(output_order) != len(atom_indices) or set(output_order) != set(
        atom_indices
    ):
        raise ValueError(
            "RDKit canonical SMILES atom output order does not match "
            "the fragment component"
        )
    query = Chem.MolFromSmiles(smiles)
    if query is None or query.GetNumAtoms() != len(output_order):
        raise ValueError("Canonical fragment SMILES does not round-trip")
    return smiles, output_order


def _fragment_components(
    mol: Chem.Mol,
    cut_combo: Sequence[tuple[int, int, int]],
) -> list[dict]:
    """Split selected bonds while preserving each fragment atom correspondence."""

    dummy_labels = [(idx + 1, idx + 1) for idx, _row in enumerate(cut_combo)]
    frag_mol = Chem.FragmentOnBonds(
        mol,
        [int(row[0]) for row in cut_combo],
        addDummies=True,
        dummyLabels=dummy_labels,
    )
    n_original_atoms = int(mol.GetNumAtoms())
    components: list[dict] = []
    for frag_indices in Chem.GetMolFrags(frag_mol, asMols=False, sanitizeFrags=False):
        frag_indices = tuple(int(idx) for idx in frag_indices)
        smiles, canonical_atom_order = _component_smiles_and_atom_order(
            frag_mol,
            frag_indices,
        )
        original_atoms = sorted(
            idx
            for idx in frag_indices
            if idx < n_original_atoms and frag_mol.GetAtomWithIdx(idx).GetAtomicNum() != 0
        )
        dummy_isotopes = sorted(
            int(frag_mol.GetAtomWithIdx(idx).GetIsotope())
            for idx in frag_indices
            if frag_mol.GetAtomWithIdx(idx).GetAtomicNum() == 0
        )
        components.append(
            {
                "frag_atom_indices": list(frag_indices),
                "original_atoms": original_atoms,
                "dummy_isotopes": dummy_isotopes,
                "smiles": smiles,
                "canonical_atom_order": canonical_atom_order,
            }
        )
    return components


def _core_smiles_query_mapping(
    parent_mol: Chem.Mol,
    core_components: Sequence[dict],
    core_atoms: set[int],
    *,
    n_original_atoms: int,
) -> tuple[str, list[tuple[int, int]], list[tuple[int, int]]]:
    """Map shared-core query atom ids to original parent atom ids.

    RDKit writes each canonical fragment SMILES and records the exact atom
    output order. Query atom ids are the atom positions obtained by parsing
    the dot-joined canonical component SMILES. Unlike sorted parent indices,
    this label space is shared by every molecule in one ``core_smiles`` group.
    """

    ordered_components = sorted(
        core_components,
        key=lambda component: (
            str(component["smiles"]),
            tuple(int(idx) for idx in component["canonical_atom_order"]),
        ),
    )
    core_smiles = ".".join(
        str(component["smiles"]) for component in ordered_components
    )
    mapping: list[tuple[int, int]] = []
    atomic_numbers: list[tuple[int, int]] = []
    query_offset = 0
    for component in ordered_components:
        component_query = Chem.MolFromSmiles(str(component["smiles"]))
        if component_query is None:
            raise ValueError("Core component SMILES does not parse")
        output_order = [
            int(idx) for idx in component["canonical_atom_order"]
        ]
        if component_query.GetNumAtoms() != len(output_order):
            raise ValueError("Core component atom-order length mismatch")
        for local_query_idx, parent_atom_idx in enumerate(output_order):
            if parent_atom_idx < n_original_atoms:
                query_atom_idx = query_offset + local_query_idx
                mapping.append((query_atom_idx, parent_atom_idx))
                atomic_numbers.append(
                    (
                        query_atom_idx,
                        int(
                            parent_mol.GetAtomWithIdx(
                                parent_atom_idx
                            ).GetAtomicNum()
                        ),
                    )
                )
        query_offset += component_query.GetNumAtoms()
    if {parent for _query, parent in mapping} != set(core_atoms):
        raise ValueError(
            "Canonical core mapping does not cover the declared parent core"
        )
    query = Chem.MolFromSmiles(core_smiles)
    if query is None or query.GetNumAtoms() != query_offset:
        raise ValueError("Joined core SMILES does not preserve query atom labels")
    expected_query_atoms = {
        atom.GetIdx()
        for atom in query.GetAtoms()
        if atom.GetAtomicNum() != 0
    }
    if {query_idx for query_idx, _parent in mapping} != expected_query_atoms:
        raise ValueError(
            "Canonical core mapping does not cover every non-dummy query atom"
        )
    for query_idx, atomic_number in atomic_numbers:
        if query.GetAtomWithIdx(query_idx).GetAtomicNum() != atomic_number:
            raise ValueError("Core query/parent element mismatch")
    return core_smiles, mapping, atomic_numbers


def _attachment_atoms_for_variable(
    cut_combo: Sequence[tuple[int, int, int]],
    variable_atoms: set[int],
    core_atoms: set[int],
) -> tuple[list[int], list[int], list[int]]:
    """Locate changed-region atoms adjacent to the retained core."""

    core_attach: set[int] = set()
    variable_attach: set[int] = set()
    labels: list[int] = []
    for label_idx, (_bond_idx, atom_a, atom_b) in enumerate(cut_combo, start=1):
        a_in_var = int(atom_a) in variable_atoms
        b_in_var = int(atom_b) in variable_atoms
        if a_in_var == b_in_var:
            continue
        labels.append(int(label_idx))
        if a_in_var:
            variable_attach.add(int(atom_a))
            core_attach.add(int(atom_b))
        else:
            variable_attach.add(int(atom_b))
            core_attach.add(int(atom_a))
    return sorted(core_attach), sorted(variable_attach), labels


def fragment_entry(
    entry: dict,
    *,
    pattern: str,
    min_cuts: int,
    max_cuts: int,
    max_cut_bonds: int,
    min_core_atoms: int,
) -> tuple[list[dict], Counter]:
    """Generate RDKit-fragmentation MMP records for one molecule.

    For single-cut MMPs, the larger side is treated as the core and the smaller
    side as the variable substituent. For double-cut records, each component is
    considered as a possible variable fragment and retained as a separate class.
    """

    mol = entry["mol"]
    stats: Counter = Counter()
    cut_bond_rows = cuttable_bonds(mol, pattern=pattern, max_cut_bonds=max_cut_bonds)

    if not cut_bond_rows:
        stats["molecules_without_cuttable_bonds"] += 1
        return [], stats

    atom_count = int(entry["n_atoms"])
    fragments: list[dict] = []
    seen: set[tuple] = set()
    for cut_count in range(int(min_cuts), int(max_cuts) + 1):
        if cut_count > len(cut_bond_rows):
            continue
        for cut_combo in combinations(cut_bond_rows, cut_count):
            components = _fragment_components(mol, cut_combo)
            if len(components) < 2:
                stats["drop_fragment_less_than_two_components"] += 1
                continue

            if cut_count == 1:
                # Single-cut replacement MMPs are interpreted as one core plus
                # one variable substituent. The smaller side is the variable.
                variable_indices = [
                    min(
                        range(len(components)),
                        key=lambda idx: (
                            len(components[idx]["original_atoms"]),
                            components[idx]["smiles"],
                        ),
                    )
                ]
            else:
                variable_indices = list(range(len(components)))

            for variable_idx in variable_indices:
                variable_component = components[variable_idx]
                core_components = [comp for idx, comp in enumerate(components) if idx != variable_idx]
                variable_atoms = set(int(idx) for idx in variable_component["original_atoms"])
                core_atoms = set(
                    int(idx)
                    for comp in core_components
                    for idx in comp["original_atoms"]
                )
                if not variable_atoms or len(core_atoms) < int(min_core_atoms):
                    stats["drop_small_core_or_empty_variable"] += 1
                    continue

                core_attach, variable_attach, attachment_labels = _attachment_atoms_for_variable(
                    cut_combo,
                    variable_atoms=variable_atoms,
                    core_atoms=core_atoms,
                )
                if not core_attach:
                    stats["drop_no_core_attachment"] += 1
                    continue

                (
                    core_smiles,
                    core_query_atom_to_parent,
                    core_query_atomic_numbers,
                ) = _core_smiles_query_mapping(
                    mol,
                    core_components,
                    core_atoms,
                    n_original_atoms=atom_count,
                )
                variable_smiles = str(variable_component["smiles"])
                key = (
                    int(cut_count),
                    core_smiles,
                    variable_smiles,
                    tuple(sorted(variable_atoms)),
                )
                if key in seen:
                    stats["drop_duplicate_fragment_record"] += 1
                    continue
                seen.add(key)

                mmp_class = "single_cut" if int(cut_count) == 1 else "double_cut"
                fragments.append(
                    {
                        "fragment_id": f"{entry['row_idx']}::{cut_count}::{len(fragments)}",
                        "document_id": str(entry["document_id"]),
                        "molecule_id": str(entry["molecule_id"]),
                        "row_idx": int(entry["row_idx"]),
                        "dataset_index": entry["dataset_index"],
                        "graph_file": str(entry["graph_file"]),
                        "smiles": str(entry["smiles"]),
                        "canonical_isomeric_smiles": str(entry["canonical_isomeric_smiles"]),
                        "canonical_nonisomeric_smiles": str(entry["canonical_nonisomeric_smiles"]),
                        "pIC50": float(entry["pIC50"]),
                        "fingerprint": entry["fingerprint"],
                        "n_atoms": int(atom_count),
                        "cut_count": int(cut_count),
                        "mmp_class": mmp_class,
                        "core_smiles": core_smiles,
                        "variable_smiles": variable_smiles,
                        "core_atoms": sorted(core_atoms),
                        "core_query_atom_to_parent": [
                            [int(query_idx), int(parent_idx)]
                            for query_idx, parent_idx
                            in core_query_atom_to_parent
                        ],
                        "core_query_atomic_numbers": [
                            [int(query_idx), int(atomic_number)]
                            for query_idx, atomic_number
                            in core_query_atomic_numbers
                        ],
                        "core_mapping_method": CORE_MAPPING_METHOD,
                        "variable_atoms": sorted(variable_atoms),
                        "core_atom_count": int(len(core_atoms)),
                        "variable_atom_count": int(len(variable_atoms)),
                        "core_fraction": float(len(core_atoms) / max(1, atom_count)),
                        "attachment_core_atoms": core_attach,
                        "attachment_variable_atoms": variable_attach,
                        "attachment_labels": attachment_labels,
                        "cut_bond_indices": [int(row[0]) for row in cut_combo],
                        "cut_atom_pairs": [[int(row[1]), int(row[2])] for row in cut_combo],
                    }
                )
                stats[f"fragment_records_{mmp_class}"] += 1
    return fragments, stats


def _core_pairs_from_shared_query(
    frag_i: dict,
    frag_j: dict,
    *,
    core_smiles: str,
) -> list[tuple[int, int]]:
    """Pair parent atoms carrying the same canonical shared-core query id."""

    if (
        frag_i.get("core_mapping_method") != CORE_MAPPING_METHOD
        or frag_j.get("core_mapping_method") != CORE_MAPPING_METHOD
    ):
        raise ValueError("Fragment record has no strict shared-core mapping")
    mapping_i = {
        int(query_idx): int(parent_idx)
        for query_idx, parent_idx
        in frag_i["core_query_atom_to_parent"]
    }
    mapping_j = {
        int(query_idx): int(parent_idx)
        for query_idx, parent_idx
        in frag_j["core_query_atom_to_parent"]
    }
    if len(mapping_i) != len(frag_i["core_query_atom_to_parent"]):
        raise ValueError("Fragment i core query mapping is not one-to-one")
    if len(mapping_j) != len(frag_j["core_query_atom_to_parent"]):
        raise ValueError("Fragment j core query mapping is not one-to-one")
    if set(mapping_i) != set(mapping_j):
        raise ValueError("Shared-core query atom labels differ between molecules")
    if set(mapping_i.values()) != set(frag_i["core_atoms"]):
        raise ValueError("Fragment i core mapping does not cover its core atoms")
    if set(mapping_j.values()) != set(frag_j["core_atoms"]):
        raise ValueError("Fragment j core mapping does not cover its core atoms")

    numbers_i = {
        int(query_idx): int(atomic_number)
        for query_idx, atomic_number
        in frag_i["core_query_atomic_numbers"]
    }
    numbers_j = {
        int(query_idx): int(atomic_number)
        for query_idx, atomic_number
        in frag_j["core_query_atomic_numbers"]
    }
    query = Chem.MolFromSmiles(core_smiles)
    if query is None:
        raise ValueError("Shared core SMILES does not parse")
    expected_labels = {
        atom.GetIdx()
        for atom in query.GetAtoms()
        if atom.GetAtomicNum() != 0
    }
    if set(mapping_i) != expected_labels:
        raise ValueError("Shared-core mapping does not cover the core query")
    if set(numbers_i) != expected_labels or set(numbers_j) != expected_labels:
        raise ValueError("Shared-core element labels are incomplete")

    pairs: list[tuple[int, int]] = []
    for query_idx in sorted(expected_labels):
        query_atomic_number = int(
            query.GetAtomWithIdx(query_idx).GetAtomicNum()
        )
        if (
            numbers_i[query_idx] != query_atomic_number
            or numbers_j[query_idx] != query_atomic_number
        ):
            raise ValueError("Shared-core mapping pairs unequal elements")
        pairs.append((mapping_i[query_idx], mapping_j[query_idx]))
    return pairs


def build_mmp_pairs(
    entries: Sequence[dict],
    *,
    pattern: str,
    min_cuts: int,
    max_cuts: int,
    max_cut_bonds: int,
    min_core_atoms: int,
) -> tuple[list[dict], dict[str, dict], list[dict], Counter]:
    """Pair molecules sharing a fragment core within the same document group."""

    stats: Counter = Counter()
    fragment_records: list[dict] = []
    for entry in entries:
        rows, row_stats = fragment_entry(
            entry,
            pattern=pattern,
            min_cuts=min_cuts,
            max_cuts=max_cuts,
            max_cut_bonds=max_cut_bonds,
            min_core_atoms=min_core_atoms,
        )
        fragment_records.extend(rows)
        stats.update(row_stats)

    grouped: dict[tuple[str, int, str], list[dict]] = defaultdict(list)
    for rec in fragment_records:
        grouped[(str(rec["document_id"]), int(rec["cut_count"]), str(rec["core_smiles"]))].append(rec)

    pair_records: list[dict] = []
    pair_metrics: dict[str, dict] = {}
    seen_pair_keys: set[tuple] = set()
    for (document_id, cut_count, core_smiles), records in sorted(grouped.items()):
        records = sorted(records, key=lambda rec: (int(rec["row_idx"]), str(rec["variable_smiles"])))
        molecule_count = len({int(rec["row_idx"]) for rec in records})
        if molecule_count < 2:
            stats["drop_core_group_single_molecule"] += 1
            continue

        for frag_i, frag_j in combinations(records, 2):
            if int(frag_i["row_idx"]) == int(frag_j["row_idx"]):
                continue
            if int(frag_i["row_idx"]) > int(frag_j["row_idx"]):
                frag_i, frag_j = frag_j, frag_i

            uniqueness_key = (
                int(frag_i["row_idx"]),
                int(frag_j["row_idx"]),
                int(cut_count),
                str(core_smiles),
                str(frag_i["variable_smiles"]),
                str(frag_j["variable_smiles"]),
            )
            if uniqueness_key in seen_pair_keys:
                stats["drop_duplicate_pair_record"] += 1
                continue
            seen_pair_keys.add(uniqueness_key)

            fp_similarity = float(DataStructs.TanimotoSimilarity(frag_i["fingerprint"], frag_j["fingerprint"]))
            delta_pic50 = float(frag_i["pIC50"]) - float(frag_j["pIC50"])
            core_fraction_min = float(min(float(frag_i["core_fraction"]), float(frag_j["core_fraction"])))
            core_fraction_mean = float((float(frag_i["core_fraction"]) + float(frag_j["core_fraction"])) / 2.0)
            pair_key = build_fragment_pair_key(
                frag_i["row_idx"],
                frag_j["row_idx"],
                cut_count=int(cut_count),
                core_smiles=str(core_smiles),
                variable_i_smiles=str(frag_i["variable_smiles"]),
                variable_j_smiles=str(frag_j["variable_smiles"]),
            )
            pair_id = int(len(pair_records))
            mmp_class = "single_cut" if int(cut_count) == 1 else "double_cut"
            canonical_isomeric_smiles_i = str(frag_i["canonical_isomeric_smiles"])
            canonical_isomeric_smiles_j = str(frag_j["canonical_isomeric_smiles"])
            canonical_nonisomeric_smiles_i = str(frag_i["canonical_nonisomeric_smiles"])
            canonical_nonisomeric_smiles_j = str(frag_j["canonical_nonisomeric_smiles"])
            connectivity_match = bool(canonical_nonisomeric_smiles_i == canonical_nonisomeric_smiles_j)
            isomeric_smiles_match = bool(canonical_isomeric_smiles_i == canonical_isomeric_smiles_j)
            stereo_only_pair = bool(connectivity_match and not isomeric_smiles_match)
            core_pairs = _core_pairs_from_shared_query(
                frag_i,
                frag_j,
                core_smiles=str(core_smiles),
            )
            rec = {
                "pair_id": pair_id,
                "pair_key": pair_key,
                "document_id": str(document_id),
                "molecule_1_id": str(frag_i["molecule_id"]),
                "molecule_2_id": str(frag_j["molecule_id"]),
                "dataset_index_1": frag_i["dataset_index"],
                "dataset_index_2": frag_j["dataset_index"],
                "i_id": int(frag_i["row_idx"]),
                "j_id": int(frag_j["row_idx"]),
                "graph_i_file": str(frag_i["graph_file"]),
                "graph_j_file": str(frag_j["graph_file"]),
                "pIC50_i": float(frag_i["pIC50"]),
                "pIC50_j": float(frag_j["pIC50"]),
                "dy": float(delta_pic50),
                "abs_dy": float(abs(delta_pic50)),
                "fp_similarity": fp_similarity,
                "fp_distance": float(1.0 - fp_similarity),
                "structure_distance": float(1.0 - core_fraction_min),
                "passes_similarity_threshold": None,
                "passes_abs_delta_pic50_threshold": None,
                "is_selected": False,
                "fragmentation_method": FRAGMENTATION_METHOD,
                "rdmmpa_pattern": str(pattern),
                "cut_count": int(cut_count),
                "mmp_class": mmp_class,
                "is_primary_single_cut": bool(int(cut_count) == 1),
                "core_smiles": str(core_smiles),
                "variable_i_smiles": str(frag_i["variable_smiles"]),
                "variable_j_smiles": str(frag_j["variable_smiles"]),
                "mmp_transform": f"{frag_i['variable_smiles']}>>{frag_j['variable_smiles']}",
                "core_atom_count_i": int(frag_i["core_atom_count"]),
                "core_atom_count_j": int(frag_j["core_atom_count"]),
                "core_atom_count_min": int(min(int(frag_i["core_atom_count"]), int(frag_j["core_atom_count"]))),
                "variable_atom_count_i": int(frag_i["variable_atom_count"]),
                "variable_atom_count_j": int(frag_j["variable_atom_count"]),
                "diff_total_count": int(frag_i["variable_atom_count"]) + int(frag_j["variable_atom_count"]),
                "core_fraction_i": float(frag_i["core_fraction"]),
                "core_fraction_j": float(frag_j["core_fraction"]),
                "core_fraction_min": core_fraction_min,
                "core_fraction_mean": core_fraction_mean,
                "core_i": list(frag_i["core_atoms"]),
                "core_j": list(frag_j["core_atoms"]),
                "core_pairs": core_pairs,
                "core_mapping_method": CORE_MAPPING_METHOD,
                "core_correspondence_symmetric": True,
                "diff_i": list(frag_i["variable_atoms"]),
                "diff_j": list(frag_j["variable_atoms"]),
                "attachment_i": list(frag_i["attachment_core_atoms"]),
                "attachment_j": list(frag_j["attachment_core_atoms"]),
                "attachment_variable_i": list(frag_i["attachment_variable_atoms"]),
                "attachment_variable_j": list(frag_j["attachment_variable_atoms"]),
                "cut_bond_indices_i": list(frag_i["cut_bond_indices"]),
                "cut_bond_indices_j": list(frag_j["cut_bond_indices"]),
                "cut_atom_pairs_i": list(frag_i["cut_atom_pairs"]),
                "cut_atom_pairs_j": list(frag_j["cut_atom_pairs"]),
                "smiles_i": str(frag_i["smiles"]),
                "smiles_j": str(frag_j["smiles"]),
                "canonical_isomeric_smiles_i": canonical_isomeric_smiles_i,
                "canonical_isomeric_smiles_j": canonical_isomeric_smiles_j,
                "canonical_nonisomeric_smiles_i": canonical_nonisomeric_smiles_i,
                "canonical_nonisomeric_smiles_j": canonical_nonisomeric_smiles_j,
                "connectivity_match": connectivity_match,
                "isomeric_smiles_match": isomeric_smiles_match,
                "stereo_only_pair": stereo_only_pair,
            }
            pair_records.append(rec)
            stats[f"pair_records_{mmp_class}"] += 1
            if connectivity_match:
                stats["pair_records_same_connectivity"] += 1
            if stereo_only_pair:
                stats["pair_records_stereo_only"] += 1

            for left, right, sign in [
                (frag_i, frag_j, 1.0),
                (frag_j, frag_i, -1.0),
            ]:
                key = build_pair_metric_key(left["row_idx"], right["row_idx"])
                pair_metrics[key] = {
                    "pair_key": key,
                    "document_id": str(document_id),
                    "source_pair_id": int(pair_id),
                    "i_id": int(left["row_idx"]),
                    "j_id": int(right["row_idx"]),
                    "dy": float(sign * delta_pic50),
                    "abs_dy": float(abs(delta_pic50)),
                    "fp_similarity": fp_similarity,
                    "structure_distance": float(1.0 - core_fraction_min),
                    "core_fraction_min": core_fraction_min,
                    "cut_count": int(cut_count),
                    "mmp_class": mmp_class,
                    "core_smiles": str(core_smiles),
                }
    stats["fragment_records_total"] = int(len(fragment_records))
    stats["pair_records_total"] = int(len(pair_records))
    return pair_records, pair_metrics, fragment_records, stats


def pair_table_fieldnames(records: Sequence[dict]) -> list[str]:
    preferred = [
        "pair_id",
        "pair_key",
        "document_id",
        "molecule_1_id",
        "molecule_2_id",
        "dataset_index_1",
        "dataset_index_2",
        "i_id",
        "j_id",
        "graph_i_file",
        "graph_j_file",
        "pIC50_i",
        "pIC50_j",
        "dy",
        "abs_dy",
        "fp_similarity",
        "fp_distance",
        "structure_distance",
        "fragmentation_method",
        "cut_count",
        "mmp_class",
        "is_primary_single_cut",
        "core_smiles",
        "variable_i_smiles",
        "variable_j_smiles",
        "mmp_transform",
        "core_atom_count_i",
        "core_atom_count_j",
        "core_atom_count_min",
        "variable_atom_count_i",
        "variable_atom_count_j",
        "diff_total_count",
        "core_fraction_i",
        "core_fraction_j",
        "core_fraction_min",
        "core_fraction_mean",
        "passes_similarity_threshold",
        "passes_abs_delta_pic50_threshold",
        "is_selected",
        "core_i_json",
        "core_j_json",
        "core_pairs_json",
        "core_pair_candidates_json",
        "diff_i_json",
        "diff_j_json",
        "attachment_i_json",
        "attachment_j_json",
        "attachment_variable_i_json",
        "attachment_variable_j_json",
        "cut_bond_indices_i_json",
        "cut_bond_indices_j_json",
        "cut_atom_pairs_i_json",
        "cut_atom_pairs_j_json",
        "smiles_i",
        "smiles_j",
        "canonical_isomeric_smiles_i",
        "canonical_isomeric_smiles_j",
        "canonical_nonisomeric_smiles_i",
        "canonical_nonisomeric_smiles_j",
        "connectivity_match",
        "isomeric_smiles_match",
        "stereo_only_pair",
    ]
    existing = set().union(*(rec.keys() for rec in records)) if records else set()
    extras = sorted(existing - {name.removesuffix("_json") for name in preferred} - set(preferred))
    return preferred + extras


def _csv_value(rec: dict, fieldname: str) -> object:
    if fieldname.endswith("_json"):
        key = fieldname[: -len("_json")]
        return json.dumps(rec.get(key, []), ensure_ascii=False)
    value = rec.get(fieldname, "")
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def save_pair_table_csv(path: Path, records: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = pair_table_fieldnames(records)
    with path.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow({name: _csv_value(rec, name) for name in fieldnames})


@dataclass(frozen=True)
class MMPSubsetConfig:
    """Chemical selection thresholds for the new-endpoint MMP population."""

    min_fingerprint_similarity: float = 0.1
    min_abs_delta_pic50: float = 0.1
    primary_mmp_class: str = "single_cut"
    exclude_same_connectivity_pairs: bool = True
    include_stereo_only_pairs: bool = True
    min_double_cut_core_fraction: float = 0.3
    max_double_cut_diff_total_count: int = 60
    require_double_cut_two_attachments: bool = True
    collapse_fragmentations_by_molecule_pair: bool = True
    max_excluded_examples_per_reason: int = 12


def parse_mmp_classes(value: str) -> tuple[str, ...]:
    """Parse a comma-separated MMP class list while preserving legacy single values."""
    classes = tuple(part.strip() for part in str(value).split(",") if part.strip())
    if not classes:
        raise ValueError("primary_mmp_class must contain at least one MMP class.")
    return classes


@lru_cache(maxsize=None)
def _canonicalize_smiles(smiles: str, isomeric: bool) -> str:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=bool(isomeric))


def _pair_context(record: dict, position: int) -> str:
    return (
        f"pair_records[{position}] "
        f"(pair_id={record.get('pair_id', '<missing>')!s}, "
        f"pair_key={record.get('pair_key', '<missing>')!s})"
    )


def _validate_record_smiles(record: dict, *, context: str) -> None:
    for side in ("i", "j"):
        source_field = f"smiles_{side}"
        source_smiles = str(record.get(source_field, "") or "").strip()
        if not source_smiles:
            raise ValueError(f"{context} has invalid or missing {source_field}")
        try:
            expected = {
                "isomeric": _canonicalize_smiles(source_smiles, True),
                "nonisomeric": _canonicalize_smiles(source_smiles, False),
            }
        except ValueError as exc:
            raise ValueError(
                f"{context} has invalid or missing {source_field}"
            ) from exc
        for canonical_kind in ("isomeric", "nonisomeric"):
            field = f"canonical_{canonical_kind}_smiles_{side}"
            existing = str(record.get(field, "") or "").strip()
            if existing:
                try:
                    _canonicalize_smiles(
                        existing,
                        canonical_kind == "isomeric",
                    )
                except ValueError as exc:
                    raise ValueError(f"{context} has invalid {field}") from exc
                if existing != expected[canonical_kind]:
                    raise ValueError(
                        f"{context} {field} does not exactly match its "
                        f"source {source_field}"
                    )


def _canonical_smiles_from_record(record: dict, side: str, *, isomeric: bool) -> str:
    field = f"canonical_{'isomeric' if isomeric else 'nonisomeric'}_smiles_{side}"
    existing = str(record.get(field, "") or "").strip()
    if existing:
        return existing
    smiles = str(record.get(f"smiles_{side}", "") or "").strip()
    if not smiles:
        return ""
    return _canonicalize_smiles(smiles, bool(isomeric))


def annotate_connectivity(record: dict) -> dict:
    updated = dict(record)
    canonical_isomeric_smiles_i = _canonical_smiles_from_record(updated, "i", isomeric=True)
    canonical_isomeric_smiles_j = _canonical_smiles_from_record(updated, "j", isomeric=True)
    canonical_nonisomeric_smiles_i = _canonical_smiles_from_record(updated, "i", isomeric=False)
    canonical_nonisomeric_smiles_j = _canonical_smiles_from_record(updated, "j", isomeric=False)
    connectivity_match = bool(
        canonical_nonisomeric_smiles_i
        and canonical_nonisomeric_smiles_j
        and canonical_nonisomeric_smiles_i == canonical_nonisomeric_smiles_j
    )
    isomeric_smiles_match = bool(
        canonical_isomeric_smiles_i
        and canonical_isomeric_smiles_j
        and canonical_isomeric_smiles_i == canonical_isomeric_smiles_j
    )
    updated.update(
        {
            "canonical_isomeric_smiles_i": canonical_isomeric_smiles_i,
            "canonical_isomeric_smiles_j": canonical_isomeric_smiles_j,
            "canonical_nonisomeric_smiles_i": canonical_nonisomeric_smiles_i,
            "canonical_nonisomeric_smiles_j": canonical_nonisomeric_smiles_j,
            "connectivity_match": connectivity_match,
            "isomeric_smiles_match": isomeric_smiles_match,
            "stereo_only_pair": bool(connectivity_match and not isomeric_smiles_match),
        }
    )
    return updated


def _required_sequence_length(
    record: dict,
    field: str,
    *,
    context: str,
) -> int:
    if field not in record or record[field] is None:
        raise ValueError(f"{context} is missing required {field}")
    value = record[field]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{context} has invalid JSON in {field}"
            ) from exc
        if not isinstance(parsed, list):
            raise ValueError(f"{context} {field} JSON must encode a list")
        return len(parsed)
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{context} {field} must be a list or JSON list")
    return len(value)


def _required_finite_metric(
    record: dict,
    field: str,
    *,
    context: str,
) -> float:
    if field not in record or record[field] is None:
        raise ValueError(f"{context} is missing required {field}")
    if isinstance(record[field], bool):
        raise ValueError(f"{context} {field} must be finite")
    try:
        value = float(record[field])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} {field} must be finite") from exc
    if not math.isfinite(value):
        raise ValueError(f"{context} {field} must be finite")
    return value


def _required_integer_metric(
    record: dict,
    field: str,
    *,
    context: str,
) -> int:
    if field not in record or record[field] is None:
        raise ValueError(f"{context} is missing required {field}")
    return exact_integer(
        record[field],
        name=f"{context} {field}",
    )


def molecule_pair_key(record: dict) -> str:
    smiles_i = str(record.get("canonical_isomeric_smiles_i", "") or "").strip()
    smiles_j = str(record.get("canonical_isomeric_smiles_j", "") or "").strip()
    if smiles_i and smiles_j:
        lo, hi = sorted((smiles_i, smiles_j))
        return f"canonical_isomeric_smiles::{lo}::{hi}"

    mol_i = str(record.get("molecule_1_id", record.get("i_id", "")) or "").strip()
    mol_j = str(record.get("molecule_2_id", record.get("j_id", "")) or "").strip()
    lo, hi = sorted((mol_i, mol_j))
    return f"{record.get('document_id', '')}::{lo}::{hi}"


def representative_sort_key(record: dict) -> tuple:
    return (
        int(record["diff_total_count"]),
        -float(record["core_fraction_min"]),
        -float(record["fp_similarity"]),
        -float(record["abs_dy"]),
        int(record["cut_count"]),
        str(record.get("pair_key", "")),
    )


def select_pair_records(pair_records: Sequence[dict], cfg: MMPSubsetConfig) -> tuple[list[dict], list[dict], Counter]:
    """Apply the fixed chemical, similarity and target-difference selection rules."""

    selected: list[dict] = []
    annotated: list[dict] = []
    stats: Counter = Counter()
    allowed_mmp_classes = set(parse_mmp_classes(cfg.primary_mmp_class))

    for position, rec in enumerate(pair_records):
        if not isinstance(rec, dict):
            raise TypeError(f"pair_records[{position}] must be a dictionary")
        context = _pair_context(rec, position)
        _validate_record_smiles(rec, context=context)
        updated = annotate_connectivity(rec)
        if "cut_count" not in updated or updated["cut_count"] is None:
            raise ValueError(f"{context} is missing required cut_count")
        cut_count = exact_integer(
            updated["cut_count"],
            name=f"{context} cut_count",
        )
        fp_similarity = _required_finite_metric(
            updated,
            "fp_similarity",
            context=context,
        )
        abs_dy = _required_finite_metric(
            updated,
            "abs_dy",
            context=context,
        )
        core_fraction_min = _required_finite_metric(
            updated,
            "core_fraction_min",
            context=context,
        )
        diff_total_count = _required_integer_metric(
            updated,
            "diff_total_count",
            context=context,
        )
        attachment_i_count = _required_sequence_length(
            updated,
            "attachment_i",
            context=context,
        )
        attachment_j_count = _required_sequence_length(
            updated,
            "attachment_j",
            context=context,
        )
        attachment_variable_i_count = _required_sequence_length(
            updated,
            "attachment_variable_i",
            context=context,
        )
        attachment_variable_j_count = _required_sequence_length(
            updated,
            "attachment_variable_j",
            context=context,
        )

        passes_primary_class = str(updated.get("mmp_class", "")) in allowed_mmp_classes
        passes_similarity = fp_similarity >= float(cfg.min_fingerprint_similarity)
        passes_activity = abs_dy >= float(cfg.min_abs_delta_pic50)
        connectivity_match = bool(updated.get("connectivity_match", False))
        stereo_only_pair = bool(updated.get("stereo_only_pair", False))
        excluded_by_connectivity = bool(
            cfg.exclude_same_connectivity_pairs
            and connectivity_match
            and not (bool(cfg.include_stereo_only_pairs) and stereo_only_pair)
        )
        passes_connectivity = not excluded_by_connectivity
        passes_stereo_only_inclusion = bool(stereo_only_pair and passes_connectivity)
        passes_double_cut_two_attachment = (
            cut_count != 2
            or not bool(cfg.require_double_cut_two_attachments)
            or (
                attachment_i_count == 2
                and attachment_j_count == 2
                and attachment_variable_i_count == 2
                and attachment_variable_j_count == 2
            )
        )
        passes_double_cut_core_fraction = (
            cut_count != 2
            or core_fraction_min > float(cfg.min_double_cut_core_fraction)
        )
        passes_double_cut_diff_total_count = (
            cut_count != 2
            or diff_total_count <= int(cfg.max_double_cut_diff_total_count)
        )
        passes_double_cut_quality = bool(
            passes_double_cut_two_attachment
            and passes_double_cut_core_fraction
            and passes_double_cut_diff_total_count
        )
        preliminary_training = bool(
            passes_primary_class
            and passes_similarity
            and passes_activity
            and passes_connectivity
            and passes_double_cut_quality
        )

        updated["molecule_pair_key"] = molecule_pair_key(updated)
        updated["double_cut_attachment_i_count"] = int(attachment_i_count)
        updated["double_cut_attachment_j_count"] = int(attachment_j_count)
        updated["double_cut_attachment_variable_i_count"] = int(attachment_variable_i_count)
        updated["double_cut_attachment_variable_j_count"] = int(attachment_variable_j_count)
        updated["passes_similarity_threshold"] = bool(passes_similarity)
        updated["passes_abs_delta_pic50_threshold"] = bool(passes_activity)
        updated["passes_connectivity_change"] = bool(passes_connectivity)
        updated["passes_stereo_only_inclusion"] = bool(passes_stereo_only_inclusion)
        updated["passes_double_cut_two_attachment"] = bool(passes_double_cut_two_attachment)
        updated["passes_double_cut_core_fraction"] = bool(passes_double_cut_core_fraction)
        updated["passes_double_cut_diff_total_count"] = bool(passes_double_cut_diff_total_count)
        updated["passes_double_cut_quality"] = bool(passes_double_cut_quality)
        updated["_passes_primary_class"] = bool(passes_primary_class)
        updated["_preliminary_training"] = bool(preliminary_training)
        annotated.append(updated)

    grouped_selected: dict[str, list[dict]] = defaultdict(list)
    for rec in annotated:
        if bool(rec.get("_preliminary_training", False)):
            grouped_selected[str(rec["molecule_pair_key"])].append(rec)
    selected_molecule_pair_keys: set[str] = set()
    if bool(cfg.collapse_fragmentations_by_molecule_pair):
        for records in grouped_selected.values():
            best = sorted(records, key=representative_sort_key)[0]
            selected_molecule_pair_keys.add(str(best["pair_key"]))
    else:
        for records in grouped_selected.values():
            selected_molecule_pair_keys.update(str(rec["pair_key"]) for rec in records)

    for updated in annotated:
        passes_primary_class = bool(updated.pop("_passes_primary_class"))
        preliminary_training = bool(updated.pop("_preliminary_training"))
        passes_molecule_pair_representative = True
        if bool(cfg.collapse_fragmentations_by_molecule_pair) and preliminary_training:
            passes_molecule_pair_representative = str(updated.get("pair_key")) in selected_molecule_pair_keys

        exclusion_reasons: list[str] = []
        if not passes_primary_class:
            exclusion_reasons.append("non_primary_mmp_class")
            stats["drop_non_primary_mmp_class"] += 1
        if not bool(updated["passes_connectivity_change"]):
            exclusion_reasons.append("same_connectivity_pair")
            stats["drop_same_connectivity_pair"] += 1
            if bool(updated.get("stereo_only_pair", False)):
                stats["drop_stereo_only_pair"] += 1
        if not bool(updated["passes_similarity_threshold"]):
            exclusion_reasons.append("similarity_threshold")
            stats["drop_similarity_threshold"] += 1
        if not bool(updated["passes_abs_delta_pic50_threshold"]):
            exclusion_reasons.append("abs_delta_pic50_threshold")
            stats["drop_abs_delta_pic50_threshold"] += 1
        if not bool(updated["passes_double_cut_two_attachment"]):
            exclusion_reasons.append("double_cut_two_attachment")
            stats["drop_double_cut_two_attachment"] += 1
        if not bool(updated["passes_double_cut_core_fraction"]):
            exclusion_reasons.append("double_cut_core_fraction")
            stats["drop_double_cut_core_fraction"] += 1
        if not bool(updated["passes_double_cut_diff_total_count"]):
            exclusion_reasons.append("double_cut_diff_total_count")
            stats["drop_double_cut_diff_total_count"] += 1
        if not passes_molecule_pair_representative:
            exclusion_reasons.append("molecule_pair_non_min_diff_representative")
            stats["drop_molecule_pair_non_min_diff_representative"] += 1

        is_training = bool(preliminary_training and passes_molecule_pair_representative)
        updated["passes_molecule_pair_representative"] = bool(passes_molecule_pair_representative)
        updated["is_training"] = bool(is_training)
        updated["exclusion_reason"] = ";".join(exclusion_reasons)
        if is_training:
            stats["selected"] += 1
            if bool(updated.get("stereo_only_pair", False)):
                stats["selected_stereo_only_pair"] += 1
            selected.append(updated)
        else:
            stats["excluded"] += 1

    stats["source_total"] = int(len(pair_records))
    stats["eligible_pre_representative"] = int(sum(len(records) for records in grouped_selected.values()))
    stats["eligible_molecule_pair_groups"] = int(len(grouped_selected))
    stats["eligible_duplicate_rows_by_molecule_pair"] = int(
        sum(max(len(records) - 1, 0) for records in grouped_selected.values())
    )
    stats["selected_molecule_pair_representatives"] = int(len(selected_molecule_pair_keys))
    return selected, annotated, stats


SUBSTRUCTURE_MATCH_LIMIT = 10_001


def _json_int_list(value: object, *, field: str) -> list[int]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{field} must be a list")
    return [int(item) for item in value]


def _cut_combo(record: dict, side: str, mol: Chem.Mol) -> list[tuple[int, int, int]]:
    bond_indices = _json_int_list(
        record[f"cut_bond_indices_{side}"],
        field=f"cut_bond_indices_{side}",
    )
    atom_pairs = record[f"cut_atom_pairs_{side}"]
    if not isinstance(atom_pairs, (list, tuple)):
        raise TypeError(f"cut_atom_pairs_{side} must be a list")
    if len(bond_indices) != len(atom_pairs):
        raise ValueError(f"cut declarations differ in length for side {side}")

    combo: list[tuple[int, int, int]] = []
    for bond_idx, raw_pair in zip(bond_indices, atom_pairs):
        if not isinstance(raw_pair, (list, tuple)) or len(raw_pair) != 2:
            raise ValueError(f"invalid cut atom pair for side {side}: {raw_pair}")
        atom_a, atom_b = (int(raw_pair[0]), int(raw_pair[1]))
        bond = mol.GetBondWithIdx(int(bond_idx))
        actual_pair = {
            int(bond.GetBeginAtomIdx()),
            int(bond.GetEndAtomIdx()),
        }
        if actual_pair != {atom_a, atom_b}:
            raise ValueError(
                f"cut bond/atom mismatch for side {side}: "
                f"bond {bond_idx}, declared {(atom_a, atom_b)}"
            )
        combo.append((int(bond_idx), atom_a, atom_b))
    if not combo:
        raise ValueError(f"empty cut declaration for side {side}")
    return combo


def _mapping_candidate_count(
    *,
    frag_mol: Chem.Mol,
    core_smiles: str,
    core_atoms: set[int],
    n_original_atoms: int,
) -> int:
    query = Chem.MolFromSmiles(core_smiles)
    if query is None:
        raise ValueError("core_smiles does not parse")
    query_labels = [
        int(atom.GetIdx())
        for atom in query.GetAtoms()
        if atom.GetAtomicNum() != 0
    ]
    raw_matches = frag_mol.GetSubstructMatches(
        query,
        uniquify=False,
        useChirality=True,
        maxMatches=SUBSTRUCTURE_MATCH_LIMIT,
    )
    if len(raw_matches) >= SUBSTRUCTURE_MATCH_LIMIT:
        raise ValueError("core correspondence exceeded the explicit match limit")

    candidates: set[tuple[int, ...]] = set()
    for match in raw_matches:
        parent_assignment = tuple(int(match[label]) for label in query_labels)
        if any(idx >= n_original_atoms for idx in parent_assignment):
            continue
        if set(parent_assignment) == core_atoms:
            candidates.add(parent_assignment)
    if not candidates:
        raise ValueError("no exact shared-core substructure correspondence found")
    return len(candidates)


def _core_pair_candidates_from_automorphisms(
    frag_i: dict,
    frag_j: dict,
    *,
    core_smiles: str,
) -> list[list[list[int]]]:
    """Enumerate exact atom mappings induced by shared-core automorphisms."""

    mapping_i = {
        int(query_idx): int(parent_idx)
        for query_idx, parent_idx in frag_i["core_query_atom_to_parent"]
    }
    mapping_j = {
        int(query_idx): int(parent_idx)
        for query_idx, parent_idx in frag_j["core_query_atom_to_parent"]
    }
    query = Chem.MolFromSmiles(core_smiles)
    if query is None:
        raise ValueError("core_smiles does not parse")
    core_query_indices = tuple(
        int(atom.GetIdx())
        for atom in query.GetAtoms()
        if atom.GetAtomicNum() != 0
    )
    if set(mapping_i) != set(core_query_indices):
        raise ValueError("fragment i does not cover every core query atom")
    if set(mapping_j) != set(core_query_indices):
        raise ValueError("fragment j does not cover every core query atom")

    matches = query.GetSubstructMatches(
        query,
        uniquify=False,
        useChirality=True,
        maxMatches=SUBSTRUCTURE_MATCH_LIMIT,
    )
    if len(matches) >= SUBSTRUCTURE_MATCH_LIMIT:
        raise ValueError("core automorphisms exceeded the explicit match limit")

    candidate_set: set[tuple[tuple[int, int], ...]] = set()
    for match in matches:
        candidate = tuple(
            (mapping_i[query_idx], mapping_j[int(match[query_idx])])
            for query_idx in core_query_indices
        )
        left = tuple(atom_i for atom_i, _atom_j in candidate)
        right = tuple(atom_j for _atom_i, atom_j in candidate)
        if len(set(left)) != len(core_query_indices):
            raise ValueError("core automorphism candidate is not injective on i")
        if len(set(right)) != len(core_query_indices):
            raise ValueError("core automorphism candidate is not injective on j")
        if set(left) != set(mapping_i.values()):
            raise ValueError("core automorphism candidate changes i core membership")
        if set(right) != set(mapping_j.values()):
            raise ValueError("core automorphism candidate changes j core membership")
        candidate_set.add(candidate)
    if not candidate_set:
        raise ValueError("shared core has no valid automorphism candidate")

    candidates = sorted(candidate_set)
    numbers_i_by_parent = {
        mapping_i[int(query_idx)]: int(number)
        for query_idx, number in frag_i["core_query_atomic_numbers"]
    }
    numbers_j_by_parent = {
        mapping_j[int(query_idx)]: int(number)
        for query_idx, number in frag_j["core_query_atomic_numbers"]
    }
    if set(numbers_i_by_parent) != set(mapping_i.values()):
        raise ValueError("fragment i core element metadata is incomplete")
    if set(numbers_j_by_parent) != set(mapping_j.values()):
        raise ValueError("fragment j core element metadata is incomplete")
    for candidate in candidates:
        for atom_i, atom_j in candidate:
            if numbers_i_by_parent[atom_i] != numbers_j_by_parent[atom_j]:
                raise ValueError(
                    "core automorphism candidate maps unequal elements"
                )
    return [
        [[int(atom_i), int(atom_j)] for atom_i, atom_j in candidate]
        for candidate in candidates
    ]


def _reconstruct_side(
    record: dict,
    *,
    side: str,
    dataset_row: dict[str, str],
    graph_meta_row: dict,
) -> tuple[dict, int]:
    """Recover core and changed-region atoms for one molecular endpoint."""

    index_key = "i_id" if side == "i" else "j_id"
    smiles_key = f"smiles_{side}"
    core_key = f"core_{side}"
    diff_key = f"diff_{side}"
    molecule_key = "molecule_1_id" if side == "i" else "molecule_2_id"
    graph_key = f"graph_{side}_file"
    value_key = "pIC50_i" if side == "i" else "pIC50_j"

    source_index = int(record[index_key])
    dataset_smiles = str(dataset_row["Smiles"]).strip()
    if str(record[smiles_key]) != dataset_smiles:
        raise ValueError(
            f"pair {record['pair_id']} side {side}: source SMILES mismatch"
        )
    dataset_molecule_id = str(
        dataset_row.get("cid") or dataset_row.get("number") or ""
    ).strip()
    if str(record[molecule_key]) != dataset_molecule_id:
        raise ValueError(
            f"pair {record['pair_id']} side {side}: molecule id mismatch"
        )
    graph_file = str(
        graph_meta_row.get("file")
        or graph_meta_row.get("filename")
        or ""
    ).strip()
    if str(record[graph_key]) != graph_file:
        raise ValueError(
            f"pair {record['pair_id']} side {side}: graph filename mismatch"
        )
    if abs(float(record[value_key]) - float(dataset_row["pIC50"])) > 1e-12:
        raise ValueError(
            f"pair {record['pair_id']} side {side}: pIC50 mismatch"
        )

    mol = Chem.MolFromSmiles(dataset_smiles)
    if mol is None:
        raise ValueError(f"dataset row {source_index} has invalid SMILES")
    n_atoms = int(mol.GetNumAtoms())
    core_atoms = set(
        _json_int_list(record[core_key], field=core_key)
    )
    diff_atoms = set(
        _json_int_list(record[diff_key], field=diff_key)
    )
    all_atoms = set(range(n_atoms))
    if core_atoms & diff_atoms or core_atoms | diff_atoms != all_atoms:
        raise ValueError(
            f"pair {record['pair_id']} side {side}: core/diff is not a partition"
        )

    combo = _cut_combo(record, side, mol)
    components = _fragment_components(mol, combo)
    variable_components = [
        component
        for component in components
        if set(component["original_atoms"]) == diff_atoms
    ]
    if len(variable_components) != 1:
        raise ValueError(
            f"pair {record['pair_id']} side {side}: "
            f"expected one declared variable component, got {len(variable_components)}"
        )
    variable = variable_components[0]
    core_components = [
        component for component in components if component is not variable
    ]
    generated_core_smiles, query_to_parent, query_atomic_numbers = (
        _core_smiles_query_mapping(
            mol,
            core_components,
            core_atoms,
            n_original_atoms=n_atoms,
        )
    )
    if generated_core_smiles != str(record["core_smiles"]):
        raise ValueError(
            f"pair {record['pair_id']} side {side}: reconstructed core SMILES differs"
        )

    cut_bonds = [int(row[0]) for row in combo]
    dummy_labels = [(idx + 1, idx + 1) for idx in range(len(combo))]
    frag_mol = Chem.FragmentOnBonds(
        mol,
        cut_bonds,
        addDummies=True,
        dummyLabels=dummy_labels,
    )
    candidate_count = _mapping_candidate_count(
        frag_mol=frag_mol,
        core_smiles=generated_core_smiles,
        core_atoms=core_atoms,
        n_original_atoms=n_atoms,
    )
    fragment_record = {
        "core_atoms": sorted(core_atoms),
        "core_query_atom_to_parent": [
            [int(query_idx), int(parent_idx)]
            for query_idx, parent_idx in query_to_parent
        ],
        "core_query_atomic_numbers": [
            [int(query_idx), int(atomic_number)]
            for query_idx, atomic_number in query_atomic_numbers
        ],
        "core_mapping_method": CORE_MAPPING_METHOD,
    }
    return fragment_record, candidate_count


def _correct_records(
    pair_records: Sequence[dict],
    *,
    dataset_rows: Sequence[dict[str, str]],
    graph_meta: Sequence[dict],
) -> tuple[list[dict], dict[str, int]]:
    """Correct core correspondences and retain valid automorphism candidates."""

    corrected: list[dict] = []
    audit = {
        "pair_count": 0,
        "original_pairs_with_element_mismatch": 0,
        "original_mapped_atoms_with_element_mismatch": 0,
        "corrected_pairs_with_element_mismatch": 0,
        "corrected_mapped_atoms_with_element_mismatch": 0,
        "unique_correspondence_pairs": 0,
        "symmetry_ambiguous_pairs": 0,
        "max_mapping_candidate_count": 0,
        "max_core_pair_automorphism_candidate_count": 0,
        "excluded_pairs": 0,
    }
    for source_record in pair_records:
        record = dict(source_record)
        i_id, j_id = int(record["i_id"]), int(record["j_id"])
        if not (0 <= i_id < len(dataset_rows) and 0 <= j_id < len(dataset_rows)):
            raise IndexError(f"pair {record['pair_id']} endpoint index is out of range")
        if not (i_id < len(graph_meta) and j_id < len(graph_meta)):
            raise IndexError(f"pair {record['pair_id']} graph metadata is out of range")
        expected_dy = float(dataset_rows[i_id]["pIC50"]) - float(
            dataset_rows[j_id]["pIC50"]
        )
        if abs(float(record["dy"]) - expected_dy) > 1e-12:
            raise ValueError(f"pair {record['pair_id']} dy differs from source labels")

        mol_i = Chem.MolFromSmiles(str(dataset_rows[i_id]["Smiles"]).strip())
        mol_j = Chem.MolFromSmiles(str(dataset_rows[j_id]["Smiles"]).strip())
        if mol_i is None or mol_j is None:
            raise ValueError(f"pair {record['pair_id']} endpoint SMILES does not parse")

        original_mismatches = sum(
            int(
                mol_i.GetAtomWithIdx(int(atom_i)).GetAtomicNum()
                != mol_j.GetAtomWithIdx(int(atom_j)).GetAtomicNum()
            )
            for atom_i, atom_j in record["core_pairs"]
        )
        audit["original_mapped_atoms_with_element_mismatch"] += original_mismatches
        if original_mismatches:
            audit["original_pairs_with_element_mismatch"] += 1

        frag_i, candidate_i = _reconstruct_side(
            record,
            side="i",
            dataset_row=dataset_rows[i_id],
            graph_meta_row=graph_meta[i_id],
        )
        frag_j, candidate_j = _reconstruct_side(
            record,
            side="j",
            dataset_row=dataset_rows[j_id],
            graph_meta_row=graph_meta[j_id],
        )
        core_pairs = _core_pairs_from_shared_query(
            frag_i,
            frag_j,
            core_smiles=str(record["core_smiles"]),
        )
        reverse_pairs = _core_pairs_from_shared_query(
            frag_j,
            frag_i,
            core_smiles=str(record["core_smiles"]),
        )
        if reverse_pairs != [(right, left) for left, right in core_pairs]:
            raise ValueError(f"pair {record['pair_id']} is not reversal-symmetric")
        core_pair_candidates = _core_pair_candidates_from_automorphisms(
            frag_i,
            frag_j,
            core_smiles=str(record["core_smiles"]),
        )
        reverse_candidates = _core_pair_candidates_from_automorphisms(
            frag_j,
            frag_i,
            core_smiles=str(record["core_smiles"]),
        )
        expected_reverse_candidates = {
            tuple(
                sorted(
                    (int(atom_j), int(atom_i))
                    for atom_i, atom_j in candidate
                )
            )
            for candidate in core_pair_candidates
        }
        observed_reverse_candidates = {
            tuple(
                sorted(
                    (int(atom_i), int(atom_j))
                    for atom_i, atom_j in candidate
                )
            )
            for candidate in reverse_candidates
        }
        if observed_reverse_candidates != expected_reverse_candidates:
            raise ValueError(
                f"pair {record['pair_id']} automorphism candidates are not "
                "reversal-symmetric"
            )
        if [
            [int(atom_i), int(atom_j)]
            for atom_i, atom_j in core_pairs
        ] not in core_pair_candidates:
            raise ValueError(
                f"pair {record['pair_id']} canonical mapping is absent from "
                "its automorphism candidates"
            )

        corrected_mismatches = sum(
            int(
                mol_i.GetAtomWithIdx(atom_i).GetAtomicNum()
                != mol_j.GetAtomWithIdx(atom_j).GetAtomicNum()
            )
            for atom_i, atom_j in core_pairs
        )
        audit["corrected_mapped_atoms_with_element_mismatch"] += corrected_mismatches
        if corrected_mismatches:
            audit["corrected_pairs_with_element_mismatch"] += 1

        unique = len(core_pair_candidates) == 1
        audit[
            "unique_correspondence_pairs"
            if unique
            else "symmetry_ambiguous_pairs"
        ] += 1
        audit["max_mapping_candidate_count"] = max(
            audit["max_mapping_candidate_count"],
            candidate_i,
            candidate_j,
        )
        audit["max_core_pair_automorphism_candidate_count"] = max(
            audit["max_core_pair_automorphism_candidate_count"],
            len(core_pair_candidates),
        )
        record.update(
            {
                "core_pairs": [
                    [int(atom_i), int(atom_j)]
                    for atom_i, atom_j in core_pairs
                ],
                "core_mapping_method": CORE_MAPPING_METHOD,
                "core_mapping_candidate_count_i": int(candidate_i),
                "core_mapping_candidate_count_j": int(candidate_j),
                "core_pair_candidates": core_pair_candidates,
                "core_pair_automorphism_candidate_count": len(
                    core_pair_candidates
                ),
                "core_correspondence_unique": bool(unique),
                "core_correspondence_symmetric": True,
            }
        )
        corrected.append(record)
        audit["pair_count"] += 1

    if audit["corrected_pairs_with_element_mismatch"] != 0:
        raise ValueError("corrected output still contains unequal core elements")
    return corrected, audit
