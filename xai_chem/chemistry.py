"""Chemistry for notebook-based atom-contribution experiments."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Protocol
import numpy as np
from numpy.typing import NDArray
from typing import Any
from rdkit import Chem


ATOM_MAPPING_VERSION = (
    "rdkit_heavy_atom_explicit_smiles_to_graph_permutation_v2"
)


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _freeze_metadata(value: Any) -> Any:
    """Defensively copy JSON-like provenance into immutable containers."""

    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("metadata mapping keys must be strings")
        return MappingProxyType(
            {key: _freeze_metadata(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_metadata(item) for item in value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        normalized = float(value)
        if not np.isfinite(normalized):
            raise ValueError("metadata floating-point values must be finite")
        return normalized
    raise TypeError(
        "metadata values must be JSON-compatible immutable provenance"
    )


@dataclass(frozen=True)
class InputProvenance:
    """Hashes for the exact dataset and ordered graph-file manifest."""

    dataset_id: str
    dataset_sha256: str
    graph_manifest_sha256: str
    graph_count: int
    atom_mapping_version: str = ATOM_MAPPING_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.dataset_id, str) or not self.dataset_id:
            raise ValueError("dataset_id must be a non-empty string")
        if self.dataset_id != self.dataset_id.strip():
            raise ValueError("dataset_id must not have surrounding whitespace")
        for name in ("dataset_sha256", "graph_manifest_sha256"):
            value = getattr(self, name)
            if not isinstance(value, str) or not _SHA256.fullmatch(value):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if (
            isinstance(self.graph_count, bool)
            or not isinstance(self.graph_count, (int, np.integer))
            or int(self.graph_count) < 1
        ):
            raise ValueError("graph_count must be a positive integer")
        if self.atom_mapping_version != ATOM_MAPPING_VERSION:
            raise ValueError(
                f"atom_mapping_version must be {ATOM_MAPPING_VERSION!r}"
            )
        object.__setattr__(self, "graph_count", int(self.graph_count))

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "dataset_sha256": self.dataset_sha256,
            "graph_manifest_sha256": self.graph_manifest_sha256,
            "graph_count": self.graph_count,
            "atom_mapping_version": self.atom_mapping_version,
            "formal_charge_source": "SMILES",
            "hydrogen_policy": "remove_all_atomic_number_1_isotopes",
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "InputProvenance":
        if not isinstance(payload, Mapping):
            raise TypeError("input provenance must be a mapping")
        expected = {
            "dataset_id",
            "dataset_sha256",
            "graph_manifest_sha256",
            "graph_count",
            "atom_mapping_version",
            "formal_charge_source",
            "hydrogen_policy",
        }
        if set(payload) != expected:
            raise ValueError("input provenance fields do not match the schema")
        if payload["formal_charge_source"] != "SMILES":
            raise ValueError("formal_charge_source must be 'SMILES'")
        if (
            payload["hydrogen_policy"]
            != "remove_all_atomic_number_1_isotopes"
        ):
            raise ValueError("input provenance hydrogen_policy is invalid")
        return cls(
            dataset_id=payload["dataset_id"],
            dataset_sha256=payload["dataset_sha256"],
            graph_manifest_sha256=payload["graph_manifest_sha256"],
            graph_count=payload["graph_count"],
            atom_mapping_version=payload["atom_mapping_version"],
        )


def _integer_vector(
    value: object,
    *,
    name: str,
    length: int | None = None,
) -> NDArray[np.int64]:
    array = np.asarray(value)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.issubdtype(array.dtype, np.integer) or np.issubdtype(
        array.dtype, np.bool_
    ):
        raise ValueError(f"{name} must contain integers")
    if length is not None and len(array) != length:
        raise ValueError(f"{name} must contain exactly {length} entries")
    return np.array(array, dtype=np.int64, copy=True)


def _require_permutation(array: NDArray[np.int64], *, name: str) -> None:
    expected = np.arange(len(array), dtype=np.int64)
    if not np.array_equal(np.sort(array), expected):
        raise ValueError(f"{name} must be a permutation of 0..{len(array) - 1}")


@dataclass(frozen=True)
class MoleculeInput:
    """Canonical molecule identity and graph atom order."""

    graph_idx: int
    sample_id: str
    smiles: str
    atomic_numbers: NDArray[np.int64]
    smiles_to_graph_atom_index: NDArray[np.int64]

    def __post_init__(self) -> None:
        if isinstance(self.graph_idx, bool) or not isinstance(
            self.graph_idx, (int, np.integer)
        ):
            raise TypeError("graph_idx must be an integer")
        if int(self.graph_idx) < 0:
            raise ValueError("graph_idx must be non-negative")
        if not isinstance(self.sample_id, str) or not self.sample_id:
            raise ValueError("sample_id must be a non-empty string")
        if self.sample_id != self.sample_id.strip():
            raise ValueError("sample_id must not have surrounding whitespace")
        if not isinstance(self.smiles, str) or not self.smiles:
            raise ValueError("smiles must be a non-empty string")
        if self.smiles != self.smiles.strip():
            raise ValueError("smiles must not have surrounding whitespace")

        atomic_numbers = _integer_vector(
            self.atomic_numbers, name="atomic_numbers"
        )
        if len(atomic_numbers) == 0:
            raise ValueError("atomic_numbers must not be empty")
        if np.any((atomic_numbers < 1) | (atomic_numbers > 118)):
            raise ValueError("atomic_numbers must be in the range 1..118")
        smiles_to_graph_atom_index = _integer_vector(
            self.smiles_to_graph_atom_index,
            name="smiles_to_graph_atom_index",
            length=len(atomic_numbers),
        )
        _require_permutation(
            smiles_to_graph_atom_index,
            name="smiles_to_graph_atom_index",
        )

        atomic_numbers.setflags(write=False)
        smiles_to_graph_atom_index.setflags(write=False)
        object.__setattr__(self, "graph_idx", int(self.graph_idx))
        object.__setattr__(self, "atomic_numbers", atomic_numbers)
        object.__setattr__(
            self,
            "smiles_to_graph_atom_index",
            smiles_to_graph_atom_index,
        )


@dataclass(frozen=True)
class AtomRepresentation:
    """One representation row per atom plus an explicit graph-order mapping."""

    values: NDArray[np.float32]
    atomic_numbers: NDArray[np.int64]
    encoder_atom_index: NDArray[np.int64]
    graph_atom_index: NDArray[np.int64]
    coordinates: NDArray[np.float32] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = np.asarray(self.values)
        if values.ndim != 2:
            raise ValueError("values must be a two-dimensional atom-by-feature array")
        if values.shape[0] == 0 or values.shape[1] == 0:
            raise ValueError("values must have non-zero atom and feature dimensions")
        if not np.issubdtype(values.dtype, np.floating):
            raise ValueError("values must use a real floating-point dtype")
        values = np.array(values, dtype=np.float32, copy=True)
        if not np.isfinite(values).all():
            raise ValueError("values must contain only finite numbers")

        n_atoms = values.shape[0]
        atomic_numbers = _integer_vector(
            self.atomic_numbers, name="atomic_numbers", length=n_atoms
        )
        if np.any((atomic_numbers < 1) | (atomic_numbers > 118)):
            raise ValueError("atomic_numbers must be in the range 1..118")
        encoder_atom_index = _integer_vector(
            self.encoder_atom_index,
            name="encoder_atom_index",
            length=n_atoms,
        )
        graph_atom_index = _integer_vector(
            self.graph_atom_index,
            name="graph_atom_index",
            length=n_atoms,
        )
        _require_permutation(
            encoder_atom_index, name="encoder_atom_index"
        )
        _require_permutation(graph_atom_index, name="graph_atom_index")
        coordinates = self.coordinates
        if coordinates is not None:
            coordinates = np.asarray(coordinates)
            if coordinates.shape != (n_atoms, 3):
                raise ValueError("coordinates must have shape [num_atoms, 3]")
            if not np.issubdtype(
                coordinates.dtype, np.number
            ) or np.issubdtype(coordinates.dtype, np.complexfloating):
                raise ValueError("coordinates must be real numeric values")
            coordinates = np.array(
                coordinates, dtype=np.float32, copy=True
            )
            if not np.isfinite(coordinates).all():
                raise ValueError("coordinates must contain only finite numbers")
            coordinates.setflags(write=False)
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")

        values.setflags(write=False)
        atomic_numbers.setflags(write=False)
        encoder_atom_index.setflags(write=False)
        graph_atom_index.setflags(write=False)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "atomic_numbers", atomic_numbers)
        object.__setattr__(self, "encoder_atom_index", encoder_atom_index)
        object.__setattr__(self, "graph_atom_index", graph_atom_index)
        object.__setattr__(self, "coordinates", coordinates)
        object.__setattr__(
            self, "metadata", _freeze_metadata(self.metadata)
        )


class AtomRepresentationProvider(Protocol):
    """Minimal provider interface used by the immutable cache builder."""

    name: str

    def cache_fingerprint(self) -> Mapping[str, Any]:
        """Return all settings that can change extracted values."""

    def extract_one(self, molecule: MoleculeInput) -> AtomRepresentation:
        """Extract exactly one molecule without batch padding."""


def _heavy_atom_molecule(smiles: str) -> Chem.Mol:
    """Use the heavy-atom molecular graph while preserving the declared traversal."""

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")

    editable = Chem.RWMol(molecule)
    hydrogen_indices = [
        atom.GetIdx()
        for atom in editable.GetAtoms()
        if atom.GetAtomicNum() == 1
    ]
    for atom_index in reversed(hydrogen_indices):
        editable.RemoveAtom(atom_index)
    heavy_molecule = editable.GetMol()
    Chem.SanitizeMol(heavy_molecule)
    if heavy_molecule.GetNumAtoms() == 0:
        raise ValueError("molecule has no non-hydrogen atoms")
    return heavy_molecule


def heavy_atom_numbers_from_smiles(smiles: str) -> NDArray[np.int64]:
    """Return atomic numbers after removing every hydrogen isotope."""

    molecule = _heavy_atom_molecule(smiles)
    return np.asarray(
        [atom.GetAtomicNum() for atom in molecule.GetAtoms()],
        dtype=np.int64,
    )


def heavy_atom_edges_from_smiles(smiles: str) -> NDArray[np.int64]:
    """Return the exact bidirectional bond connectivity in heavy-atom order."""

    molecule = _heavy_atom_molecule(smiles)
    edges: list[tuple[int, int]] = []
    for bond in molecule.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        edges.extend(((begin, end), (end, begin)))
    if not edges:
        return np.empty((2, 0), dtype=np.int64)
    return np.asarray(edges, dtype=np.int64).T


def mapped_bond_features_from_smiles(
    smiles: str,
    edge_index: Any,
    smiles_to_graph_atom_index: Any,
) -> NDArray[np.float32]:
    """Encode graph-ordered directed bonds after applying an explicit atom map."""

    molecule = _heavy_atom_molecule(smiles)
    _, features = _mapped_graph_chemistry(
        molecule,
        edge_index,
        smiles_to_graph_atom_index,
    )
    return features


def mapped_graph_chemistry_from_smiles(
    smiles: str,
    edge_index: Any,
    smiles_to_graph_atom_index: Any,
) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
    """Parse once and return graph-ordered atoms plus directed bond features."""

    molecule = _heavy_atom_molecule(smiles)
    return _mapped_graph_chemistry(
        molecule,
        edge_index,
        smiles_to_graph_atom_index,
    )


def _mapped_graph_chemistry(
    molecule: Chem.Mol,
    edge_index: Any,
    smiles_to_graph_atom_index: Any,
) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
    """Share one parsed molecule across atom and bond checks."""

    n_atoms = molecule.GetNumAtoms()
    mapping = _to_numpy(
        smiles_to_graph_atom_index,
        name="smiles_to_graph_atom_index",
    )
    if (
        mapping.shape != (n_atoms,)
        or not np.issubdtype(mapping.dtype, np.integer)
    ):
        raise ValueError(
            "smiles_to_graph_atom_index must be an integer atom vector"
        )
    mapping = np.asarray(mapping, dtype=np.int64)
    if not np.array_equal(
        np.sort(mapping), np.arange(n_atoms, dtype=np.int64)
    ):
        raise ValueError(
            "smiles_to_graph_atom_index must be a full atom permutation"
        )
    edges = _to_numpy(edge_index, name="graph edge_index")
    if (
        edges.ndim != 2
        or edges.shape[0] != 2
        or not np.issubdtype(edges.dtype, np.integer)
    ):
        raise ValueError("graph connectivity must have integer shape [2, num_edges]")
    edges = np.asarray(edges, dtype=np.int64)
    if edges.size and np.any((edges < 0) | (edges >= n_atoms)):
        raise ValueError("graph connectivity contains an out-of-range node")
    inverse_mapping = np.empty(n_atoms, dtype=np.int64)
    inverse_mapping[mapping] = np.arange(n_atoms, dtype=np.int64)
    graph_atomic_numbers = np.empty(n_atoms, dtype=np.int64)
    graph_atomic_numbers[mapping] = np.asarray(
        [atom.GetAtomicNum() for atom in molecule.GetAtoms()],
        dtype=np.int64,
    )

    features: list[list[float]] = []
    for graph_begin, graph_end in edges.T.tolist():
        smiles_begin = int(inverse_mapping[graph_begin])
        smiles_end = int(inverse_mapping[graph_end])
        bond = molecule.GetBondBetweenAtoms(smiles_begin, smiles_end)
        if bond is None:
            raise ValueError(
                f"mapped graph edge {(graph_begin, graph_end)} is not a bond"
            )
        bond_type = bond.GetBondType()
        one_hot = [
            float(bond_type == Chem.BondType.SINGLE),
            float(bond_type == Chem.BondType.DOUBLE),
            float(bond_type == Chem.BondType.TRIPLE),
            float(bond_type == Chem.BondType.AROMATIC),
        ]
        if sum(one_hot) != 1.0:
            raise ValueError(f"unsupported RDKit bond type: {bond_type}")
        features.append(
            one_hot
            + [
                float(bond.GetIsConjugated()),
                float(bond.IsInRing()),
            ]
        )
    return (
        graph_atomic_numbers,
        np.asarray(features, dtype=np.float32).reshape(len(features), 6),
    )


def _to_numpy(value: Any, *, name: str) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    try:
        return np.asarray(value)
    except Exception as exc:
        raise TypeError(f"{name} cannot be converted to a NumPy array") from exc


def _graph_num_nodes(graph: Any) -> int:
    value = getattr(graph, "num_nodes", None)
    if value is None:
        node_features = getattr(graph, "x", None)
        if node_features is None:
            raise ValueError("graph must expose num_nodes or x")
        value = len(node_features)
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError("graph num_nodes must be an integer")
    return int(value)


def _canonical_edges(
    edge_index: Any,
    *,
    n_nodes: int,
    source: str,
) -> tuple[tuple[int, int], ...]:
    edges = _to_numpy(edge_index, name=f"{source} edge_index")
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError(f"{source} connectivity must have shape [2, num_edges]")
    if not np.issubdtype(edges.dtype, np.integer):
        raise ValueError(f"{source} connectivity must use integer node indices")
    edges = np.asarray(edges, dtype=np.int64)
    if edges.size and np.any((edges < 0) | (edges >= n_nodes)):
        raise ValueError(f"{source} connectivity contains an out-of-range node")
    pairs = [tuple(pair) for pair in edges.T.tolist()]
    if len(pairs) != len(set(pairs)):
        raise ValueError(f"{source} connectivity contains duplicate directed edges")
    return tuple(sorted(pairs))


def molecule_input_from_graph(
    *,
    graph_idx: int,
    sample_id: str,
    smiles: str,
    graph: Any,
    smiles_to_graph_atom_index: Any,
) -> MoleculeInput:
    """Create a canonical molecule only when graph topology exactly matches SMILES."""

    smiles_atomic_numbers = heavy_atom_numbers_from_smiles(smiles)
    mapping = np.asarray(smiles_to_graph_atom_index)
    if mapping.ndim != 1 or not np.issubdtype(mapping.dtype, np.integer):
        raise ValueError(
            "smiles_to_graph_atom_index must be an integer vector"
        )
    mapping = np.asarray(mapping, dtype=np.int64)
    if not np.array_equal(
        np.sort(mapping), np.arange(len(smiles_atomic_numbers), dtype=np.int64)
    ):
        raise ValueError(
            "smiles_to_graph_atom_index must be a full atom permutation"
        )
    atomic_numbers = np.empty_like(smiles_atomic_numbers)
    atomic_numbers[mapping] = smiles_atomic_numbers
    n_nodes = _graph_num_nodes(graph)
    if n_nodes != len(atomic_numbers):
        raise ValueError(
            "graph atom count does not match the non-hydrogen SMILES atom count"
        )
    stored_atomic_numbers = getattr(graph, "atomic_numbers", None)
    if stored_atomic_numbers is None:
        raise ValueError("graph must expose explicit atomic_numbers")
    stored_atomic_numbers = _to_numpy(
        stored_atomic_numbers, name="graph atomic_numbers"
    )
    if (
        stored_atomic_numbers.ndim != 1
        or not np.issubdtype(stored_atomic_numbers.dtype, np.integer)
        or not np.array_equal(
            np.asarray(stored_atomic_numbers, dtype=np.int64),
            atomic_numbers,
        )
    ):
        raise ValueError(
            "graph atomic_numbers do not match the mapped SMILES atoms"
        )
    stored_mapping = getattr(graph, "smiles_to_graph_atom_index", None)
    if stored_mapping is None:
        raise ValueError(
            "graph must expose explicit smiles_to_graph_atom_index"
        )
    stored_mapping = _to_numpy(
        stored_mapping, name="graph smiles_to_graph_atom_index"
    )
    if (
        stored_mapping.ndim != 1
        or not np.issubdtype(stored_mapping.dtype, np.integer)
        or not np.array_equal(
            np.asarray(stored_mapping, dtype=np.int64), mapping
        )
    ):
        raise ValueError(
            "graph smiles_to_graph_atom_index does not match the requested map"
        )
    graph_edges = _canonical_edges(
        getattr(graph, "edge_index", None),
        n_nodes=n_nodes,
        source="graph",
    )
    smiles_edges = heavy_atom_edges_from_smiles(smiles)
    expected_edges = _canonical_edges(
        mapping[smiles_edges],
        n_nodes=n_nodes,
        source="SMILES",
    )
    if graph_edges != expected_edges:
        raise ValueError("graph connectivity does not match the SMILES connectivity")
    stored_edge_attr = getattr(graph, "edge_attr", None)
    if stored_edge_attr is None:
        raise ValueError("graph must expose explicit edge_attr")
    stored_edge_attr = _to_numpy(stored_edge_attr, name="graph edge_attr")
    expected_edge_attr = mapped_bond_features_from_smiles(
        smiles,
        getattr(graph, "edge_index", None),
        mapping,
    )
    if (
        stored_edge_attr.shape != expected_edge_attr.shape
        or not np.issubdtype(stored_edge_attr.dtype, np.number)
        or np.issubdtype(stored_edge_attr.dtype, np.complexfloating)
        or not np.isfinite(stored_edge_attr).all()
        or not np.array_equal(
            np.asarray(stored_edge_attr, dtype=np.float32),
            expected_edge_attr,
        )
    ):
        raise ValueError(
            "graph edge_attr does not match the mapped SMILES bond features"
        )
    return MoleculeInput(
        graph_idx=graph_idx,
        sample_id=sample_id,
        smiles=smiles,
        atomic_numbers=atomic_numbers,
        smiles_to_graph_atom_index=mapping,
    )


def validate_representation_mapping(
    molecule: MoleculeInput,
    representation: AtomRepresentation,
) -> None:
    """Reject any representation whose rows cannot map bijectively to graph atoms."""

    if len(representation.atomic_numbers) != len(molecule.atomic_numbers):
        raise ValueError(
            "representation atom count does not match the graph atom count"
        )
    expected_graph_indices = molecule.smiles_to_graph_atom_index[
        representation.encoder_atom_index
    ]
    if not np.array_equal(
        representation.graph_atom_index, expected_graph_indices
    ):
        raise ValueError(
            "representation encoder-to-graph atom mapping is inconsistent "
            "with the molecule mapping"
        )
    expected = molecule.atomic_numbers[representation.graph_atom_index]
    if not np.array_equal(representation.atomic_numbers, expected):
        raise ValueError(
            "representation atomic-number sequence does not match its graph mapping"
        )
