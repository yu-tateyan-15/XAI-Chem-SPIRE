"""Representations for notebook-based atom-contribution experiments."""

from __future__ import annotations

import hashlib
from pathlib import Path
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from typing import Any, Callable
import numpy as np
from rdkit import Chem
from .chemistry import validate_representation_mapping
from .chemistry import AtomRepresentation
from .chemistry import MoleculeInput
import csv
import json
import re
import shutil
import uuid
from typing import Any, Mapping, Sequence
from numpy.typing import NDArray
from .io import write_json
from .chemistry import heavy_atom_numbers_from_smiles
from .chemistry import validate_representation_mapping
from .chemistry import AtomRepresentation
from .chemistry import AtomRepresentationProvider
from .chemistry import InputProvenance
from .chemistry import MoleculeInput
from .chemistry import _freeze_metadata
import io
from collections.abc import Mapping
import pandas as pd
from .io import load_pyg_data
from .io import exact_integer
from .chemistry import molecule_input_from_graph
from .chemistry import InputProvenance
from .chemistry import MoleculeInput


def sha256_file(path: Path | str) -> str:
    """Return the SHA-256 digest of one external model artifact."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


OFFICIAL_CHEMELEON_SHA256 = (
    "c376624d3407204e780a0ed13a9ac097cc9bb1c13ef89cdbc633c1715c183651"
)


OFFICIAL_CHEMELEON_MD5 = "6a80b54fdb7de37ef0374d302f01e8ce"


OFFICIAL_CHEMELEON_SIZE_BYTES = 34_859_448


OFFICIAL_CHEMELEON_ZENODO_RECORD_ID = 15_460_715


OFFICIAL_CHEMELEON_ZENODO_DOI = "10.5281/zenodo.15460715"


OFFICIAL_CHEMELEON_LICENSE = "MIT"


CHEMELEON_ATOM_DIM = 2048


CHEMPROP_VERSION = "2.2.0"


_OFFICIAL_HYPER_PARAMETERS = {
    "d_v": 72,
    "d_e": 14,
    "d_h": CHEMELEON_ATOM_DIM,
    "bias": False,
    "depth": 6,
    "dropout": 0.0,
    "activation": "relu",
    "undirected": False,
    "d_vd": None,
    "V_d_transform": None,
    "graph_transform": None,
}


def _md5_file(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class _CheMeleonRuntime:
    featurizer: Any
    batch_mol_graph_factory: Callable[..., Any]
    encoder: Any
    chemprop_version: str


class CheMeleonSingletonProvider:
    """Load only the SHA-pinned official CheMeleon message-passing weights."""

    name = "chemeleon"

    def __init__(
        self,
        *,
        checkpoint_path: Path | str,
        device: str = "cpu",
        runtime_factory: Callable[..., Any] | None = None,
    ) -> None:
        checkpoint = Path(checkpoint_path).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"CheMeleon checkpoint does not exist: {checkpoint}"
            )
        normalized_device = str(device).strip().lower()
        if normalized_device == "cuda":
            normalized_device = "cuda:0"
        if normalized_device not in {"cpu", "cuda:0"}:
            raise ValueError("device must be 'cpu', 'cuda', or 'cuda:0'")
        checkpoint_sha256 = sha256_file(checkpoint)
        checkpoint_md5 = _md5_file(checkpoint)
        checkpoint_size_bytes = checkpoint.stat().st_size
        if (
            runtime_factory is None
            and checkpoint_sha256 != OFFICIAL_CHEMELEON_SHA256
        ):
            raise ValueError(
                "checkpoint SHA-256 is not the approved CheMeleon checkpoint"
            )
        if (
            runtime_factory is None
            and checkpoint_md5 != OFFICIAL_CHEMELEON_MD5
        ):
            raise ValueError(
                "checkpoint MD5 does not match Zenodo record 15460715"
            )
        if (
            runtime_factory is None
            and checkpoint_size_bytes != OFFICIAL_CHEMELEON_SIZE_BYTES
        ):
            raise ValueError(
                "checkpoint size does not match Zenodo record 15460715"
            )
        self.checkpoint_path = checkpoint
        self.device = normalized_device
        self._checkpoint_sha256 = checkpoint_sha256
        self._checkpoint_md5 = checkpoint_md5
        self._checkpoint_size_bytes = checkpoint_size_bytes
        self._runtime_factory = runtime_factory
        self._runtime: Any | None = None

    def cache_fingerprint(self) -> dict[str, Any]:
        """Describe the exact checkpoint, encoder tap, and 2D atom contract."""

        if self._runtime_factory is not None:
            raise RuntimeError(
                "custom runtime_factory is test-only and cannot build a "
                "scientific cache"
            )
        runtime = self._ensure_runtime()
        import torch
        from rdkit import rdBase

        fingerprint = {
            "provider": self.name,
            "model_name": "CheMeleon",
            "zenodo_record_id": OFFICIAL_CHEMELEON_ZENODO_RECORD_ID,
            "zenodo_doi": OFFICIAL_CHEMELEON_ZENODO_DOI,
            "checkpoint_filename": "chemeleon_mp.pt",
            "checkpoint_sha256": self._checkpoint_sha256,
            "checkpoint_md5": self._checkpoint_md5,
            "checkpoint_size_bytes": self._checkpoint_size_bytes,
            "checkpoint_source": (
                "https://zenodo.org/records/15460715/files/"
                "chemeleon_mp.pt"
            ),
            "checkpoint_trust_policy": (
                "zenodo_published_md5_and_size_plus_local_sha256_pin_v1"
            ),
            "checkpoint_coverage_policy": "state_dict_strict_true_exact_v1",
            "chemprop_version": runtime.chemprop_version,
            "chemprop_required_version": CHEMPROP_VERSION,
            "code_license": OFFICIAL_CHEMELEON_LICENSE,
            "weights_license": OFFICIAL_CHEMELEON_LICENSE,
            "code_license_source": (
                "https://github.com/JacksonBurns/chemeleon/blob/main/LICENSE"
            ),
            "weights_license_source": (
                "https://zenodo.org/records/15460715"
            ),
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "rdkit_version": rdBase.rdkitVersion,
            "device": self.device,
            "input_modality": "2d_graph",
            "batch_size": 1,
            "native_batching": False,
            "embedding_dim": CHEMELEON_ATOM_DIM,
            "atom_order": "rdkit_smiles_atom_order",
            "graph_mapping": "explicit_smiles_to_graph_atom_index",
            "explicit_hydrogen_policy": "reject_h_d_t",
            "extraction_point": (
                "BondMessagePassing.forward_output_before_mean_pooling"
            ),
        }
        if self.device == "cuda:0":
            fingerprint.update(
                {
                    "cuda_runtime_version": torch.version.cuda,
                    "cuda_device_name": torch.cuda.get_device_name(0),
                }
            )
        return fingerprint

    def _build_official_runtime(self) -> _CheMeleonRuntime:
        try:
            from chemprop import featurizers, nn
            from chemprop.data import BatchMolGraph
        except ImportError as exc:
            raise ImportError(
                "CheMeleon extraction requires chemprop==2.2.0 and "
                "Python >=3.11; install the optional dependency in an "
                "isolated environment"
            ) from exc
        try:
            chemprop_version = importlib_metadata.version("chemprop")
        except importlib_metadata.PackageNotFoundError as exc:
            raise ImportError(
                "CheMeleon extraction requires chemprop==2.2.0 and "
                "Python >=3.11"
            ) from exc
        if chemprop_version != CHEMPROP_VERSION:
            raise ImportError(
                "CheMeleon extraction requires exactly chemprop==2.2.0; "
                f"found {chemprop_version!r}"
            )

        import torch

        if self.device == "cuda:0" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested for CheMeleon, but CUDA is unavailable"
            )
        try:
            checkpoint = torch.load(
                self.checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
        except TypeError as exc:
            raise RuntimeError(
                "secure weights-only checkpoint loading is required; use "
                "a PyTorch version supporting weights_only=True"
            ) from exc
        if not isinstance(checkpoint, dict) or set(checkpoint) != {
            "hyper_parameters",
            "state_dict",
        }:
            raise ValueError(
                "CheMeleon checkpoint must contain exactly "
                "'hyper_parameters' and 'state_dict'"
            )
        hyper_parameters = checkpoint["hyper_parameters"]
        if (
            not isinstance(hyper_parameters, dict)
            or hyper_parameters != _OFFICIAL_HYPER_PARAMETERS
        ):
            raise ValueError(
                "CheMeleon checkpoint hyper-parameters do not match the "
                "approved 2048-dimensional D-MPNN"
            )
        state_dict = checkpoint["state_dict"]
        if not isinstance(state_dict, dict):
            raise ValueError(
                "CheMeleon checkpoint state_dict must be a mapping"
            )

        encoder = nn.BondMessagePassing(**hyper_parameters)
        try:
            incompatible = encoder.load_state_dict(
                state_dict, strict=True
            )
        except RuntimeError as exc:
            raise ValueError(
                f"CheMeleon checkpoint strict load failed: {exc}"
            ) from exc
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise ValueError(
                "CheMeleon checkpoint did not cover the encoder exactly: "
                f"missing={list(incompatible.missing_keys)}, "
                f"unexpected={list(incompatible.unexpected_keys)}"
            )
        featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
        if (
            featurizer.atom_fdim != _OFFICIAL_HYPER_PARAMETERS["d_v"]
            or featurizer.bond_fdim != _OFFICIAL_HYPER_PARAMETERS["d_e"]
        ):
            raise ValueError(
                "Chemprop default graph features do not match the approved "
                "CheMeleon checkpoint"
            )
        return _CheMeleonRuntime(
            featurizer=featurizer,
            batch_mol_graph_factory=BatchMolGraph,
            encoder=encoder,
            chemprop_version=chemprop_version,
        )

    def _ensure_runtime(self) -> Any:
        if self._runtime is not None:
            return self._runtime
        runtime = (
            self._build_official_runtime()
            if self._runtime_factory is None
            else self._runtime_factory(
                checkpoint_path=self.checkpoint_path,
                device=self.device,
            )
        )
        for name in (
            "featurizer",
            "batch_mol_graph_factory",
            "encoder",
            "chemprop_version",
        ):
            if not hasattr(runtime, name):
                raise TypeError(
                    f"CheMeleon runtime must expose {name!r}"
                )
        if not callable(runtime.featurizer):
            raise TypeError("CheMeleon featurizer must be callable")
        if not callable(
            getattr(runtime.featurizer, "atom_featurizer", None)
        ):
            raise TypeError(
                "CheMeleon featurizer must expose the Chemprop 2.2.0 "
                "atom_featurizer callable"
            )
        if not callable(runtime.batch_mol_graph_factory):
            raise TypeError("CheMeleon batch graph factory must be callable")
        if not callable(runtime.encoder):
            raise TypeError("CheMeleon encoder must be callable")
        if getattr(runtime.encoder, "output_dim", None) != CHEMELEON_ATOM_DIM:
            raise ValueError("CheMeleon encoder output_dim must be 2048")
        runtime.encoder.eval()
        runtime.encoder.to(self.device)
        self._runtime = runtime
        return runtime

    @staticmethod
    def _parse_heavy_atoms(molecule: MoleculeInput) -> Chem.Mol:
        parser = Chem.SmilesParserParams()
        parser.removeHs = False
        parsed_with_hydrogens = Chem.MolFromSmiles(molecule.smiles, parser)
        if parsed_with_hydrogens is None:
            raise ValueError(
                f"RDKit could not parse SMILES for {molecule.sample_id!r}"
            )
        if any(
            atom.GetAtomicNum() == 1
            for atom in parsed_with_hydrogens.GetAtoms()
        ):
            raise ValueError(
                "CheMeleon provider forbids explicit H/D/T atoms because "
                "the graph contract contains heavy atoms only"
            )
        parsed = Chem.MolFromSmiles(molecule.smiles)
        if parsed is None:
            raise ValueError(
                f"RDKit could not parse SMILES for {molecule.sample_id!r}"
            )
        expected_atomic_numbers = molecule.atomic_numbers[
            molecule.smiles_to_graph_atom_index
        ]
        parsed_atomic_numbers = np.asarray(
            [atom.GetAtomicNum() for atom in parsed.GetAtoms()],
            dtype=np.int64,
        )
        if not np.array_equal(
            parsed_atomic_numbers, expected_atomic_numbers
        ):
            raise ValueError(
                "CheMeleon RDKit atom order does not match the explicit "
                "SMILES-to-graph mapping"
            )
        return parsed

    def extract_one(self, molecule: MoleculeInput) -> AtomRepresentation:
        """Return Chemprop message-passing atom states before mean pooling."""

        parsed = self._parse_heavy_atoms(molecule)
        runtime = self._ensure_runtime()
        mol_graph = runtime.featurizer(parsed)
        n_atoms = len(molecule.atomic_numbers)
        vertices = np.asarray(getattr(mol_graph, "V", None))
        if vertices.ndim != 2 or vertices.shape[0] != n_atoms:
            raise ValueError(
                "CheMeleon featurizer atom count does not match the molecule"
            )
        expected_vertices = np.asarray(
            [
                runtime.featurizer.atom_featurizer(atom)
                for atom in parsed.GetAtoms()
            ],
            dtype=np.single,
        )
        if (
            vertices.shape != expected_vertices.shape
            or not np.array_equal(vertices, expected_vertices)
        ):
            raise ValueError(
                "CheMeleon feature rows do not match parsed RDKit atom order"
            )
        batch_graph = runtime.batch_mol_graph_factory([mol_graph])
        if len(batch_graph) != 1:
            raise ValueError(
                "CheMeleon extraction requires exactly one molecule per call"
            )
        batch_vertices = getattr(batch_graph, "V", None)
        batch_membership = getattr(batch_graph, "batch", None)
        if batch_vertices is None or tuple(batch_vertices.shape)[0] != n_atoms:
            raise ValueError(
                "CheMeleon singleton batch atom count is inconsistent"
            )
        batch_vertices_array = np.asarray(
            batch_vertices.detach().cpu()
            if hasattr(batch_vertices, "detach")
            else batch_vertices
        )
        if (
            batch_vertices_array.shape != vertices.shape
            or not np.array_equal(batch_vertices_array, vertices)
        ):
            raise ValueError(
                "CheMeleon batching changed atom feature row order"
            )
        if batch_membership is None:
            raise ValueError(
                "CheMeleon singleton batch must expose atom membership"
            )
        membership = np.asarray(
            batch_membership.detach().cpu()
            if hasattr(batch_membership, "detach")
            else batch_membership
        )
        if membership.shape != (n_atoms,) or np.any(membership != 0):
            raise ValueError(
                "CheMeleon singleton batch membership is inconsistent"
            )
        batch_graph.to(self.device)

        import torch

        with torch.inference_mode():
            hidden = runtime.encoder(batch_graph)
        if not hasattr(hidden, "detach"):
            raise TypeError(
                "CheMeleon message passing must return a torch.Tensor"
            )
        values = hidden.detach().cpu().numpy()
        if values.shape != (n_atoms, CHEMELEON_ATOM_DIM):
            raise ValueError(
                "CheMeleon pre-pooling atom representation must have shape "
                f"{(n_atoms, CHEMELEON_ATOM_DIM)}, got {values.shape}"
            )

        identity = np.arange(n_atoms, dtype=np.int64)
        representation = AtomRepresentation(
            values=values,
            atomic_numbers=molecule.atomic_numbers[
                molecule.smiles_to_graph_atom_index
            ],
            encoder_atom_index=identity,
            graph_atom_index=molecule.smiles_to_graph_atom_index,
            coordinates=None,
            metadata={
                "checkpoint_sha256": self._checkpoint_sha256,
                "checkpoint_md5": self._checkpoint_md5,
                "zenodo_record_id": OFFICIAL_CHEMELEON_ZENODO_RECORD_ID,
                "input_modality": "2d_graph",
                "atom_order": "rdkit_smiles_atom_order",
                "extraction_point": (
                    "BondMessagePassing.forward_output_before_mean_pooling"
                ),
                "native_batch_size": 1,
                "embedding_dim": CHEMELEON_ATOM_DIM,
                "chemprop_version": runtime.chemprop_version,
            },
        )
        validate_representation_mapping(molecule, representation)
        return representation


CACHE_ID_SCHEMA_VERSION = 1


LEGACY_MANIFEST_SCHEMA_VERSION = 1


MANIFEST_SCHEMA_VERSION = 2


_HISTORICAL_MANIFEST_RECORDS_KEY = "artifacts"


ARRAY_FILENAMES = (
    "representations.npy",
    "offsets.npy",
    "atomic_numbers.npy",
    "encoder_atom_index.npy",
    "graph_atom_index.npy",
)


COORDINATE_FILENAME = "coordinates.npy"


REQUIRED_FILENAMES = frozenset(
    ("manifest.json", "molecules.csv", *ARRAY_FILENAMES, "COMPLETE")
)


MOLECULE_COLUMNS = (
    "graph_idx",
    "sample_id",
    "smiles",
    "n_atoms",
    "offset_start",
    "offset_stop",
    "provider_metadata",
)


_SAFE_PROVIDER_NAME = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")


class CacheValidationError(ValueError):
    """A cache exists but is incomplete, corrupt, or incompatible."""


def _json_default(value: Any) -> Any:
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(
        f"Object of type {type(value).__name__} is not JSON serializable"
    )


def _canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
        default=_json_default,
    )


def _canonical_fingerprint(fingerprint: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(fingerprint, Mapping):
        raise TypeError("provider fingerprint must be a mapping")
    serialized = _canonical_json(dict(fingerprint))
    decoded = json.loads(serialized)
    if not isinstance(decoded, dict):
        raise TypeError("provider fingerprint must serialize to a JSON object")
    return decoded


def _validate_singleton_extraction_policy(
    fingerprint: Mapping[str, Any],
) -> None:
    """Reject scientific caches that opt into native mixed batching."""

    batch_size = fingerprint.get("batch_size")
    if batch_size is not None and batch_size != 1:
        raise ValueError(
            "scientific atom-representation caches require batch_size=1"
        )
    native_batching = fingerprint.get("native_batching")
    if native_batching is not None and native_batching is not False:
        raise ValueError(
            "scientific atom-representation caches forbid native batching"
        )


def _validate_molecule_sequence(
    molecules: Sequence[MoleculeInput],
) -> tuple[MoleculeInput, ...]:
    items = tuple(molecules)
    if not items:
        raise ValueError("at least one molecule is required")
    expected_indices = list(range(len(items)))
    actual_indices = [molecule.graph_idx for molecule in items]
    if actual_indices != expected_indices:
        raise ValueError("graph_idx must be contiguous and ordered from zero")
    sample_ids = [molecule.sample_id for molecule in items]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("sample_id values must be unique")
    return items


def build_cache_id(
    provider_fingerprint: Mapping[str, Any],
    molecules: Sequence[MoleculeInput],
    *,
    input_provenance: InputProvenance,
) -> str:
    """Hash every provider setting and molecule identity that defines a cache."""

    items = _validate_molecule_sequence(molecules)
    if not isinstance(input_provenance, InputProvenance):
        raise TypeError("input_provenance must be InputProvenance")
    if input_provenance.graph_count != len(items):
        raise ValueError("input provenance graph_count does not match molecules")
    for molecule in items:
        expected_numbers = heavy_atom_numbers_from_smiles(molecule.smiles)
        observed_smiles_order = molecule.atomic_numbers[
            molecule.smiles_to_graph_atom_index
        ]
        if not np.array_equal(observed_smiles_order, expected_numbers):
            raise ValueError(
                "molecule atomic numbers do not match its SMILES at "
                f"graph_idx {molecule.graph_idx}"
            )
    payload = {
        "schema_version": CACHE_ID_SCHEMA_VERSION,
        "provider": _canonical_fingerprint(provider_fingerprint),
        "input_provenance": input_provenance.as_dict(),
        "molecules": [
            {
                "graph_idx": molecule.graph_idx,
                "sample_id": molecule.sample_id,
                "smiles": molecule.smiles,
                "atomic_numbers": molecule.atomic_numbers.tolist(),
                "smiles_to_graph_atom_index": (
                    molecule.smiles_to_graph_atom_index.tolist()
                ),
            }
            for molecule in items
        ],
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RepresentationBundle:
    """Memory-mapped cache and its validated molecule index."""

    root: Path
    manifest: Mapping[str, Any]
    molecule_rows: tuple[Mapping[str, Any], ...]
    values: NDArray[np.float32]
    offsets: NDArray[np.int64]
    atomic_numbers: NDArray[np.integer]
    encoder_atom_index: NDArray[np.int64]
    graph_atom_index: NDArray[np.int64]
    coordinates: NDArray[np.float32] | None

    @property
    def metadata(self) -> Mapping[str, Any]:
        """Expose cache-level information through a neutral public name."""

        return self.manifest

    def get(self, graph_idx: int) -> AtomRepresentation:
        if isinstance(graph_idx, bool) or not isinstance(
            graph_idx, (int, np.integer)
        ):
            raise TypeError("graph_idx must be an integer")
        index = int(graph_idx)
        if index < 0 or index >= len(self.molecule_rows):
            raise KeyError(f"graph_idx {index} is absent from the cache")
        row = self.molecule_rows[index]
        if row["graph_idx"] != index:
            raise CacheValidationError("molecule index is not graph_idx ordered")
        start = int(self.offsets[index])
        stop = int(self.offsets[index + 1])
        return AtomRepresentation(
            values=self.values[start:stop],
            atomic_numbers=self.atomic_numbers[start:stop],
            encoder_atom_index=self.encoder_atom_index[start:stop],
            graph_atom_index=self.graph_atom_index[start:stop],
            coordinates=(
                None
                if self.coordinates is None
                else self.coordinates[start:stop]
            ),
            metadata={
                "cache_id": self.manifest["cache_id"],
                "sample_id": row["sample_id"],
                "provider": row["provider_metadata"],
            },
        )


def _output_record(path: Path, array: np.ndarray) -> dict[str, Any]:
    return {
        "filename": path.name,
        "sha256": sha256_file(path),
        "shape": list(array.shape),
        "dtype": str(array.dtype),
    }


def _provider_metadata_with_coordinate_digest(
    output: AtomRepresentation,
) -> dict[str, Any]:
    metadata = dict(output.metadata)
    if output.coordinates is None:
        if "returned_coordinate_sha256" in metadata:
            raise ValueError(
                "coordinate-free output declares a coordinate digest"
            )
        return metadata
    digest = hashlib.sha256(
        np.ascontiguousarray(output.coordinates, dtype="<f4").tobytes()
    ).hexdigest()
    declared = metadata.get("returned_coordinate_sha256")
    if declared is not None and declared != digest:
        raise ValueError(
            "provider coordinate digest does not match its coordinates"
        )
    metadata["returned_coordinate_sha256"] = digest
    return metadata


def _write_molecules_csv(
    path: Path,
    molecules: Sequence[MoleculeInput],
    offsets: NDArray[np.int64],
    outputs: Sequence[AtomRepresentation],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MOLECULE_COLUMNS)
        writer.writeheader()
        for index, molecule in enumerate(molecules):
            writer.writerow(
                {
                    "graph_idx": molecule.graph_idx,
                    "sample_id": molecule.sample_id,
                    "smiles": molecule.smiles,
                    "n_atoms": len(molecule.atomic_numbers),
                    "offset_start": int(offsets[index]),
                    "offset_stop": int(offsets[index + 1]),
                    "provider_metadata": _canonical_json(
                        _provider_metadata_with_coordinate_digest(
                            outputs[index]
                        )
                    ),
                }
            )


def _write_cache(
    target: Path,
    *,
    cache_id: str,
    fingerprint: Mapping[str, Any],
    input_provenance: InputProvenance,
    molecules: Sequence[MoleculeInput],
    outputs: Sequence[AtomRepresentation],
) -> None:
    """Persist raw singleton atom states together with offsets and atom mappings."""

    dimensions = {output.values.shape[1] for output in outputs}
    if len(dimensions) != 1:
        raise ValueError("all representations must have the same feature dimension")
    coordinate_presence = {
        output.coordinates is not None for output in outputs
    }
    if len(coordinate_presence) != 1:
        raise ValueError(
            "a cache cannot mix coordinate-bearing and coordinate-free outputs"
        )
    has_coordinates = coordinate_presence == {True}
    for output in outputs:
        _provider_metadata_with_coordinate_digest(output)

    counts = np.asarray(
        [len(output.atomic_numbers) for output in outputs], dtype=np.int64
    )
    offsets = np.concatenate(
        (np.zeros(1, dtype=np.int64), np.cumsum(counts, dtype=np.int64))
    )
    arrays: dict[str, np.ndarray] = {
        "representations.npy": np.concatenate(
            [output.values for output in outputs], axis=0
        ).astype(np.float32, copy=False),
        "offsets.npy": offsets,
        "atomic_numbers.npy": np.concatenate(
            [output.atomic_numbers for output in outputs]
        ).astype(np.int16, copy=False),
        "encoder_atom_index.npy": np.concatenate(
            [output.encoder_atom_index for output in outputs]
        ).astype(np.int64, copy=False),
        "graph_atom_index.npy": np.concatenate(
            [output.graph_atom_index for output in outputs]
        ).astype(np.int64, copy=False),
    }
    if has_coordinates:
        arrays[COORDINATE_FILENAME] = np.concatenate(
            [output.coordinates for output in outputs], axis=0
        ).astype(np.float32, copy=False)

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir()
    try:
        for filename, array in arrays.items():
            np.save(temporary / filename, array, allow_pickle=False)
        _write_molecules_csv(
            temporary / "molecules.csv", molecules, offsets, outputs
        )

        output_records = {
            filename: _output_record(temporary / filename, array)
            for filename, array in arrays.items()
        }
        csv_path = temporary / "molecules.csv"
        output_records["molecules.csv"] = {
            "filename": csv_path.name,
            "sha256": sha256_file(csv_path),
            "rows": len(molecules),
            "columns": list(MOLECULE_COLUMNS),
        }
        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "cache_id": cache_id,
            "provider": dict(fingerprint),
            "input_provenance": input_provenance.as_dict(),
            "extraction_mode": "singleton",
            "coordinate_mode": (
                "cartesian_3d" if has_coordinates else "none"
            ),
            "molecule_count": len(molecules),
            "atom_count": int(offsets[-1]),
            "embedding_dim": int(next(iter(dimensions))),
            "outputs": output_records,
        }
        manifest_path = write_json(temporary / "manifest.json", manifest)
        (temporary / "COMPLETE").write_text(
            f"{sha256_file(manifest_path)}\n", encoding="ascii"
        )
        if target.exists():
            raise FileExistsError(f"cache target appeared during build: {target}")
        temporary.rename(target)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _read_manifest(root: Path) -> dict[str, Any]:
    try:
        payload = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CacheValidationError(f"invalid manifest.json: {exc}") from exc
    if not isinstance(payload, dict):
        raise CacheValidationError("manifest.json must contain a JSON object")
    return payload


def _manifest_output_records(
    manifest: Mapping[str, Any],
) -> Mapping[str, Any]:
    schema_version = manifest.get("schema_version")
    if schema_version == MANIFEST_SCHEMA_VERSION:
        if _HISTORICAL_MANIFEST_RECORDS_KEY in manifest:
            raise CacheValidationError(
                "schema-v2 manifest must use outputs, not the legacy key"
            )
        records = manifest.get("outputs")
        field_name = "outputs"
    elif schema_version == LEGACY_MANIFEST_SCHEMA_VERSION:
        if "outputs" in manifest:
            raise CacheValidationError(
                "schema-v1 manifest must use its historical key exactly"
            )
        records = manifest.get(_HISTORICAL_MANIFEST_RECORDS_KEY)
        field_name = "historical output records"
    else:
        raise CacheValidationError("unsupported cache schema_version")
    if not isinstance(records, Mapping):
        raise CacheValidationError(f"manifest {field_name} must be an object")
    return records


def _verify_file_set_and_hashes(root: Path, manifest: Mapping[str, Any]) -> None:
    entries = tuple(root.iterdir())
    invalid_entries = [
        path.name for path in entries if path.is_symlink() or not path.is_file()
    ]
    if invalid_entries:
        raise CacheValidationError(
            "cache contains non-regular entries: "
            + ", ".join(sorted(invalid_entries))
        )
    coordinate_mode = manifest.get("coordinate_mode")
    if coordinate_mode not in {"none", "cartesian_3d"}:
        raise CacheValidationError("manifest coordinate_mode is invalid")
    expected_files = set(REQUIRED_FILENAMES)
    expected_outputs = set(ARRAY_FILENAMES) | {"molecules.csv"}
    if coordinate_mode == "cartesian_3d":
        expected_files.add(COORDINATE_FILENAME)
        expected_outputs.add(COORDINATE_FILENAME)
    actual = {path.name for path in entries}
    missing = expected_files - actual
    if missing:
        raise CacheValidationError(
            f"cache is incomplete; missing {', '.join(sorted(missing))}"
        )
    unexpected = actual - expected_files
    if unexpected:
        raise CacheValidationError(
            f"cache contains unexpected files: {', '.join(sorted(unexpected))}"
        )

    try:
        complete_digest = (root / "COMPLETE").read_text(
            encoding="ascii"
        ).strip()
    except (OSError, UnicodeError) as exc:
        raise CacheValidationError(f"cannot read COMPLETE: {exc}") from exc
    manifest_digest = sha256_file(root / "manifest.json")
    if complete_digest != manifest_digest:
        raise CacheValidationError("COMPLETE does not match manifest SHA-256")

    outputs = _manifest_output_records(manifest)
    if set(outputs) != expected_outputs:
        raise CacheValidationError("manifest output set is not exact")
    for filename in sorted(expected_outputs):
        record = outputs[filename]
        if not isinstance(record, dict) or record.get("filename") != filename:
            raise CacheValidationError(f"invalid output record for {filename}")
        expected_digest = record.get("sha256")
        if not isinstance(expected_digest, str) or len(expected_digest) != 64:
            raise CacheValidationError(f"invalid SHA-256 record for {filename}")
        if sha256_file(root / filename) != expected_digest:
            raise CacheValidationError(f"SHA-256 mismatch for {filename}")


def _load_molecule_rows(path: Path) -> tuple[dict[str, Any], ...]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != MOLECULE_COLUMNS:
                raise CacheValidationError(
                    "molecules.csv columns do not match the cache schema"
                )
            raw_rows = list(reader)
    except OSError as exc:
        raise CacheValidationError(f"cannot read molecules.csv: {exc}") from exc

    rows: list[dict[str, Any]] = []
    try:
        for raw in raw_rows:
            provider_metadata = json.loads(raw["provider_metadata"])
            if not isinstance(provider_metadata, dict):
                raise ValueError("provider_metadata must be a JSON object")
            rows.append(
                {
                    "graph_idx": int(raw["graph_idx"]),
                    "sample_id": raw["sample_id"],
                    "smiles": raw["smiles"],
                    "n_atoms": int(raw["n_atoms"]),
                    "offset_start": int(raw["offset_start"]),
                    "offset_stop": int(raw["offset_stop"]),
                    "provider_metadata": provider_metadata,
                }
            )
    except (KeyError, TypeError, ValueError) as exc:
        raise CacheValidationError(
            f"molecules.csv contains an invalid value: {exc}"
        ) from exc
    return tuple(rows)


def _load_array(root: Path, filename: str) -> np.ndarray:
    try:
        return np.load(root / filename, allow_pickle=False, mmap_mode="r")
    except (OSError, ValueError) as exc:
        raise CacheValidationError(f"cannot load {filename}: {exc}") from exc


def _validate_loaded_bundle(bundle: RepresentationBundle) -> None:
    """Audit cached array alignment before attaching atom states to graphs."""

    manifest = bundle.manifest
    _manifest_output_records(manifest)
    cache_id = manifest.get("cache_id")
    if not isinstance(cache_id, str) or len(cache_id) != 64:
        raise CacheValidationError("manifest cache_id is invalid")
    if manifest.get("extraction_mode") != "singleton":
        raise CacheValidationError("cache was not extracted in singleton mode")
    coordinate_mode = manifest.get("coordinate_mode")
    if coordinate_mode not in {"none", "cartesian_3d"}:
        raise CacheValidationError("manifest coordinate_mode is invalid")
    fingerprint = manifest.get("provider")
    if not isinstance(fingerprint, Mapping):
        raise CacheValidationError("manifest provider must be an object")
    try:
        canonical_fingerprint = _canonical_fingerprint(fingerprint)
        _validate_singleton_extraction_policy(canonical_fingerprint)
    except (TypeError, ValueError) as exc:
        raise CacheValidationError(
            f"manifest provider fingerprint is invalid: {exc}"
        ) from exc
    try:
        input_provenance = InputProvenance.from_dict(
            manifest.get("input_provenance")
        )
    except (TypeError, ValueError) as exc:
        raise CacheValidationError(
            f"manifest input provenance is invalid: {exc}"
        ) from exc

    rows = bundle.molecule_rows
    n_molecules = len(rows)
    if manifest.get("molecule_count") != n_molecules:
        raise CacheValidationError("manifest molecule_count does not match CSV")
    if input_provenance.graph_count != n_molecules:
        raise CacheValidationError(
            "input provenance graph_count does not match the cache"
        )
    if [row["graph_idx"] for row in rows] != list(range(n_molecules)):
        raise CacheValidationError("molecules.csv graph_idx must be ordered from zero")
    sample_ids = [row["sample_id"] for row in rows]
    if any(not value for value in sample_ids) or len(sample_ids) != len(
        set(sample_ids)
    ):
        raise CacheValidationError("molecules.csv sample_id values must be unique")
    if any(not row["smiles"] for row in rows):
        raise CacheValidationError("molecules.csv contains an empty SMILES")

    offsets = bundle.offsets
    if offsets.dtype != np.dtype(np.int64) or offsets.shape != (
        n_molecules + 1,
    ):
        raise CacheValidationError("offsets.npy must be int64 with length N+1")
    if offsets[0] != 0 or np.any(np.diff(offsets) <= 0):
        raise CacheValidationError("offsets.npy must be strictly increasing from zero")
    n_atoms = int(offsets[-1])
    if manifest.get("atom_count") != n_atoms:
        raise CacheValidationError("manifest atom_count does not match offsets")

    values = bundle.values
    embedding_dim = manifest.get("embedding_dim")
    if (
        values.dtype != np.dtype(np.float32)
        or values.ndim != 2
        or values.shape != (n_atoms, embedding_dim)
    ):
        raise CacheValidationError(
            "representations.npy shape or dtype does not match the manifest"
        )
    if not np.isfinite(values).all():
        raise CacheValidationError("representations.npy contains non-finite values")
    coordinates = bundle.coordinates
    if coordinate_mode == "none":
        if coordinates is not None:
            raise CacheValidationError(
                "coordinate-free cache unexpectedly loaded coordinates"
            )
    else:
        if (
            coordinates is None
            or coordinates.dtype != np.dtype(np.float32)
            or coordinates.shape != (n_atoms, 3)
        ):
            raise CacheValidationError(
                "coordinates.npy must be float32 with shape [atom_count, 3]"
            )
        if not np.isfinite(coordinates).all():
            raise CacheValidationError(
                "coordinates.npy contains non-finite values"
            )

    one_dimensional = {
        "atomic_numbers.npy": bundle.atomic_numbers,
        "encoder_atom_index.npy": bundle.encoder_atom_index,
        "graph_atom_index.npy": bundle.graph_atom_index,
    }
    for filename, array in one_dimensional.items():
        if array.ndim != 1 or len(array) != n_atoms:
            raise CacheValidationError(f"{filename} must have atom_count entries")
        if not np.issubdtype(array.dtype, np.integer):
            raise CacheValidationError(f"{filename} must contain integers")
    if np.any((bundle.atomic_numbers < 1) | (bundle.atomic_numbers > 118)):
        raise CacheValidationError("atomic_numbers.npy contains an invalid element")

    outputs = _manifest_output_records(manifest)
    arrays = {
        "representations.npy": values,
        "offsets.npy": offsets,
        **one_dimensional,
    }
    if coordinates is not None:
        arrays[COORDINATE_FILENAME] = coordinates
    for filename, array in arrays.items():
        record = outputs[filename]
        if tuple(record.get("shape", ())) != tuple(array.shape):
            raise CacheValidationError(f"manifest shape mismatch for {filename}")
        if record.get("dtype") != str(array.dtype):
            raise CacheValidationError(f"manifest dtype mismatch for {filename}")

    for index, row in enumerate(rows):
        start = int(offsets[index])
        stop = int(offsets[index + 1])
        if (
            row["n_atoms"] != stop - start
            or row["offset_start"] != start
            or row["offset_stop"] != stop
        ):
            raise CacheValidationError(
                f"molecules.csv offsets disagree at graph_idx {index}"
            )
        expected = np.arange(stop - start, dtype=np.int64)
        if not np.array_equal(
            np.sort(bundle.encoder_atom_index[start:stop]), expected
        ):
            raise CacheValidationError(
                f"encoder_atom_index is not a permutation at graph_idx {index}"
            )
        if not np.array_equal(
            np.sort(bundle.graph_atom_index[start:stop]), expected
        ):
            raise CacheValidationError(
                f"graph_atom_index is not a permutation at graph_idx {index}"
            )
        if coordinates is not None:
            coordinate_digest = hashlib.sha256(
                np.ascontiguousarray(
                    coordinates[start:stop], dtype="<f4"
                ).tobytes()
            ).hexdigest()
            provider_metadata = row["provider_metadata"]
            if (
                provider_metadata.get("returned_coordinate_sha256")
                != coordinate_digest
            ):
                raise CacheValidationError(
                    "coordinate provenance digest mismatch at "
                    f"graph_idx {index}"
                )

    reconstructed_molecules: list[MoleculeInput] = []
    for index, row in enumerate(rows):
        start = int(offsets[index])
        stop = int(offsets[index + 1])
        graph_order_numbers = np.empty(stop - start, dtype=np.int64)
        graph_order_numbers[bundle.graph_atom_index[start:stop]] = (
            bundle.atomic_numbers[start:stop]
        )
        smiles_to_graph_atom_index = np.empty(stop - start, dtype=np.int64)
        smiles_to_graph_atom_index[
            bundle.encoder_atom_index[start:stop]
        ] = bundle.graph_atom_index[start:stop]
        try:
            reconstructed_molecules.append(
                MoleculeInput(
                    graph_idx=row["graph_idx"],
                    sample_id=row["sample_id"],
                    smiles=row["smiles"],
                    atomic_numbers=graph_order_numbers,
                    smiles_to_graph_atom_index=(
                        smiles_to_graph_atom_index
                    ),
                )
            )
        except (TypeError, ValueError) as exc:
            raise CacheValidationError(
                f"invalid molecule record at graph_idx {index}: {exc}"
            ) from exc
    try:
        expected_cache_id = build_cache_id(
            canonical_fingerprint,
            reconstructed_molecules,
            input_provenance=input_provenance,
        )
    except (TypeError, ValueError) as exc:
        raise CacheValidationError(
            f"cached molecule chemistry is invalid: {exc}"
        ) from exc
    if cache_id != expected_cache_id:
        raise CacheValidationError(
            "manifest cache_id does not match the cached scientific content"
        )


def _validate_open_bundle_structure(bundle: RepresentationBundle) -> None:
    """Check mmap structure without rereading every cached numeric value."""

    metadata = bundle.metadata
    outputs = _manifest_output_records(metadata)
    rows = bundle.molecule_rows
    n_molecules = len(rows)
    if metadata.get("molecule_count") != n_molecules:
        raise CacheValidationError("cache molecule_count does not match CSV")
    if [row["graph_idx"] for row in rows] != list(range(n_molecules)):
        raise CacheValidationError("molecules.csv graph_idx must be ordered from zero")

    offsets = bundle.offsets
    if offsets.dtype != np.dtype(np.int64) or offsets.shape != (n_molecules + 1,):
        raise CacheValidationError("offsets.npy must be int64 with length N+1")
    if offsets[0] != 0 or np.any(np.diff(offsets) <= 0):
        raise CacheValidationError("offsets.npy must be strictly increasing from zero")
    n_atoms = int(offsets[-1])
    if metadata.get("atom_count") != n_atoms:
        raise CacheValidationError("cache atom_count does not match offsets")

    embedding_dim = metadata.get("embedding_dim")
    if (
        bundle.values.dtype != np.dtype(np.float32)
        or bundle.values.shape != (n_atoms, embedding_dim)
    ):
        raise CacheValidationError(
            "representations.npy shape or dtype does not match cache metadata"
        )
    one_dimensional = {
        "atomic_numbers.npy": bundle.atomic_numbers,
        "encoder_atom_index.npy": bundle.encoder_atom_index,
        "graph_atom_index.npy": bundle.graph_atom_index,
    }
    for filename, array in one_dimensional.items():
        if array.ndim != 1 or len(array) != n_atoms:
            raise CacheValidationError(f"{filename} must have atom_count entries")
        if not np.issubdtype(array.dtype, np.integer):
            raise CacheValidationError(f"{filename} must contain integers")

    coordinate_mode = metadata.get("coordinate_mode")
    if coordinate_mode == "none":
        if bundle.coordinates is not None:
            raise CacheValidationError("coordinate-free cache loaded coordinates")
    elif coordinate_mode == "cartesian_3d":
        if (
            bundle.coordinates is None
            or bundle.coordinates.dtype != np.dtype(np.float32)
            or bundle.coordinates.shape != (n_atoms, 3)
        ):
            raise CacheValidationError(
                "coordinates.npy must be float32 with shape [atom_count, 3]"
            )
    else:
        raise CacheValidationError("cache coordinate_mode is invalid")

    arrays = {
        "representations.npy": bundle.values,
        "offsets.npy": offsets,
        **one_dimensional,
    }
    if bundle.coordinates is not None:
        arrays[COORDINATE_FILENAME] = bundle.coordinates
    for filename, array in arrays.items():
        record = outputs[filename]
        if tuple(record.get("shape", ())) != tuple(array.shape):
            raise CacheValidationError(f"cache shape mismatch for {filename}")
        if record.get("dtype") != str(array.dtype):
            raise CacheValidationError(f"cache dtype mismatch for {filename}")

    for index, row in enumerate(rows):
        start = int(offsets[index])
        stop = int(offsets[index + 1])
        if (
            row["n_atoms"] != stop - start
            or row["offset_start"] != start
            or row["offset_stop"] != stop
        ):
            raise CacheValidationError(
                f"molecules.csv offsets disagree at graph_idx {index}"
            )


def load_representation_cache(path: Path | str) -> RepresentationBundle:
    """Load a complete cache from the caller's explicit directory path."""

    root = Path(path)
    if not root.is_dir():
        raise CacheValidationError(f"cache directory does not exist: {root}")
    manifest = _read_manifest(root)
    _verify_file_set_and_hashes(root, manifest)
    rows = _load_molecule_rows(root / "molecules.csv")
    bundle = RepresentationBundle(
        root=root,
        manifest=_freeze_metadata(manifest),
        molecule_rows=tuple(_freeze_metadata(row) for row in rows),
        values=_load_array(root, "representations.npy"),
        offsets=_load_array(root, "offsets.npy"),
        atomic_numbers=_load_array(root, "atomic_numbers.npy"),
        encoder_atom_index=_load_array(root, "encoder_atom_index.npy"),
        graph_atom_index=_load_array(root, "graph_atom_index.npy"),
        coordinates=(
            _load_array(root, COORDINATE_FILENAME)
            if manifest.get("coordinate_mode") == "cartesian_3d"
            else None
        ),
    )
    _validate_loaded_bundle(bundle)
    return bundle


def open_representation_cache(path: Path | str) -> RepresentationBundle:
    """Memory-map a completed cache for training without whole-file rereads."""

    root = Path(path)
    if not root.is_dir():
        raise CacheValidationError(f"cache directory does not exist: {root}")
    metadata = _read_manifest(root)
    rows = _load_molecule_rows(root / "molecules.csv")
    bundle = RepresentationBundle(
        root=root,
        manifest=_freeze_metadata(metadata),
        molecule_rows=tuple(_freeze_metadata(row) for row in rows),
        values=_load_array(root, "representations.npy"),
        offsets=_load_array(root, "offsets.npy"),
        atomic_numbers=_load_array(root, "atomic_numbers.npy"),
        encoder_atom_index=_load_array(root, "encoder_atom_index.npy"),
        graph_atom_index=_load_array(root, "graph_atom_index.npy"),
        coordinates=(
            _load_array(root, COORDINATE_FILENAME)
            if metadata.get("coordinate_mode") == "cartesian_3d"
            else None
        ),
    )
    _validate_open_bundle_structure(bundle)
    return bundle


def _validate_bundle_against_inputs(
    bundle: RepresentationBundle,
    *,
    cache_id: str,
    fingerprint: Mapping[str, Any],
    input_provenance: InputProvenance,
    molecules: Sequence[MoleculeInput],
) -> None:
    """Require the cache molecule order to match the declared graph inputs."""

    if bundle.manifest["cache_id"] != cache_id:
        raise CacheValidationError("cache_id does not match requested inputs")
    if _canonical_fingerprint(
        bundle.manifest["provider"]
    ) != _canonical_fingerprint(fingerprint):
        raise CacheValidationError("provider fingerprint does not match the cache")
    if bundle.manifest["input_provenance"] != input_provenance.as_dict():
        raise CacheValidationError("input provenance does not match the cache")
    if len(bundle.molecule_rows) != len(molecules):
        raise CacheValidationError("cache molecule count does not match inputs")
    for molecule, row in zip(molecules, bundle.molecule_rows):
        if (
            row["graph_idx"] != molecule.graph_idx
            or row["sample_id"] != molecule.sample_id
            or row["smiles"] != molecule.smiles
        ):
            raise CacheValidationError(
                f"molecule identity mismatch at graph_idx {molecule.graph_idx}"
            )
        cached = bundle.get(molecule.graph_idx)
        try:
            validate_representation_mapping(molecule, cached)
        except ValueError as exc:
            raise CacheValidationError(
                f"atomic-number mapping mismatch at graph_idx {molecule.graph_idx}"
            ) from exc


def build_or_reuse_cache(
    cache_root: Path | str,
    provider: AtomRepresentationProvider,
    molecules: Sequence[MoleculeInput],
    *,
    input_provenance: InputProvenance,
    provider_scoped_root: bool = False,
) -> RepresentationBundle:
    """Reuse an exact valid cache or build it once; never repair it implicitly."""

    items = _validate_molecule_sequence(molecules)
    provider_name = getattr(provider, "name", None)
    if not isinstance(provider_name, str):
        raise TypeError("provider.name must be a string")
    provider_name = provider_name.strip().lower()
    if not _SAFE_PROVIDER_NAME.fullmatch(provider_name):
        raise ValueError("provider.name contains unsafe path characters")
    fingerprint = _canonical_fingerprint(provider.cache_fingerprint())
    _validate_singleton_extraction_policy(fingerprint)
    if fingerprint.get("provider") != provider_name:
        raise ValueError(
            "provider fingerprint must contain its normalized provider name"
        )
    cache_id = build_cache_id(
        fingerprint, items, input_provenance=input_provenance
    )
    provider_root = (
        Path(cache_root)
        if provider_scoped_root
        else Path(cache_root) / provider_name
    )
    target = provider_root

    if target.exists():
        bundle = load_representation_cache(target)
        _validate_bundle_against_inputs(
            bundle,
            cache_id=cache_id,
            fingerprint=fingerprint,
            input_provenance=input_provenance,
            molecules=items,
        )
        return bundle

    outputs: list[AtomRepresentation] = []
    for molecule in items:
        output = provider.extract_one(molecule)
        if not isinstance(output, AtomRepresentation):
            raise TypeError("provider.extract_one must return AtomRepresentation")
        validate_representation_mapping(molecule, output)
        outputs.append(output)
    fingerprint_dimension = fingerprint.get("embedding_dim")
    if (
        isinstance(fingerprint_dimension, bool)
        or not isinstance(fingerprint_dimension, int)
        or fingerprint_dimension < 1
    ):
        raise ValueError(
            "provider fingerprint must contain a positive integer embedding_dim"
        )
    observed_dimensions = {output.values.shape[1] for output in outputs}
    if observed_dimensions != {fingerprint_dimension}:
        raise ValueError(
            "provider output dimension does not match its cache fingerprint"
        )

    _write_cache(
        target,
        cache_id=cache_id,
        fingerprint=fingerprint,
        input_provenance=input_provenance,
        molecules=items,
        outputs=outputs,
    )
    bundle = load_representation_cache(target)
    _validate_bundle_against_inputs(
        bundle,
        cache_id=cache_id,
        fingerprint=fingerprint,
        input_provenance=input_provenance,
        molecules=items,
    )
    return bundle


EXPLICIT_HYDROGEN_EXCLUSION = (
    "explicit H/D/T is absent from Uni-Mol2 heavy-atom nodes"
)


@dataclass(frozen=True)
class ExcludedMolecule:
    source_row_index: int
    molecule_id: int
    sample_id: str
    smiles: str
    reason: str


@dataclass(frozen=True)
class LoadedMoleculeInputs:
    molecules: tuple[MoleculeInput, ...]
    input_provenance: InputProvenance
    source_row_indices: tuple[int, ...]
    exclusions: tuple[ExcludedMolecule, ...]


def _has_explicit_hydrogen(smiles: str) -> bool:
    parser = Chem.SmilesParserParams()
    parser.removeHs = False
    molecule = Chem.MolFromSmiles(smiles, parser)
    if molecule is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")
    return any(atom.GetAtomicNum() == 1 for atom in molecule.GetAtoms())


def _resolve_graph_file_template(
    graph_directory: Path,
    graph_file_template: str | None,
) -> str:
    """Use an explicit template or the graph manifest's declared template."""

    if graph_file_template is not None:
        template = str(graph_file_template)
    else:
        manifest_path = graph_directory / "manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            template = str(
                manifest.get(
                    "graph_file_template",
                    "graph_{molecule_id:03d}.pt",
                )
            )
        else:
            template = "graph_{molecule_id:03d}.pt"
    try:
        template.format(molecule_id=1)
    except (KeyError, ValueError) as exc:
        raise ValueError(
            "graph_file_template must contain a valid {molecule_id} field"
        ) from exc
    return template


def load_molecule_inputs_with_provenance(
    dataset_csv: Path | str,
    graph_root: Path | str,
    *,
    dataset_id: str | None = None,
    molecule_id_column: str = "index",
    sample_id_column: str = "index",
    smiles_column: str = "Smiles",
    graph_file_template: str | None = None,
    exclude_explicit_hydrogens: bool = True,
    additional_exclusions: Mapping[int, str] | None = None,
) -> LoadedMoleculeInputs:
    """Read, hash, and validate every included input without a second file pass."""

    dataset_path = Path(dataset_csv)
    graph_directory = Path(graph_root)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset CSV does not exist: {dataset_path}")
    if not graph_directory.is_dir():
        raise NotADirectoryError(f"Graph directory does not exist: {graph_directory}")
    resolved_graph_file_template = _resolve_graph_file_template(
        graph_directory,
        graph_file_template,
    )
    dataset_bytes = dataset_path.read_bytes()
    frame = pd.read_csv(io.BytesIO(dataset_bytes))
    required = {molecule_id_column, sample_id_column, smiles_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"Dataset CSV is missing columns: {missing}")
    identifiers = frame[molecule_id_column]
    if identifiers.isna().any() or identifiers.duplicated().any():
        raise ValueError("Dataset molecule identifiers must be non-null and unique")
    declared_exclusions = {
        int(source_row_index): str(reason)
        for source_row_index, reason in (
            {} if additional_exclusions is None else additional_exclusions
        ).items()
    }
    invalid_exclusion_rows = sorted(
        source_row_index
        for source_row_index in declared_exclusions
        if source_row_index < 0 or source_row_index >= len(frame)
    )
    if invalid_exclusion_rows:
        raise IndexError(
            "additional_exclusions contains out-of-range source rows: "
            f"{invalid_exclusion_rows}"
        )
    if any(not reason.strip() for reason in declared_exclusions.values()):
        raise ValueError("additional_exclusions reasons must be non-empty")

    molecules: list[MoleculeInput] = []
    source_row_indices: list[int] = []
    exclusions: list[ExcludedMolecule] = []
    graph_records: list[dict[str, object]] = []
    for source_row_index, row in frame.iterrows():
        molecule_id = row[molecule_id_column]
        sample_id = str(row[sample_id_column])
        numeric_id = exact_integer(
            molecule_id,
            name=f"dataset row {source_row_index} molecule id",
        )
        smiles = str(row[smiles_column])
        if source_row_index in declared_exclusions:
            exclusions.append(
                ExcludedMolecule(
                    source_row_index=int(source_row_index),
                    molecule_id=numeric_id,
                    sample_id=sample_id,
                    smiles=smiles,
                    reason=declared_exclusions[source_row_index],
                )
            )
            continue
        has_explicit_hydrogen = _has_explicit_hydrogen(smiles)
        if has_explicit_hydrogen and exclude_explicit_hydrogens:
            exclusions.append(
                ExcludedMolecule(
                    source_row_index=int(source_row_index),
                    molecule_id=numeric_id,
                    sample_id=sample_id,
                    smiles=smiles,
                    reason=EXPLICIT_HYDROGEN_EXCLUSION,
                )
            )
            continue
        if has_explicit_hydrogen:
            raise ValueError(
                "Dataset contains explicit H/D/T; active Uni-Mol2 extraction "
                "requires a recorded exclusion policy"
            )

        filename = resolved_graph_file_template.format(
            molecule_id=numeric_id
        )
        graph_path = graph_directory / filename
        if not graph_path.is_file():
            raise FileNotFoundError(
                f"Missing graph for molecule {molecule_id}: {graph_path}"
            )
        graph_bytes = graph_path.read_bytes()
        graph = load_pyg_data(io.BytesIO(graph_bytes), map_location="cpu")
        if exact_integer(
            getattr(graph, "idx", -1),
            name=f"{graph_path} graph.idx",
        ) != numeric_id:
            raise ValueError(f"Graph molecule id mismatch: {graph_path}")
        if exact_integer(
            getattr(graph, "graph_idx", -1),
            name=f"{graph_path} graph.graph_idx",
        ) != source_row_index:
            raise ValueError(f"Graph source row mismatch: {graph_path}")
        if str(getattr(graph, "sample_id", "")) != sample_id:
            raise ValueError(f"Graph sample id mismatch: {graph_path}")
        if str(getattr(graph, "smiles", "")) != smiles:
            raise ValueError(f"Graph SMILES mismatch: {graph_path}")
        mapping = getattr(graph, "smiles_to_graph_atom_index", None)
        if mapping is None:
            raise ValueError(f"Graph atom mapping is missing: {graph_path}")

        cache_graph_idx = len(molecules)
        molecules.append(
            molecule_input_from_graph(
                graph_idx=cache_graph_idx,
                sample_id=sample_id,
                smiles=smiles,
                graph=graph,
                smiles_to_graph_atom_index=mapping,
            )
        )
        source_row_indices.append(int(source_row_index))
        graph_records.append(
            {
                "cache_graph_idx": cache_graph_idx,
                "source_row_index": int(source_row_index),
                "molecule_id": numeric_id,
                "filename": filename,
                "sha256": hashlib.sha256(graph_bytes).hexdigest(),
            }
        )
    if not molecules:
        raise ValueError("Dataset contains no eligible molecules")

    canonical_graph_manifest = json.dumps(
        graph_records,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    provenance = InputProvenance(
        dataset_id=(
            (
                f"{dataset_path.stem}__declared_eligibility_excluded"
                if declared_exclusions
                else f"{dataset_path.stem}__explicit_hydrogen_excluded"
            )
            if dataset_id is None
            else dataset_id
        ),
        dataset_sha256=hashlib.sha256(dataset_bytes).hexdigest(),
        graph_manifest_sha256=hashlib.sha256(
            canonical_graph_manifest
        ).hexdigest(),
        graph_count=len(molecules),
    )
    return LoadedMoleculeInputs(
        molecules=tuple(molecules),
        input_provenance=provenance,
        source_row_indices=tuple(source_row_indices),
        exclusions=tuple(exclusions),
    )
