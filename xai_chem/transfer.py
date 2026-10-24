"""Transfer for notebook-based atom-contribution experiments."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from io import StringIO
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, rdPartialCharges


RDKit_DESCRIPTORS = (
    "gasteiger_charge", "gasteiger_charge_square", "crippen_logp",
    "crippen_molar_refractivity", "tpsa_A2", "atomic_mass", "total_hydrogens",
)


def load_freesolv_snapshot(path: Path) -> pd.DataFrame:
    """Read experimental hydration energies in kcal/mol without changing sign."""
    names = ["cid", "Smiles", "name", "hydration_kcal_mol", "experimental_uncertainty",
             "calculated", "calculated_uncertainty", "experimental_reference",
             "calculated_reference", "notes"]
    # Only whole header lines are comments: '#' inside a SMILES is a triple bond.
    text = "".join(line for line in path.read_text().splitlines(keepends=True) if not line.startswith("#"))
    frame = pd.read_csv(StringIO(text), sep=";", names=names, skipinitialspace=True)
    for key in ("cid", "Smiles", "name"):
        frame[key] = frame[key].str.strip()
    # Canonicalization changes traversal only; every graph is rebuilt in this order.
    frame["Smiles"] = frame["Smiles"].map(
        lambda value: Chem.MolToSmiles(Chem.MolFromSmiles(value), isomericSmiles=True)
    )
    frame.insert(0, "index", np.arange(1, len(frame) + 1))
    return frame


def build_regression_mmp(
    frame: pd.DataFrame, graph_root: Path, output_csv: Path, *,
    target_column: str = "hydration_kcal_mol", dataset_group: str = "freesolv_dataset_group",
) -> pd.DataFrame:
    """Use the published MMP chemistry on one explicit regression endpoint/group."""
    from .mmp import build_dataset_entries
    from .mmp import build_mmp_pairs
    from .mmp import load_graph_meta
    from .mmp import save_pair_table_csv
    from .mmp import DEFAULT_RDMMPA_PATTERN
    from .mmp import MMPSubsetConfig
    from .mmp import select_pair_records
    from .mmp import _correct_records

    # The historical builder's internal pIC50 slot stores the explicit generic target.
    # No logarithm or sign conversion is applied. The exported columns are target_i/j.
    builder_frame = frame.assign(
        pIC50=frame[target_column], document_chembl_ids=dataset_group,
    )
    graph_meta = load_graph_meta(graph_root / "manifest.json")
    entries, _ = build_dataset_entries(
        builder_frame.to_dict("records"), graph_meta, morgan_radius=2, morgan_nbits=2048,
    )
    pairs, _, _, _ = build_mmp_pairs(
        entries, pattern=DEFAULT_RDMMPA_PATTERN, min_cuts=1, max_cuts=1,
        max_cut_bonds=20, min_core_atoms=3,
    )
    # Preserve the original filters: similarity >= 0.1 and |delta target| >= 0.1
    # in native endpoint units; the historical config field is named pIC50.
    selected, _, _ = select_pair_records(pairs, MMPSubsetConfig())
    corrected, _ = _correct_records(
        selected, dataset_rows=builder_frame.to_dict("records"), graph_meta=graph_meta,
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    save_pair_table_csv(output_csv, corrected)
    table = pd.read_csv(output_csv).rename(columns={"pIC50_i": "target_i", "pIC50_j": "target_j"})
    table.to_csv(output_csv, index=False)
    return table


def rdkit_atom_descriptors(frame: pd.DataFrame) -> pd.DataFrame:
    """Calculate seven interpretable 2D descriptors independently of CheMeleon."""
    # NOTE: CONTRACT graph_id is the zero-based frame row; preserve frame order.
    # atom_idx is the SMILES index; map graph-indexed tables via smiles_to_graph_atom_index.
    records = []
    for row_idx, row in frame.iterrows():
        mol = Chem.MolFromSmiles(row["Smiles"])
        rdPartialCharges.ComputeGasteigerCharges(mol)
        crippen = rdMolDescriptors._CalcCrippenContribs(mol)
        tpsa = rdMolDescriptors._CalcTPSAContribs(mol)
        for atom in mol.GetAtoms():
            atom_idx = atom.GetIdx()
            charge = float(atom.GetProp("_GasteigerCharge"))
            records.append({
                "graph_id": row_idx, "molecule_id": str(row["cid"]), "atom_idx": atom_idx,
                "atom_symbol": atom.GetSymbol(), "smiles": row["Smiles"],
                "gasteiger_charge": charge, "gasteiger_charge_square": charge ** 2,
                "crippen_logp": crippen[atom_idx][0],
                "crippen_molar_refractivity": crippen[atom_idx][1],
                "tpsa_A2": tpsa[atom_idx], "atomic_mass": atom.GetMass(),
                "total_hydrogens": atom.GetTotalNumHs(),
            })
    result = pd.DataFrame(records)
    if not np.isfinite(result[list(RDKit_DESCRIPTORS)].to_numpy()).all():
        raise ValueError("RDKit descriptors contain non-finite values")
    return result


def measure_ac_constraints(results, raw_dataset, mmp_path: Path, *, split: str) -> dict:
    """Measure identical four-component losses on either model's saved raw ACs."""
    from .training import load_fold_checkpoint
    from .mmp_loss import load_mmp_pair_records
    from .mmp_loss import compute_mmp_pair_loss
    from .fragment_loss import build_fragment_frequency_spec
    from .fragment_loss import compute_fragment_frequency_loss
    from .mmp_loss import load_mmp_atom_mapping_sources
    from .mmp_loss import reconstruct_mmp_graph_records

    loaded = load_fold_checkpoint(results.checkpoint, device="cpu")
    dataset = raw_dataset.with_scaler(loaded.scaler)
    selected = results.atom_contributions.query("split == @split")
    contributions = {
        int(row): torch.tensor(atoms.sort_values("graph_atom_index")["atom_contribution_scaled"].to_numpy(), dtype=torch.float32)
        for row, atoms in selected.groupby("source_row_index")
    }
    rows = set(contributions)
    records = load_mmp_pair_records(mmp_path)
    mapping = load_mmp_atom_mapping_sources(
        pd.read_csv(mmp_path), records,
        expected_mapping_method=results.training_config["expected_mmp_core_mapping_method"],
    )
    pairs = tuple(record for record in records if record.i_id in rows and record.j_id in rows)
    graph_pairs = reconstruct_mmp_graph_records(dataset, pairs, mapping).records
    mmp = compute_mmp_pair_loss(contributions, graph_pairs, target_alpha=loaded.scaler.alpha,
                               core_weight=1.0, difference_weight=1.0, huber_beta=0.25)
    train_spec = build_fragment_frequency_spec(dataset, loaded.metadata["train_rows"],
                                              min_heavy_atoms=1, min_count=3, min_frequency=0.1)
    # Validation inherits training frequencies and margin; only atom occurrences change.
    if split == "validation":
        from .fragment_loss import FragmentOccurrence
        from .fragment_loss import brics_fragments_for_smiles
        occurrences = []
        for row in sorted(rows):
            graph = dataset[dataset.position_for_source_row(row)]
            mapping_array = graph.smiles_to_graph_atom_index.numpy()
            for smiles, atom_indices in brics_fragments_for_smiles(graph.smiles, min_heavy_atoms=1):
                if smiles in train_spec.frequencies:
                    occurrences.append(FragmentOccurrence(row, smiles, tuple(int(mapping_array[i]) for i in atom_indices)))
        spec = replace(train_spec, occurrences=tuple(occurrences))
    else:
        spec = train_spec
    fragment = compute_fragment_frequency_loss(contributions, spec, positive_weight=1.0, consistency_weight=1.0)
    return {"split": split, "mmp_pairs": len(pairs), "core": float(mmp.core),
            "difference": float(mmp.difference), "positive": float(fragment.positive),
            "consistency": float(fragment.consistency)}
