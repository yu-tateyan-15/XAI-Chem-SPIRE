"""WISP's five metrics on SPIRE's corrected MMPs and raw atom contributions.

Definitions follow Modeling/src/analysis/wisp_metrics_official.py and
Janssen et al., DOI:10.1039/D5DD00398A, https://github.com/kerjans/ml-XAI.
WISP reports squared Pearson correlation, not regression R-squared.
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd


def build_wisp_pair_table(mmp, predictions, atoms, *, split):
    """Compute i-minus-j gaps on pairs with both endpoints in one split.

    Caller supplies additive predictions and corrected, disjoint MMP regions.
    One population SD over all train/validation raw atoms of this model is used
    for both splits, matching the project's fold-wide WISP evaluation.
    Normalization is for metrics only; plotted and saved raw ACs are untouched.
    """
    atom_std = float(np.std(atoms["atom_contribution_raw"], ddof=0))
    if not np.isfinite(atom_std) or atom_std <= 0:
        raise ValueError("WISP requires a finite positive atom population SD")
    pred = predictions.query("split == @split").set_index("source_row_index")
    eligible = mmp[mmp.i_id.isin(pred.index) & mmp.j_id.isin(pred.index)]
    values = {
        int(row): group.set_index("smiles_atom_index")["atom_contribution_raw"]
        for row, group in atoms.groupby("source_row_index")
    }
    rows = []
    for pair in eligible.itertuples(index=False):
        left, right = values[pair.i_id], values[pair.j_id]
        a, b = pred.loc[pair.i_id], pred.loc[pair.j_id]
        observed_gap = float(a.target_raw - b.target_raw)
        # Exported tensor targets are float32; keep the original measured gap
        # while allowing the rounding of its two target operands.
        target_tolerance = 1e-6 + np.finfo(np.float32).eps * (abs(a.target_raw) + abs(b.target_raw))
        if not np.isclose(pair.dy, observed_gap, atol=target_tolerance, rtol=0):
            raise ValueError("MMP target difference does not match observations")
        whole = float(left.sum() - right.sum())
        variable = float(left.loc[json.loads(pair.diff_i_json)].sum()
                         - right.loc[json.loads(pair.diff_j_json)].sum())
        constant = float(left.loc[json.loads(pair.core_i_json)].sum()
                         - right.loc[json.loads(pair.core_j_json)].sum())
        neighbor = float(left.loc[json.loads(pair.attachment_i_json)].sum()
                         - right.loc[json.loads(pair.attachment_j_json)].sum())
        prediction = float(a.prediction_raw - b.prediction_raw)
        rows.append({
            "pair_id": pair.pair_id, "split": split,
            "i_id": pair.i_id, "j_id": pair.j_id,
            # Keep the original precision of the supplied measured target gap.
            "prediction_gap": prediction, "observed_gap": float(pair.dy),
            "whole_attribution_gap_raw": whole,
            "variable_attribution_gap_raw": variable,
            "constant_attribution_gap_raw": constant,
            "variable_plus_neighbor_attribution_gap_raw": variable + neighbor,
            "whole_attribution_gap": whole / atom_std,
            "variable_attribution_gap": variable / atom_std,
            "constant_attribution_gap": constant / atom_std,
            "variable_plus_neighbor_attribution_gap": (variable + neighbor) / atom_std,
            "additivity_gap_raw": whole - prediction,
        })
    return pd.DataFrame(rows), atom_std


def _pearson_r_squared(left, right):
    """Keep an undefined correlation explicit rather than reporting zero."""
    value = float(np.corrcoef(left, right)[0, 1] ** 2)
    if not np.isfinite(value):
        raise ValueError("WISP Pearson correlation is undefined")
    return value


def summarize_official_wisp_metrics(pairs):
    """Return the official five metrics, without project-specific extensions."""
    columns = ["prediction_gap", "observed_gap", "whole_attribution_gap",
               "variable_attribution_gap", "constant_attribution_gap"]
    if len(pairs) < 2 or not np.isfinite(pairs[columns].to_numpy()).all():
        raise ValueError("WISP requires at least two finite pairs")
    return {
        "n_pairs_evaluated": len(pairs),
        "whole_to_prediction_pearson_r_squared": _pearson_r_squared(
            pairs.prediction_gap, pairs.whole_attribution_gap),
        "variable_to_prediction_pearson_r_squared": _pearson_r_squared(
            pairs.prediction_gap, pairs.variable_attribution_gap),
        "whole_to_observed_pearson_r_squared": _pearson_r_squared(
            pairs.observed_gap, pairs.whole_attribution_gap),
        "constant_attribution_gap_std": float(np.std(pairs.constant_attribution_gap, ddof=0)),
        # Official direction classification uses > 0; a zero is non-positive.
        "variable_to_prediction_direction_accuracy": float(np.mean(
            (pairs.variable_attribution_gap > 0) == (pairs.prediction_gap > 0))),
    }
