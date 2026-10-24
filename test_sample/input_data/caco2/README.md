# Caco-2

Source: [TDC_Caco2_Wang_4259569](https://tdcommons.ai/single_pred_tasks/adme/#caco-2-cell-effective-permeability-wang-et-al).
[Original file](https://dataverse.harvard.edu/api/access/datafile/4259569?format=original), accessed 2026-10-24.
Target: Caco-2 log10(Papp / cm s⁻¹); copied without an additional transform.
Wang et al., *Journal of Chemical Information and Modeling* 56 (2016), 763–773,
“ADME Properties Evaluation in Drug Discovery: Prediction of Caco-2 Cell Permeability
Using a Combination of NSGA-II and Boosting.” The original-format file has 910 rows,
rather than the 906 listed in the TDC overview; this notebook explicitly deduplicates it.
The notebook downloads the unchanged source to its `output/` run directory,
filters structures, deduplicates canonical SMILES, and uses all eligible rows.
The split is seeded random 80/20, not the official scaffold benchmark.
Source data are downloaded rather than redistributed; no new data license is asserted.
