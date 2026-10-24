# Mayr nucleophilicity in DCM

Source: [BNNLab_Mayr_GAS_SET_DCM](https://github.com/BNNLab/Nucleophilicity).
[Original file](https://raw.githubusercontent.com/BNNLab/Nucleophilicity/main/GAS_SET_16_Descriptors.csv), accessed 2026-10-24.
Target: Mayr N (DCM); copied without an additional transform.
The 904-row source has 372 DCM observations. B/Si are retained; 19 Sn/Ge-containing
structures lack finite Gasteiger descriptors in this example. The resulting set has 353 molecules.
Data accompany “Predicting Solvent-Dependent Nucleophilicity Parameter with a Causal
Structure Property Relationship”; [archived release](https://doi.org/10.5281/zenodo.4705840).
The notebook downloads the unchanged source to its `output/` run directory,
filters structures, deduplicates canonical SMILES, and uses all eligible rows.
Keep dichloromethane only; ignore all supplied molecular descriptors.
Source data are downloaded rather than redistributed; no new data license is asserted.
