# LD50 toxicity example

Source: [TDC acute toxicity LD50](https://tdcommons.ai/single_pred_tasks/tox/#acute-toxicity-ld50),
Harvard Dataverse [file 4267146](https://dataverse.harvard.edu/api/access/datafile/4267146), retrieved 2026-10-24.
Zhu et al., *Chemical Research in Toxicology* 22 (2009), 1913–1921,
[doi:10.1021/tx900189p](https://doi.org/10.1021/tx900189p).
The upstream 7,385-row table is retained unchanged, with gzip compression.
[PyTDC states CC BY 4.0](https://pytdc.arcell.ai/datasets/tox/LD50_Zhu);
[license text](https://creativecommons.org/licenses/by/4.0/legalcode).

Target: log10[1/(LD50 in mol/kg)], copied as supplied; larger values indicate
lower lethal doses. The notebook visibly excludes invalid/disconnected structures
and atoms outside H/C/N/O/F/P/S/Cl/Br/I, retains the first canonical-SMILES occurrence,
and selects 1,024 rows with pandas random_state=42, independent of target values.
A seeded random 80/20 split tests execution; it is not the official scaffold benchmark.
No rows, seed, or parameters are selected by validation performance.
