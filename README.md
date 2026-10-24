# XAI-Chem-SPIRE

**Surrogate-based Physicochemical Interpretation of Regularized Evidence**

Atom-contribution regularization and physicochemical reconstruction for molecular prediction.

<p align="center">
  <img src="assets/graphical_abstract.png" width="480" alt="XAI in Chemistry: atom-level evidence to physicochemical interpretation">
</p>

## Core study: SGLT2

| Executed notebook | Purpose | Data source |
|---|---|---|
| [Training R](notebooks_for_sglt2/output/executed_notebooks/01_training_R_executed.ipynb) · [U](notebooks_for_sglt2/output/executed_notebooks/01_training_U_executed.ipynb) | SGLT2 model training and AC regularization | [ChEMBL 37](https://www.ebi.ac.uk/chembl/), CHEMBL3884 |
| [Reference reconstruction](notebooks_for_sglt2/output/executed_notebooks/02_reconstruction_executed.ipynb) | In-sample reconstruction of reference R ACs | [This study](sglt2_data/README.md): reference ACs and computed descriptors |

## Independent test samples

| Executed notebook | Purpose | Data source |
|---|---|---|
| [FreeSolv](test_sample/output/executed_notebooks/freesolv_executed.ipynb) | Hydration free-energy validation | [FreeSolv v0.52](https://github.com/MobleyLab/FreeSolv) |
| [LD50](test_sample/output/executed_notebooks/ld50_toxicity_executed.ipynb) | Acute oral toxicity validation | [TDC / Zhu et al.](https://tdcommons.ai/single_pred_tasks/tox/#acute-toxicity-ld50) |
| [Lipophilicity](test_sample/output/executed_notebooks/lipophilicity_executed.ipynb) | logD validation | [TDC / AstraZeneca](https://tdcommons.ai/single_pred_tasks/adme/#lipophilicity-astrazeneca) |
| [Caco-2](test_sample/output/executed_notebooks/caco2_executed.ipynb) | Permeability validation | [TDC / Wang et al.](https://tdcommons.ai/single_pred_tasks/adme/#caco-2-cell-effective-permeability-wang-et-al) |
| [Mayr / DCM](test_sample/output/executed_notebooks/mayr_dcm_executed.ipynb) | Nucleophilicity validation | [BNNLab / Mayr](https://github.com/BNNLab/Nucleophilicity) |

[Data sources and licenses](sglt2_data/README.md)
