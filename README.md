# XAI-Chem-SPIRE

**Surrogate-based Physicochemical Interpretation of Regularized Evidence**

Atom-contribution regularization and physicochemical reconstruction for molecular prediction.

<p align="center">
  <img src="assets/graphical_abstract.png" width="480" alt="XAI in Chemistry: atom-level evidence to physicochemical interpretation">
</p>

## Notebooks

| Notebook | Purpose |
|---|---|
| [01_training.ipynb](notebooks_for_sglt2/01_training.ipynb) | SGLT2 model training and AC regularization |
| [02_reconstruction.ipynb](notebooks_for_sglt2/02_reconstruction.ipynb) | In-sample reconstruction of SGLT2 ACs from physicochemical descriptors |
| [freesolv.ipynb](test_sample/freesolv.ipynb) | Hydration free-energy validation example |
| [ld50_toxicity.ipynb](test_sample/ld50_toxicity.ipynb) | Acute oral toxicity validation example |
| [lipophilicity.ipynb](test_sample/lipophilicity.ipynb) | Lipophilicity (logD) validation example |
| [caco2.ipynb](test_sample/caco2.ipynb) | Caco-2 permeability validation example |
| [mayr_dcm.ipynb](test_sample/mayr_dcm.ipynb) | Mayr nucleophilicity in DCM validation example |

[Data sources and licenses](sglt2_data/README.md)
