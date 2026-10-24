# XAI-Chem-SPIRE

**Surrogate-based Physicochemical Interpretation of Regularized Evidence**

Atom-contribution regularization and physicochemical reconstruction for molecular prediction.

<p align="center">
  <img src="assets/graphical_abstract.png" width="480" alt="XAI in Chemistry: atom-level evidence to physicochemical interpretation">
</p>

## Setup

Python 3.11+. Run from the repository root.

```bash
python -m pip install -r requirements.txt
export XAI_CHEM_ROOT="$PWD"
```

Training also requires the pretrained CheMeleon encoder:

```bash
mkdir -p checkpoints
curl -L --fail https://zenodo.org/records/15460715/files/chemeleon_mp.pt \
  -o checkpoints/chemeleon_mp.pt
```

## Notebooks

Open a notebook with the Python environment above and run its cells in order.

| Notebook | Purpose |
|---|---|
| [01_training](notebooks_for_sglt2/01_training.ipynb) | SGLT2 training with (R) or without (U) AC regularization |
| [02_reconstruction](notebooks_for_sglt2/02_reconstruction.ipynb) | In-sample descriptor reconstruction; runs independently using bundled ACs, without encoder weights |
| [test_sample](test_sample/) | Independent FreeSolv, LD50, Lipophilicity, Caco-2 and Mayr/DCM examples; encoder required |

In Notebook 01, select `configs/sglt2_R.json` or `configs/sglt2_U.json` with `CONFIG_PATH`.
Results are saved under the corresponding notebook directory's `output/`.

[Data sources and licenses](sglt2_data/README.md) · [Recorded dependency versions](requirements-tested.txt)
