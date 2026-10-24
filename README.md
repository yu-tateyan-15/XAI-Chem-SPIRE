# XAI-Chem-SPIRE

SPIRE: **Surrogate-based Physicochemical Interpretation of Regularized Evidence**.
Notebook reproduction of molecular training, AC regularization, and descriptor reconstruction.
Python 3.11+; run from the repository root. Set `RUN_MODE="reuse"` to inspect an existing configured run.
Training requires deterministic Torch kernels; repeatability is scoped to the same software and hardware.
Recorded CPU environment: Python 3.12.5 on macOS arm64; [exact dependency versions](requirements-tested.txt).

```bash
python -m pip install -r requirements.txt
export XAI_CHEM_ROOT="$PWD"
python -m ipykernel install --sys-prefix --name xai-chem
mkdir -p checkpoints notebooks_for_sglt2/output/executed_notebooks
curl -L --fail https://zenodo.org/records/15460715/files/chemeleon_mp.pt -o checkpoints/chemeleon_mp.pt
```

| Notebook | Purpose |
|---|---|
| [01_training](notebooks_for_sglt2/01_training.ipynb) | SGLT2 inputs → graphs → CheMeleon → R/U training and AC constraints |
| [02_reconstruction](notebooks_for_sglt2/02_reconstruction.ipynb) | Reconstruct ACs with the seven physical descriptors |

Notebook 01 uses the condition, folds and original budgets in `CONFIG_PATH` (`configs/sglt2_R.json` or `configs/sglt2_U.json`).
Notebook 02 accepts the bundled AC table or a newly trained model's `atom_contributions.csv`.
Optional examples: [FreeSolv](test_sample/freesolv.ipynb), [LD50](test_sample/ld50_toxicity.ipynb),
[Lipophilicity](test_sample/lipophilicity.ipynb), [Caco-2](test_sample/caco2.ipynb), and
[Mayr/DCM](test_sample/mayr_dcm.ipynb) test U/R training, AC constraints, and MMP reconstruction.
The three new examples also report the five WISP metrics on matched U/R pairs.
These use fast 2D descriptors and single random splits;
see the notebooks for their scope and limitations. Run from the repository root:

```bash
mkdir -p test_sample/output/executed_notebooks
python -m papermill test_sample/freesolv.ipynb test_sample/output/executed_notebooks/freesolv_executed.ipynb -k xai-chem
```

Data licenses: ChEMBL CC BY-SA 3.0; FreeSolv and LD50 CC BY 4.0; see [sources and notices](sglt2_data/README.md).
Main outputs: `notebooks_for_sglt2/output/`; validation outputs: `test_sample/output/`.
The new examples download their source data; acquisition notes are in `test_sample/input_data/`.

All notebooks use the project-standard Y–Y plots and raw AC maps from `xai_chem/plotting.py`.
