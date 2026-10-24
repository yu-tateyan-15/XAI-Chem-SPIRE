# XAI-Chem-SPIRE

**SPIRE — Surrogate-based Physicochemical Interpretation of Regularized Evidence**

SPIRE interprets a molecular prediction through additive **atom contributions (ACs)**
and local surrogate models fitted to physicochemical descriptors. This repository
provides the implementation, fixed SGLT2 inputs, and notebooks for training,
AC regularization, and descriptor-based reconstruction.

The workflow has three stages:

1. Extract frozen CheMeleon atom representations and train an additive graph neural network.
2. Compare an unregularized model (**U**) with a regularized model (**R**).
   R applies four constraints: common-core consistency and changed-region difference
   agreement within matched molecular pairs (**MMPs**), plus frequent-fragment
   positivity and consistency.
3. Fit a local Ridge surrogate to the changed-region atoms of each MMP and inspect
   how seven physicochemical descriptors reconstruct the model's ACs.

## Installation

Use Python **3.11 or later** in a dedicated environment. The recorded CPU environment
is Python **3.12.5 on macOS arm64**; its direct dependency versions are listed in
[requirements-tested.txt](requirements-tested.txt).

```bash
git clone https://github.com/yu-tateyan-15/XAI-Chem-SPIRE.git
cd XAI-Chem-SPIRE
python -m pip install -r requirements.txt
python -m ipykernel install --sys-prefix --name xai-chem --display-name "XAI-Chem-SPIRE"
export XAI_CHEM_ROOT="$PWD"
mkdir -p notebooks_for_sglt2/output/executed_notebooks
```

To use the recorded direct dependency versions, replace the installation command with
`python -m pip install -r requirements.txt -c requirements-tested.txt`.
Run the commands below from the repository root. For interactive work, select the
**XAI-Chem-SPIRE** kernel in your notebook editor and run cells in order.
`XAI_CHEM_ROOT` must point to the cloned repository; alternatively, set the notebook's
`LOCAL_PROJECT_ROOT` parameter to that absolute path.

Training and the optional examples also require the official pretrained
[CheMeleon encoder](https://zenodo.org/records/15460715):

```bash
mkdir -p checkpoints
curl -L --fail https://zenodo.org/records/15460715/files/chemeleon_mp.pt \
  -o checkpoints/chemeleon_mp.pt
```

The encoder and author-trained prediction-model weights are not bundled.
The reconstruction example below works without downloading an encoder or training a model.

## SGLT2 reproduction

### Reconstruct the supplied reference ACs

[02_reconstruction.ipynb](notebooks_for_sglt2/02_reconstruction.ipynb) is an independent
starting point. It uses the bundled R fold-0 AC table and fixed physical descriptors
to fit all **23,029 MMPs**, audit the descriptor-effect sums, and display reconstruction
fidelity. Notebook 01 does not need to run first.

```bash
python -m papermill notebooks_for_sglt2/02_reconstruction.ipynb \
  notebooks_for_sglt2/output/executed_notebooks/02_reconstruction_executed.ipynb \
  -k xai-chem
```

### Train the regularized and unregularized models

[01_training.ipynb](notebooks_for_sglt2/01_training.ipynb) inspects the fixed inputs,
builds graphs and atom representations, trains the configured folds, and displays
prediction metrics, learning curves, AC constraint values, and raw AC maps.

```bash
python -m papermill notebooks_for_sglt2/01_training.ipynb \
  notebooks_for_sglt2/output/executed_notebooks/01_training_R_executed.ipynb \
  -k xai-chem -p CONFIG_PATH configs/sglt2_R.json

python -m papermill notebooks_for_sglt2/01_training.ipynb \
  notebooks_for_sglt2/output/executed_notebooks/01_training_U_executed.ipynb \
  -k xai-chem -p CONFIG_PATH configs/sglt2_U.json
```

Both configurations use seed **42** and the fixed five-fold split of **1,259 molecules**.
The original GATv2 architecture, loss coefficients, and fold-specific gradient scales
are retained. R allows up to **150 epochs** and U up to **200**; both select the best
validation checkpoint with an early-stopping patience of **50 epochs**.
CPU is the default. `RUN_MODE="reuse"` inspects an existing matching run instead of retraining.

To reconstruct a newly trained model, set Notebook 02's `ATOM_CONTRIBUTIONS_PATH`,
for example with this additional Papermill argument:

```text
-p ATOM_CONTRIBUTIONS_PATH notebooks_for_sglt2/output/sglt2/R/fold_0/atom_contributions.csv
```

Use `U` in place of `R` for the unregularized model. Notebook 02 writes to one shared
reconstruction directory, so preserve the preceding outputs before switching teachers.

## Optional validation examples

These independent notebooks belong to `test_sample/`; they are separate from the
SGLT2 reproduction sequence. Each trains U and R, measures AC constraints, displays
prediction and AC figures, and reconstructs a few structurally selected MMPs.

| Notebook | Prediction target | Data availability |
|---|---|---|
| [FreeSolv](test_sample/freesolv.ipynb) | Hydration free energy, kcal/mol | Bundled v0.52 snapshot |
| [LD50 toxicity](test_sample/ld50_toxicity.ipynb) | log10[1/(LD50 in mol/kg)] | Bundled source; fixed 1,024-molecule sample |
| [Lipophilicity](test_sample/lipophilicity.ipynb) | logD at pH 7.4 | Downloaded by the notebook |
| [Caco-2](test_sample/caco2.ipynb) | log10(Papp / cm s⁻¹) | Downloaded by the notebook |
| [Mayr / DCM](test_sample/mayr_dcm.ipynb) | Nucleophilicity parameter N in dichloromethane | Downloaded by the notebook |

```bash
mkdir -p test_sample/output/executed_notebooks
python -m papermill test_sample/freesolv.ipynb \
  test_sample/output/executed_notebooks/freesolv_executed.ipynb -k xai-chem
```

Replace `freesolv` with another notebook stem to run that example. These examples use
a smaller two-layer GCN, 100 epochs per condition, one seeded random split, and seven
fast RDKit atom descriptors. Lipophilicity, Caco-2, and Mayr/DCM also report five
WISP explanation metrics on matched training and validation MMPs.

## Inputs and outputs

```text
configs/                 Original SGLT2 R/U settings
notebooks_for_sglt2/     Main training and reconstruction notebooks
sglt2_data/              Fixed molecules, folds, MMPs, descriptors, reference ACs
test_sample/             Independent examples and data-source notes
xai_chem/                Shared models, losses, reconstruction, metrics, and plotting
```

`sglt2_data/reference_R_ac.csv.gz` is a numerical reconstruction input, allowing
Notebook 02 to run without author-trained weights. The seven SGLT2 descriptors are
CREST displacement, Hessian RMSF, Fukui +/−, neutral charge squared, and OpenFF
Lennard–Jones epsilon/sigma. Their precomputed table is supplied; the physical
calculations that produced it are outside this distribution.

| Output | Location |
|---|---|
| SGLT2 models, predictions, and per-atom AC tables | `notebooks_for_sglt2/output/sglt2/{R,U}/fold_*/` |
| SGLT2 training figures | `notebooks_for_sglt2/output/sglt2/vis/` |
| Reconstruction tables and figures | `notebooks_for_sglt2/output/reconstruction/{csv,vis}/` |
| Example-specific results | `test_sample/output/<RUN_NAME>/` |
| Executed computational records | The corresponding `output/executed_notebooks/` |

Generated weights, caches, figures, and execution records stay in ignored output
directories. Figures use the shared project style in [plotting.py](xai_chem/plotting.py),
including Y–Y plots and raw AC maps with numerical atom tables and explicit color scales.

## Reproducibility and interpretation

- Training enables deterministic Torch algorithms. Reproducibility depends on the
  software, hardware, and numerical backend; changing environments can change trained
  parameters and reported digits even with the same seed.
- ACs are additive terms in a model prediction, with a separate affine baseline.
  They do not establish causal atomic effects. Separate SGLT2 R/U runs can use different
  map limits; compare their raw tables or explicitly share a scale.
- Ridge fidelity is **in-sample reconstruction** of the fitted MMP endpoints, not
  predictive performance on unseen atoms. MMPs can share molecules and are not
  independent replicates.
- The optional examples test this workflow under their declared settings. They use
  different descriptors from SGLT2 and do not constitute official scaffold benchmarks
  or demonstrate that one regularization weight is optimal across endpoints.

## Data sources and attribution

See [data sources and interpretation](sglt2_data/README.md) and the source notes under
[test_sample/input_data](test_sample/input_data). SGLT2 data derive from ChEMBL
(CC BY-SA 3.0); the bundled FreeSolv and LD50 sources are CC BY 4.0. The other examples
download their source files and retain the upstream terms.

Please acknowledge the original datasets and [CheMeleon](https://github.com/JacksonBurns/chemeleon)
when using them. The WISP metric definitions follow
[Janssen et al.](https://doi.org/10.1039/D5DD00398A) and the
[WISP implementation](https://github.com/kerjans/ml-XAI).
