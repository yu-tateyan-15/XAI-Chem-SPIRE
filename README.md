# XAI-Chem-SPIRE

**Surrogate-based Physicochemical Interpretation of Regularized Evidence**

SPIRE explains molecular property predictions in physicochemical terms. A graph neural network assigns a contribution to each atom; chemical regularization encourages consistent contributions for shared structures. A descriptor-based surrogate then approximates these contributions to relate prediction evidence to properties such as electronic character and molecular flexibility.

Developed and tested using **Google Colab** and **macOS**. Dependency versions are listed in [requirements.txt](requirements.txt).

<p align="center">
  <img src="assets/graphical_abstract.png" width="480" alt="XAI in Chemistry: atom-level evidence to physicochemical interpretation">
</p>

## Atom contributions and their interpretation

An **atom contribution (AC)** is an atom's signed contribution to the model prediction. ACs sum to the prediction together with a separate affine baseline. Red denotes positive contributions and blue negative contributions. Both examples use the same raw color scale (±0.35 pIC50; values beyond this range saturate). Map titles show **predicted** pIC50.

### Dapagliflozin vs. 1214

Experimental pIC50: **8.89 vs. 5.05** (difference: 3.84).

![Raw atom contributions for dapagliflozin and 1214, without atom indices](assets/ac_dapa_1214.png)

**Descriptor-based reconstruction** approximates the learned ACs with atom-level physicochemical descriptors. Below, signed effects describe each descriptor's contribution to the reconstructed changed-region difference (**analog − dapagliflozin**); absolute shares show relative magnitudes, not percentages of experimental activity.

![Descriptor effects and absolute shares for dapagliflozin to 1214](assets/descriptor_effect_dapa_1214.png)

### Dapagliflozin vs. 1244

Experimental pIC50: **8.89 vs. 4.59** (difference: 4.30). This example is a two-cut structural comparison rather than a strict single-cut MMP.

![Raw atom contributions for dapagliflozin and 1244, without atom indices](assets/ac_dapa_1244.png)

![Descriptor effects and absolute shares for dapagliflozin to 1244](assets/descriptor_effect_dapa_1244.png)

*Existing regularized SGLT2 reference results (fold 0). Maps and surrogate effects describe model evidence, not experimentally measured atomic effects or causal mechanisms. Each descriptor-effect panel retains its original labeled axis range.*

## SGLT2 case study

| Executed notebook | Purpose | Data source |
|---|---|---|
| [Training R](notebooks_for_sglt2/output/executed_notebooks/01_training_R_executed.ipynb) · [U](notebooks_for_sglt2/output/executed_notebooks/01_training_U_executed.ipynb) | SGLT2 model training and AC regularization | [ChEMBL 37](https://www.ebi.ac.uk/chembl/), CHEMBL3884 |
| [Reference reconstruction](notebooks_for_sglt2/output/executed_notebooks/02_reconstruction_executed.ipynb) | In-sample reconstruction of reference R ACs | [This study](sglt2_data/README.md): reference ACs and computed descriptors |

## Additional examples

The same workflow is also illustrated on other molecular prediction tasks.

| Executed notebook | Purpose | Data source |
|---|---|---|
| [FreeSolv](test_sample/output/executed_notebooks/freesolv_executed.ipynb) | Hydration free energy | [FreeSolv v0.52](https://github.com/MobleyLab/FreeSolv) |
| [LD50](test_sample/output/executed_notebooks/ld50_toxicity_executed.ipynb) | Acute oral toxicity | [TDC / Zhu et al.](https://tdcommons.ai/single_pred_tasks/tox/#acute-toxicity-ld50) |
| [Lipophilicity](test_sample/output/executed_notebooks/lipophilicity_executed.ipynb) | Lipophilicity (logD) | [TDC / AstraZeneca](https://tdcommons.ai/single_pred_tasks/adme/#lipophilicity-astrazeneca) |
| [Caco-2](test_sample/output/executed_notebooks/caco2_executed.ipynb) | Intestinal permeability | [TDC / Wang et al.](https://tdcommons.ai/single_pred_tasks/adme/#caco-2-cell-effective-permeability-wang-et-al) |
| [Mayr / DCM](test_sample/output/executed_notebooks/mayr_dcm_executed.ipynb) | Nucleophilicity | [BNNLab / Mayr](https://github.com/BNNLab/Nucleophilicity) |

[Data sources and licenses](sglt2_data/README.md)


## Citation

If you use SPIRE or this code in your research, please cite the accompanying manuscript:

> Yusuke Tateishi, Junwei Shen, Masayoshi Aritsugi, and Shinichiro Nakamura. *Physicochemical Interpretation of Atom Contributions to Molecular Property Predictions*. Unpublished manuscript.

DOI: *** (to be assigned).
