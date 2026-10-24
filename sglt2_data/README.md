# Data and interpretation

- **SGLT2:** ChEMBL 37, target CHEMBL3884, acquired 2026-07-29. The fixed input has
  1,259 molecules, activity-balanced five-fold assignments (seed 42), two distinct
  23,029-pair MMP tables, and seven physical descriptors for 39,530 heavy atoms.
  Preserve [ChEMBL attribution](https://www.ebi.ac.uk/chembl/) and its CC BY-SA 3.0 notice.
- The physical descriptors are weighted CREST displacement, Hessian RMSF, Fukui+/−,
  neutral charge squared, and OpenFF LJ epsilon/sigma. This distribution reconstructs
  from their fixed numerical table; it does not recompute the physical calculations.
- The R fold-0 AC table supports reconstruction without author-trained weights.
  `notebooks_for_sglt2/01_training.ipynb` retains the selected model architecture, objective coefficients and
  fixed per-fold gradient scales. Exact retraining values may depend on the runtime.
- **FreeSolv (optional validation example):** the upstream v0.52 snapshot has 642 molecules. Keep its bundled
  [README](../test_sample/input_data/freesolv/README.md), [license](../test_sample/input_data/freesolv/LICENSE), and
  [dataset citation](https://github.com/MobleyLab/FreeSolv). The target is hydration
  free energy in kcal/mol, without sign reversal or logarithm; fold 0 uses seed 42.
- **LD50 (optional toxicity regression example):** see [source and selection](../test_sample/input_data/ld50/README.md).
  The fixed 1,024-molecule sample uses the supplied log10[1/(mol/kg)] target,
  sample seed 42 and a random split seed 42; it is not the TDC scaffold benchmark.
- **CheMeleon:** obtain the encoder from the official [Zenodo record](https://zenodo.org/records/15460715)
  and cite [CheMeleon](https://github.com/JacksonBurns/chemeleon). Encoding is per molecule
  and retains 2,048-dimensional atom states. Encoder weights are not redistributed.

ACs are additive prediction terms with a separate affine baseline. Ridge fidelity
is in-sample reconstruction, not a causal atomic effect. FreeSolv and LD50 use seven separate
RDKit descriptors and three MMPs selected by structure size before fitting.
These examples verify the workflow on other endpoints; they do not validate toxicity classification.
