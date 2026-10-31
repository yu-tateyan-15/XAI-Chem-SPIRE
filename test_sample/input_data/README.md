# Paper atom descriptors

Each `paper_atom_descriptors.csv` covers only the endpoints of the three displayed validation MMPs. The seven descriptors match the SGLT2 study: Fukui f⁺/f⁻ (net-charge differences), neutral charge squared, Hessian RMSF, Boltzmann-weighted CREST displacement, and OpenFF LJ ε/σ.

Calculations use GFN-FF/CREST final minima within 20 kcal/mol, AIMNet2-NSE optimized and Hessian-validated neutral geometries, harmonic RMSF at 300 K, conformer weights at 298.15 K, and OpenFF Sage 2.3.0. Each value is averaged over an explicit heavy atom and its bonded hydrogens; charge squaring precedes averaging. A single observed CREST minimum gives zero observed interconformer displacement.

`graph_id` is the zero-based curated dataset row and `atom_idx` follows the input SMILES heavy-atom order. The loader verifies exact SMILES, atom coverage and symbols. These descriptors are reconstruction inputs; the U/R training data and models are unchanged. Mayr examples were reselected within the same validation split for complete Sage assignments before ranking by experimental target difference. No missing descriptor is replaced with a 2D surrogate or a guessed value.

Computed on macOS with Python 3.11.15, PyTorch 2.12.0, AIMNet 0.2.0, ASE 3.28.0 and OpenFF Toolkit 0.18.0. The CREST 3.0.2 build passed an explicit 20 kcal/mol energy-window control before sampling.
