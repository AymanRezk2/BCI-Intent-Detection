# Source Code

This folder contains the core source code of the project:

- `data_loader.py` → Load and split datasets (PhysioNet, BCI Competition, BNCI).
- `preprocessing.py` → Signal cleaning (bandpass filter, ICA, normalization).
- `features.py` → Feature extraction (spectrograms, covariance matrices).
- `models/` → Implementations of deep learning and signal processing models:
  - `eegnet.py` → EEGNet CNN implementation.
  - `transformer.py` → Transformer-based classifier for EEG.
  - `riemannian.py` → Riemannian geometry + tangent space methods.
- `train.py` → Training pipeline (loss, optimizer, metrics).
- `evaluate.py` → Model evaluation (accuracy, confusion matrix, etc.).

👉 Import these modules inside notebooks or scripts for reproducibility.
