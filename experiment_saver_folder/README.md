# experiment_saver

A lightweight, safe utility for **TensorFlow / Keras** that automatically saves training artifacts and evaluation outputs for **binary classification** experiments.

It helps you keep every run organized in a single folder (models, history, logs, ROC files), so you can plot and compare results later without rewriting code.

---

## Features

- **Easy callbacks**: auto-creates:
  - `CSVLogger` (epoch metrics → `metrics.csv`)
  - `ModelCheckpoint` (best model)
  - `EarlyStopping` (restore best weights)
- **Saves training history**: `history.json`
- **Saves models**:
  - `best_model.keras` (based on monitored metric)
  - `final_model.keras` (end of training)
- **Auto ROC evaluation** after training:
  - `roc_fpr.npy`, `roc_tpr.npy`, `roc_thresholds.npy`
  - `roc_auc.json`
- **Shape-robust**:
  - Supports `sigmoid` outputs: `(batch,)` or `(batch, 1)`
  - Supports `softmax` outputs for binary: `(batch, 2)` (uses positive-class prob)
- **Safe JSON config saving**: best-effort conversion of non-serializable values to strings

---

## What gets saved

Inside your `run_dir/`:

- `metrics.csv` — epoch-by-epoch logs (loss, accuracy, auc, val_loss, …)
- `history.json` — same metrics as a JSON dictionary
- `best_model.keras` — best checkpoint (based on `monitor`)
- `final_model.keras` — final model at end of training
- `roc_fpr.npy` — ROC false positive rate
- `roc_tpr.npy` — ROC true positive rate
- `roc_thresholds.npy` — ROC thresholds
- `roc_auc.json` — ROC AUC value
- `config.json` — experiment metadata (optional but recommended)
- `classes.json` — class names (optional but useful)

---

## Installation

### Option A — Install from source

```bash
git clone https://github.com/AhmedAbdAlKareem1/experiment_saver.git
cd experiment_saver/experiment_saver_folder
pip install .
