# experiment_saver

A lightweight, safe utility for **TensorFlow / Keras** that automatically saves training artifacts and evaluation outputs.

It keeps every run organized inside a single folder (models, logs, history, ROC files), so you can compare experiments later without rewriting boilerplate code.

---

## Highlights

- ✅ One-line callbacks: `CSVLogger`, `ModelCheckpoint`, `EarlyStopping`
- ✅ Saves **best** + **final** model
- ✅ Saves **history.json** + **metrics.csv**
- ✅ Automatically computes **ROC / AUC** after training
- ✅ Handles common output shapes safely:
  - Sigmoid: `(batch,)` or `(batch, 1)`
  - Softmax (binary): `(batch, 2)` (uses positive-class probability)
  - Softmax (multiclass): `(batch, C)`
- ✅ JSON-safe config saving (converts non-serializable values to strings)

---

## What gets saved

Inside your `run_dir/`:

| File | Description |
|------|------------|
| `metrics.csv` | Epoch-by-epoch logs (loss, accuracy, auc, val_loss, …) |
| `history.json` | Same history data as JSON |
| `best_model.keras` | Best checkpoint (based on `monitor`) |
| `final_model.keras` | Final model at end of training |
| `roc_fpr.npy` | ROC false positive rate |
| `roc_tpr.npy` | ROC true positive rate |
| `roc_thresholds.npy` | ROC thresholds |
| `roc_auc.json` | ROC AUC summary |
| `config.json` | Experiment metadata (optional but recommended) |
| `classes.json` | Class names (optional but useful) |
| `manifest.json` | Index of important files for the run |

> **Note:** `metrics.csv` is created by `CSVLogger`, so make sure you pass `callbacks=saver.callbacks()` into `model.fit()`.

---

## Installation

### Option A — Install from source

```bash
git clone https://github.com/AhmedAbdAlKareem1/experiment_saver.git
cd experiment_saver/experiment_saver_folder
pip install .
Quick Start
Import
from experiment_saver_folder.experiment_saver import ExperimentSaver, ExperimentConfig
Binary Classification Example
# 1) Create saver
cfg = ExperimentConfig(
    run_dir="runs/cat_dog_vgg16_exp001",
    patience=5,
    monitor="val_loss",
    save_best_only=True,
    verbose=1,
)

saver = ExperimentSaver(cfg, class_names=["Cat", "Dog"])

# 2) Train with callbacks (required to generate metrics.csv)
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5,
    callbacks=saver.callbacks(),
    verbose=1,
)

# 3) Save everything after training
saved_paths = saver.save_after_fit(
    model=model,
    history=history,
    val_ds=val_ds,
    extra_config={
        "optimizer": "Adam",
        "lr": 1e-3,
        "dataset_path": r"path_to_dataset",
        "backbone": "VGG16",
        "image_size": [224, 224],
    },
)

print("Saved files:", saved_paths)
Multi-Class Classification Example (HAM10000)
# 1) Create saver
cfg = ExperimentConfig(
    run_dir="runs/ham10000_exp001",
    patience=7,
    monitor="val_loss",
    save_best_only=True,
    verbose=1,
    roc_average="macro",
    roc_multi_class="ovr",
)

saver = ExperimentSaver(
    cfg,
    class_names=["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"],
)

# 2) Train with callbacks
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    callbacks=saver.callbacks(),
    verbose=1,
)

# 3) Save everything after training
saved_paths = saver.save_after_fit(
    model=model,
    history=history,
    val_ds=val_ds,
    extra_config={
        "optimizer": "Adam",
        "lr": 1e-4,
        "dataset": "HAM10000",
        "image_size": [224, 224],
        "num_classes": 7,
    },
)

print("Saved files:", saved_paths)
Configuration
ExperimentConfig:

run_dir: folder where all artifacts are saved

monitor: metric to track best model (e.g. "val_loss", "val_auc")

patience: EarlyStopping patience

save_best_only: saves checkpoint only when improved

roc_average: "macro" | "micro" | "weighted"

roc_multi_class: "ovr" | "ovo"

positive_class_index: for binary softmax (batch,2) (default: 1)

Tips / Common Issues
metrics.csv is missing

This usually happens when:

you didn’t pass callbacks=saver.callbacks() into model.fit(), or

training crashed before finishing the first epoch.

✅ Fix:

history = model.fit(..., callbacks=saver.callbacks())
