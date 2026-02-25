# experiment-saver

A lightweight experiment saver for **TensorFlow/Keras** and **PyTorch** classification projects.

It helps you keep every run organized in one folder (models, logs, ROC/AUC, confusion matrix, classification report, environment info, etc.) without rewriting boilerplate saving code each time.

---

## What this repo contains

This repo currently provides **two independent savers**:

### 1) TensorFlow / Keras
File: `saver_tensorflow.py`

Saves:
- `metrics.csv` (via `CSVLogger`)
- `best_model.keras` (via `ModelCheckpoint`)
- `final_model.keras`
- `history.json`
- `roc_auc.json` + ROC arrays (`.npy`)
- `confusion_matrix.npy`
- `classification_report.json`
- validation outputs (`val_labels.npy`, `val_scores.npy`, `val_predictions.npy`)
- reproducibility files (`random_seeds.json`, `environment.json`, `git_commit.json`)
- `manifest.json`

### 2) PyTorch
File: `experiment_saver_torch.py`

Includes everything above, plus extra PyTorch-specific utilities:
- optimizer + scheduler state saving
- last epoch saving
- gradient norm logging
- learning-rate history logging
- epoch time logging
- model summary (optional torchinfo)
- parameter count + architecture export

---

## Installation

### Option A) Install from GitHub (editable for development)
```
git clone https://github.com/AhmedAbdAlKareem1/experiment_saver.git
cd experiment_saver
pip install -e .
```
Option B) Install as normal
```
pip install .
```
Note: dependencies depend on which backend you use (TensorFlow or PyTorch).

TensorFlow / Keras usage
```
import tensorflow as tf
from saver_tensorflow import ExperimentSaver, ExperimentConfig

saver = ExperimentSaver(
    config=ExperimentConfig(
        run_dir="runs/keras_exp001",
        monitor="val_loss",
        patience=5,
        save_best_only=True,
        verbose=1,
    ),
    class_names=["Cat", "Dog"],   # optional but recommended
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    callbacks=saver.callbacks(),
    verbose=1,
)

saved_paths = saver.save_after_fit(
    model=model,
    history=history,
    val_ds=val_ds,
    extra_config={
        "backbone": "ResNet50",
        "image_size": [224, 224],
        "optimizer": "Adam",
        "lr": 1e-4,
    },
)

print("Saved artifacts:", saved_paths)
```
PyTorch usage
```
import time
import torch
from torch.utils.data import DataLoader

from experiment_saver_torch import ExperimentSaver, ExperimentConfig

saver = ExperimentSaver(
    config=ExperimentConfig(
        run_dir="runs/torch_exp001",
        monitor="val_loss",
        patience=5,
        verbose=1,
        model_input_size=(1, 3, 224, 224),  # optional, for torchinfo summary
    ),
    class_names=["Cat", "Dog"],
)

early_stop = saver.make_early_stopping()

for epoch in range(20):
    t0 = time.time()

    # --- train ---
    model.train()
    # train_loss = ...
    # (your training code here)

    # --- validate ---
    model.eval()
    # val_loss = ...
    # val_acc = ...

    saver.log_lr(optimizer, epoch=epoch)

    metrics = {
        "train_loss": float(train_loss),
        "val_loss": float(val_loss),
        "val_acc": float(val_acc),
    }
    saver.log_epoch(epoch, metrics, model=model, epoch_time=time.time() - t0)

    if saver.should_save_best(metrics[saver.cfg.monitor]):
        saver.save_best_checkpoint(model)

    if early_stop.step(metrics[saver.cfg.monitor]):
        break

saved_paths = saver.save_after_fit(
    model=model,
    val_loader=val_loader,
    optimizer=optimizer,
    scheduler=scheduler,
    last_epoch=epoch,
)

print("Saved artifacts:", saved_paths)
```
Output folder structure

All outputs are saved into your run_dir, for example:
```
runs/torch_exp001/
├── metrics.csv
├── history.json
├── best_model.pt
├── final_model.pt
├── roc_auc.json
├── roc_fpr.npy / roc_tpr.npy / roc_thresholds.npy   (binary)
├── roc_fpr_class_0.npy ...                          (multiclass)
├── confusion_matrix.npy
├── classification_report.json
├── val_labels.npy
├── val_scores.npy
├── val_predictions.npy
├── environment.json
├── git_commit.json
├── random_seeds.json
├── manifest.json
└── ...
```
Notes

Binary classification is supported (sigmoid or 2-class softmax).

Multiclass classification is supported (softmax with C classes).

ROC/AUC is computed using scikit-learn.
Multiclass classification is supported (softmax with C classes).

ROC/AUC is computed using scikit-learn.
