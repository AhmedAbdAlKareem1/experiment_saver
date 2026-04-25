# experiment-saver

A lightweight experiment artifact saver for **TensorFlow/Keras** and **PyTorch** classification projects.

Keeps every training run organized — models, logs, ROC/AUC, confusion matrix, classification report, environment info — without writing the same saving boilerplate each time.

---

## Installation

```bash
# TensorFlow / Keras
pip install "experiment-saver[tf]"

# PyTorch
pip install "experiment-saver[torch]"

# Both
pip install "experiment-saver[tf,torch]"

# Development (adds pytest, build, twine, torchinfo)
pip install -e ".[dev]"
```

---

## Quick start

### Keras (`model.fit`)

```python
from experiment_saver import TFExperimentConfig, TFExperimentSaver

cfg = TFExperimentConfig(
    run_dir="runs/keras_exp001",
    monitor="val_loss",
    patience=5,
    verbose=1,
)
saver = TFExperimentSaver(cfg, class_names=["cat", "dog"])

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    callbacks=saver.callbacks(),
)

saved_paths = saver.save_after_fit(
    model=model,
    history=history,
    val_ds=val_ds,
    extra_config={"backbone": "ResNet50", "lr": 1e-4},
)
print(saved_paths)
```

### PyTorch (custom training loop)

```python
import time
from experiment_saver import TorchExperimentConfig, TorchExperimentSaver

cfg = TorchExperimentConfig(
    run_dir="runs/torch_exp001",
    monitor="val_loss",
    patience=5,
    verbose=1,
    model_input_size=(1, 3, 224, 224),  # optional, enables torchinfo summary
)
saver = TorchExperimentSaver(cfg, class_names=["cat", "dog"])
early_stop = saver.make_early_stopping()

for epoch in range(50):
    t0 = time.time()
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)

    saver.log_lr(optimizer, epoch=epoch)
    scheduler.step()

    metrics = {"train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc}
    saver.log_epoch(epoch, metrics, model=model, epoch_time=time.time() - t0)

    if saver.should_save_best(val_loss):
        saver.save_best_checkpoint(model)

    if early_stop.step(val_loss):
        break

saved_paths = saver.save_after_fit(
    model=model,
    val_loader=val_loader,
    optimizer=optimizer,
    scheduler=scheduler,
    last_epoch=epoch,
    device=device,
    extra_config={"lr": 1e-3, "architecture": "resnet18"},
)
print(saved_paths)
```

---

## Saved artifacts

All outputs land inside `run_dir`:

```
runs/torch_exp001/
├── metrics.csv                  # per-epoch metrics log
├── history.json                 # full epoch history dict
├── lr_history.csv               # learning-rate per epoch (PyTorch)
├── best_model.pt                # best checkpoint (state_dict + metadata)
├── final_model.pt               # final model weights + metadata
├── roc_auc.json                 # ROC/AUC summary
├── roc_fpr.npy                  # ROC curve arrays (binary)
├── roc_tpr.npy
├── roc_thresholds.npy
├── roc_fpr_class_0.npy          # per-class ROC arrays (multiclass)
├── confusion_matrix.npy
├── classification_report.json
├── val_labels.npy               # raw validation outputs
├── val_scores.npy
├── val_predictions.npy
├── environment.json             # pip freeze snapshot
├── git_commit.json              # current git HEAD
├── random_seeds.json
├── model_summary.txt            # torchinfo summary (if installed)
├── model_architecture.json      # layer/parameter counts
└── manifest.json                # run metadata index
```

### `manifest.json` format

```json
{
  "package_version": "0.2.0",
  "framework": "pytorch",
  "timestamp": "2026-04-25T10:30:00+00:00",
  "python_version": "3.11.5 ...",
  "monitor": "val_loss",
  "best_monitor_value": 0.3142,
  "class_names": ["cat", "dog"],
  "artifacts": {
    "final_model": "final_model.pt",
    "manifest_json": "manifest.json",
    ...
  }
}
```

### Best checkpoint format (PyTorch)

`best_model.pt` is a dict loadable with `torch.load`:

```python
ckpt = torch.load("runs/torch_exp001/best_model.pt", map_location="cpu")
model.load_state_dict(ckpt["model_state_dict"])
# ckpt["class_names"]        → ["cat", "dog"]
# ckpt["monitor"]            → "val_loss"
# ckpt["best_monitor_value"] → 0.3142
```

---

## Configuration reference

### `TFExperimentConfig` / `TorchExperimentConfig`

| Field | Default | Description |
|---|---|---|
| `run_dir` | `"runs/my_experiment_001"` | Output directory |
| `monitor` | `"val_loss"` | Metric to watch for best checkpoint |
| `patience` | `5` | Early-stopping patience |
| `save_best_only` | `True` | Only save checkpoint on improvement |
| `verbose` | `1` | 0 = silent, 1 = print save events |
| `roc_average` | `"macro"` | sklearn ROC average mode |
| `roc_multi_class` | `"ovr"` | sklearn multiclass strategy |
| `positive_class_index` | `1` | Index of positive class (binary) |
| `save_environment` | `True` | Save `environment.json` |
| `save_git_commit` | `True` | Save `git_commit.json` |
| `bundle_artifacts` | `False` | Zip all artifacts after saving |
| `bundle_filename` | `"artifacts.zip"` | Name of the zip file |
| `bundle_delete_originals` | `False` | Remove individual files after zipping |

**PyTorch-only fields:**

| Field | Default | Description |
|---|---|---|
| `model_input_size` | `None` | Input shape for torchinfo summary |
| `model_outputs_logits` | `False` | Set `True` if model returns raw logits |
| `save_optimizer_state` | `True` | Include optimizer state in checkpoint |
| `save_scheduler_state` | `True` | Include scheduler state in checkpoint |
| `save_grad_norms` | `True` | Log gradient norms each epoch |
| `save_lr_history` | `True` | Write `lr_history.csv` |
| `save_epoch_times` | `True` | Include per-epoch timing in history |

---

## Binary vs multiclass

Both tasks are detected automatically from the model output shape:

- **Binary**: final layer outputs 1 neuron (sigmoid) or 2 neurons (softmax).  
  ROC/AUC uses `roc_auc_score` with a single scalar.
- **Multiclass**: final layer outputs C > 2 neurons.  
  ROC/AUC uses one-vs-rest per class; `per_class_auc` is written to `roc_auc.json`.

`class_names` is optional but recommended — it labels the confusion matrix and classification report.

---

## Artifact bundling

To produce a single zip file per run:

```python
cfg = TFExperimentConfig(
    run_dir="runs/exp001",
    bundle_artifacts=True,
    bundle_filename="run_artifacts.zip",
    bundle_delete_originals=True,   # keep only the zip
)
```

Model files (`.keras`, `.h5`, `.pt`, `.pth`) are excluded from the bundle by default.

---

## Running tests

```bash
pip install -e ".[dev]"
python -m pytest tests/ -v --tb=short
```

Tests use `pytest.importorskip` — TF tests are skipped automatically when TensorFlow is not installed, and PyTorch tests are skipped when torch is not installed.

---

## Examples

| File | Description |
|---|---|
| `examples/keras_fit_binary.py` | Keras binary classification |
| `examples/keras_fit_multiclass.py` | Keras 4-class classification |
| `examples/torch_training_loop_binary.py` | PyTorch binary with custom loop |
| `examples/torch_training_loop_multiclass.py` | PyTorch 4-class with custom loop |
