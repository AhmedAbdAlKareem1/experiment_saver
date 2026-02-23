# 🧪 experiment_saver

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-2.0+-D00000?logo=keras&logoColor=white)](https://keras.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A lightweight, safe utility for **TensorFlow / Keras** that automatically captures training artifacts and evaluation outputs. It keeps every run organized inside a single folder, allowing you to compare experiments without rewriting boilerplate code.

---

## ✨ Highlights

* 🚀 **One-line Callbacks**: Instant setup for `CSVLogger`, `ModelCheckpoint`, and `EarlyStopping`.
* 💾 **Comprehensive Saving**: Automatically stores **best** + **final** models, `history.json`, and `metrics.csv`.
* 📊 **Auto-Evaluation**: Automatically computes **ROC / AUC** curves after training.
* 🛡️ **Smart Handling**: Safely manages common output shapes including Sigmoid, Binary Softmax, and Multiclass.
* 📝 **Metadata Logging**: JSON-safe config saving that handles non-serializable values gracefully.

---

## 📂 Project Structure

All artifacts are organized within your specified `run_dir/`:

| File | Description |
| :--- | :--- |
| `metrics.csv` | Epoch-by-epoch logs (loss, accuracy, AUC, etc.) |
| `history.json` | Full training history in JSON format |
| `best_model.keras` | The best checkpoint based on your monitor metric |
| `final_model.keras` | The model state at the end of training |
| `roc_fpr.npy` | ROC False Positive Rate data |
| `roc_tpr.npy` | ROC True Positive Rate data |
| `roc_auc.json` | ROC AUC summary results |
| `config.json` | Optional experiment metadata and hyperparameters |
| `manifest.json` | A master index of all generated files |

---

## ⚙️ Installation

### Install from Source
```bash
git clone [https://github.com/AhmedAbdAlKareem1/experiment_saver.git](https://github.com/AhmedAbdAlKareem1/experiment_saver.git)
cd experiment_saver/experiment_saver_folder
pip install .
```
🚀 Quick Start
1. Initialize the Saver
Python
```
from experiment_saver_folder.experiment_saver import ExperimentSaver, ExperimentConfig

cfg = ExperimentConfig(
    run_dir="runs/cat_dog_exp001",
    patience=5,
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

saver = ExperimentSaver(cfg, class_names=["Cat", "Dog"])
2. Integrate with Training
Pass the generated callbacks into model.fit() to ensure metrics.csv is created.

Python
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5,
    callbacks=saver.callbacks(),
    verbose=1
)
3. Save Artifacts
Python
saved_paths = saver.save_after_fit(
    model=model,
    history=history,
    val_ds=val_ds,
    extra_config={
        "optimizer": "Adam",
        "lr": 1e-3,
        "backbone": "VGG16"
    }
)

```
🔧 Configuration Details
The ExperimentConfig class allows for fine-grained control:

run_dir: Folder where all artifacts are saved.

monitor: Metric to track for the best model (e.g., "val_loss").

patience: Number of epochs for EarlyStopping.

roc_average: Strategy for multiclass ROC ("macro", "micro", or "weighted").

positive_class_index: Index for binary softmax (default: 1).

💡 Troubleshooting
[!IMPORTANT]
Missing metrics.csv?
This usually happens if you didn't pass callbacks=saver.callbacks() into model.fit(), or if the training crashed before finishing the first epoch.

Python
# ✅ Correct usage:
model.fit(..., callbacks=saver.callbacks())
