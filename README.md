# Experiment Saver

A lightweight, safe utility for TensorFlow/Keras to automate the saving of model artifacts, training history, and evaluation metrics.

## Features
- **Callback Integration**: Automatically sets up `ModelCheckpoint`, `EarlyStopping`, and `CSVLogger`.
- **Artifact Management**: Organizes models (`.keras`), history (`.json`), and class names in a single run directory.
- **Auto-ROC Evaluation**: Automatically computes ROC curves (FPR, TPR, Thresholds) and AUC scores using the validation dataset after training.
- **JSON Serialization**: Best-effort conversion of complex experiment configs into JSON-safe formats.
- **Shape Agnostic**: Supports both `sigmoid` (1 unit) and `softmax` (2 units) output layers.

## Installation

### From Source
```
git clone [https://github.com/AhmedAbdAlKareem1/experiment_saver.git](https://github.com/AhmedAbdAlKareem1/experiment_saver.git)
```
```
cd experiment_saver
pip install .
```
Dependencies
TensorFlow 2.x

NumPy

Scikit-learn

#Quick Start
```
from experiment_saver import ExperimentSaver, ExperimentConfig

# 1. Initialize Configuration
config = ExperimentConfig(
    run_dir="runs/cat_dog_vgg16_exp001", 
    patience=5,
    monitor="val_loss"
)

# 2. Create Saver
saver = ExperimentSaver(config=config, class_names=["Cat", "Dog"])

# 3. Train with auto-generated callbacks
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,
    callbacks=saver.callbacks(),
)

# 4. Save everything (History, Final Model, ROC curves, Config)
saved_paths = saver.save_after_fit(
    model=model,
    history=history,
    val_ds=val_ds,
    extra_config={
        "backbone": "VGG16",
        "optimizer": "Adam",
        "lr": 1e-3
    }
)

print(f"Artifacts saved to: {saved_paths['history_json']}")
```
Saved Artifacts
The utility creates the following structure in your run_dir:

best_model.keras: The model weights with the best monitored metric.

final_model.keras: The model state at the end of training.

history.json: Full training history dictionary.

metrics.csv: Epoch-by-epoch logs.

roc_fpr.npy / roc_tpr.npy: Arrays for plotting ROC curves.

roc_auc.json: The calculated Area Under the Curve.

config.json: Metadata about the experiment parameters.

# Environment
venv/

.env

