"""
Example: Keras model.fit() workflow — binary classification.

Run:
    pip install "experiment-saver[tf]"
    python examples/keras_fit_binary.py
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf

from experiment_saver import TFExperimentConfig, TFExperimentSaver


# ---------------------------------------------------------------------------
# Tiny demo dataset (replace with real data)
# ---------------------------------------------------------------------------

def make_datasets(n: int = 200, val_split: float = 0.2):
    X = np.random.randn(n, 16).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(np.int32)
    n_val = int(n * val_split)
    train_ds = (
        tf.data.Dataset.from_tensor_slices((X[n_val:], y[n_val:]))
        .shuffle(1000).batch(32)
    )
    val_ds = tf.data.Dataset.from_tensor_slices((X[:n_val], y[:n_val])).batch(32)
    return train_ds, val_ds


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(input_dim: int = 16):
    m = tf.keras.Sequential([
        tf.keras.Input((input_dim,)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    m.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return m


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    train_ds, val_ds = make_datasets()
    model = build_model()

    cfg = TFExperimentConfig(
        run_dir="runs/keras_binary_001",
        monitor="val_loss",
        patience=5,
        save_best_only=True,
        verbose=1,
    )

    saver = TFExperimentSaver(cfg, class_names=["negative", "positive"])

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
            "learning_rate": 1e-3,
            "architecture": "dense_2hidden",
        },
    )

    print("\nSaved artifacts:")
    for k, v in saved_paths.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
