"""
Example: Keras model.fit() workflow — multiclass classification (4 classes).

Run:
    pip install "experiment-saver[tf]"
    python examples/keras_fit_multiclass.py
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf

from experiment_saver import TFExperimentConfig, TFExperimentSaver

NUM_CLASSES = 4


def make_datasets(n: int = 400, val_split: float = 0.2):
    X = np.random.randn(n, 32).astype(np.float32)
    y = np.random.randint(0, NUM_CLASSES, n).astype(np.int32)
    n_val = int(n * val_split)
    train_ds = (
        tf.data.Dataset.from_tensor_slices((X[n_val:], y[n_val:]))
        .shuffle(1000).batch(32)
    )
    val_ds = tf.data.Dataset.from_tensor_slices((X[:n_val], y[:n_val])).batch(32)
    return train_ds, val_ds


def build_model(input_dim: int = 32):
    m = tf.keras.Sequential([
        tf.keras.Input((input_dim,)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(NUM_CLASSES, activation="softmax"),
    ])
    m.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return m


def main():
    class_names = ["alpha", "beta", "gamma", "delta"]
    train_ds, val_ds = make_datasets()
    model = build_model()

    cfg = TFExperimentConfig(
        run_dir="runs/keras_multiclass_001",
        monitor="val_accuracy",
        patience=5,
        verbose=1,
    )

    saver = TFExperimentSaver(cfg, class_names=class_names)

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
        extra_config={"num_classes": NUM_CLASSES},
    )

    print("\nSaved artifacts:")
    for k, v in saved_paths.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
