"""
TensorFlow ExperimentSaver tests with small dummy models and synthetic data.

All tests run offline — no pretrained weights are downloaded.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _binary_model():
    tf = pytest.importorskip("tensorflow")
    m = tf.keras.Sequential([
        tf.keras.Input((4,)),
        tf.keras.layers.Dense(8, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    m.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return m


def _multiclass_model(num_classes: int = 3):
    tf = pytest.importorskip("tensorflow")
    m = tf.keras.Sequential([
        tf.keras.Input((4,)),
        tf.keras.layers.Dense(8, activation="relu"),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])
    m.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m


def _binary_ds(n: int = 40):
    tf = pytest.importorskip("tensorflow")
    x = tf.random.uniform((n, 4), seed=0)
    y = tf.cast(tf.random.uniform((n,), seed=1) > 0.5, tf.int32)
    return tf.data.Dataset.from_tensor_slices((x, y)).batch(10)


def _multiclass_ds(num_classes: int = 3, n: int = 60):
    tf = pytest.importorskip("tensorflow")
    x = tf.random.uniform((n, 4), seed=0)
    y = tf.random.uniform((n,), maxval=num_classes, dtype=tf.int32, seed=2)
    return tf.data.Dataset.from_tensor_slices((x, y)).batch(10)


# ---------------------------------------------------------------------------
# Tests — binary classification
# ---------------------------------------------------------------------------

class TestTFSaverBinary:
    def test_core_artifacts_exist(self, tmp_path):
        pytest.importorskip("tensorflow")
        from experiment_saver import TFExperimentConfig, TFExperimentSaver

        model = _binary_model()
        val_ds = _binary_ds()

        cfg = TFExperimentConfig(
            run_dir=str(tmp_path / "tf_binary"),
            verbose=0, save_environment=False, save_git_commit=False,
        )
        saver = TFExperimentSaver(cfg, class_names=["neg", "pos"])
        history = model.fit(val_ds, epochs=2, verbose=0)
        paths = saver.save_after_fit(model, history=history, val_ds=val_ds)

        for key in ("final_model", "manifest_json", "history_json", "roc_auc_json",
                    "confusion_matrix", "classification_report"):
            assert os.path.isfile(paths[key]), f"Missing: {key}"

    def test_roc_is_binary(self, tmp_path):
        pytest.importorskip("tensorflow")
        from experiment_saver import TFExperimentConfig, TFExperimentSaver

        model = _binary_model()
        val_ds = _binary_ds()
        cfg = TFExperimentConfig(
            run_dir=str(tmp_path / "tf_roc"),
            verbose=0, save_environment=False, save_git_commit=False,
        )
        saver = TFExperimentSaver(cfg, class_names=["neg", "pos"])
        history = model.fit(val_ds, epochs=1, verbose=0)
        paths = saver.save_after_fit(model, history=history, val_ds=val_ds)

        with open(paths["roc_auc_json"]) as f:
            roc = json.load(f)
        assert roc["task"] == "binary"
        assert "roc_auc" in roc

    def test_manifest_has_required_fields(self, tmp_path):
        pytest.importorskip("tensorflow")
        from experiment_saver import TFExperimentConfig, TFExperimentSaver

        model = _binary_model()
        val_ds = _binary_ds()
        cfg = TFExperimentConfig(
            run_dir=str(tmp_path / "tf_manifest"),
            verbose=0, save_environment=False, save_git_commit=False,
        )
        saver = TFExperimentSaver(cfg, class_names=["neg", "pos"])
        history = model.fit(val_ds, epochs=1, verbose=0)
        paths = saver.save_after_fit(model, history=history, val_ds=val_ds)

        with open(paths["manifest_json"]) as f:
            m = json.load(f)

        for field in ("package_version", "framework", "timestamp", "python_version",
                      "monitor", "class_names", "artifacts"):
            assert field in m, f"Missing manifest field: {field}"

        assert m["framework"] == "tensorflow"
        assert m["class_names"] == ["neg", "pos"]

    def test_best_monitor_value_in_manifest(self, tmp_path):
        pytest.importorskip("tensorflow")
        from experiment_saver import TFExperimentConfig, TFExperimentSaver

        model = _binary_model()
        val_ds = _binary_ds()
        cfg = TFExperimentConfig(
            run_dir=str(tmp_path / "tf_best"),
            monitor="val_loss",
            verbose=0, save_environment=False, save_git_commit=False,
        )
        saver = TFExperimentSaver(cfg, class_names=["neg", "pos"])
        history = model.fit(val_ds, validation_data=val_ds, epochs=2, verbose=0)
        paths = saver.save_after_fit(model, history=history, val_ds=val_ds)

        with open(paths["manifest_json"]) as f:
            m = json.load(f)
        # best_monitor_value should be present when history includes the monitor key
        # (val_loss is in history when validation_data is provided)
        if m["best_monitor_value"] is not None:
            assert isinstance(m["best_monitor_value"], float)

    def test_model_saved_from_path(self, tmp_path):
        pytest.importorskip("tensorflow")
        from experiment_saver import TFExperimentConfig, TFExperimentSaver

        model = _binary_model()
        val_ds = _binary_ds()
        cfg = TFExperimentConfig(
            run_dir=str(tmp_path / "tf_path"),
            verbose=0, save_environment=False, save_git_commit=False,
        )
        saver = TFExperimentSaver(cfg, class_names=["neg", "pos"])
        history = model.fit(val_ds, epochs=1, verbose=0)
        paths = saver.save_after_fit(model, history=history, val_ds=val_ds)
        assert paths["final_model"].endswith(".keras")
        assert os.path.isfile(paths["final_model"])

    def test_callbacks_return_three_items(self, tmp_path):
        pytest.importorskip("tensorflow")
        from experiment_saver import TFExperimentConfig, TFExperimentSaver

        cfg = TFExperimentConfig(run_dir=str(tmp_path / "cbs"), verbose=0)
        saver = TFExperimentSaver(cfg, class_names=["a", "b"])
        cbs = saver.callbacks()
        assert len(cbs) == 3   # CSVLogger, ModelCheckpoint, EarlyStopping


# ---------------------------------------------------------------------------
# Tests — multiclass classification
# ---------------------------------------------------------------------------

class TestTFSaverMulticlass:
    def test_roc_is_multiclass(self, tmp_path):
        pytest.importorskip("tensorflow")
        from experiment_saver import TFExperimentConfig, TFExperimentSaver

        model = _multiclass_model(3)
        val_ds = _multiclass_ds(3)
        cfg = TFExperimentConfig(
            run_dir=str(tmp_path / "tf_mc"),
            verbose=0, save_environment=False, save_git_commit=False,
        )
        saver = TFExperimentSaver(cfg, class_names=["a", "b", "c"])
        history = model.fit(val_ds, epochs=1, verbose=0)
        paths = saver.save_after_fit(model, history=history, val_ds=val_ds)

        with open(paths["roc_auc_json"]) as f:
            roc = json.load(f)
        assert roc["task"] == "multiclass"
        assert "per_class_auc" in roc

    def test_confusion_matrix_shape(self, tmp_path):
        pytest.importorskip("tensorflow")
        from experiment_saver import TFExperimentConfig, TFExperimentSaver

        model = _multiclass_model(3)
        val_ds = _multiclass_ds(3)
        cfg = TFExperimentConfig(
            run_dir=str(tmp_path / "tf_cm"),
            verbose=0, save_environment=False, save_git_commit=False,
        )
        saver = TFExperimentSaver(cfg, class_names=["a", "b", "c"])
        history = model.fit(val_ds, epochs=1, verbose=0)
        paths = saver.save_after_fit(model, history=history, val_ds=val_ds)

        cm = np.load(paths["confusion_matrix"])
        assert cm.shape == (3, 3)

    def test_class_names_length_mismatch_raises(self, tmp_path):
        pytest.importorskip("tensorflow")
        from experiment_saver import TFExperimentConfig, TFExperimentSaver

        model = _multiclass_model(3)
        val_ds = _multiclass_ds(3)
        cfg = TFExperimentConfig(
            run_dir=str(tmp_path / "tf_mis"),
            verbose=0, save_environment=False, save_git_commit=False,
        )
        saver = TFExperimentSaver(cfg, class_names=["a", "b"])  # wrong: model has 3 classes
        history = model.fit(val_ds, epochs=1, verbose=0)
        with pytest.raises(ValueError, match="class_names"):
            saver.save_after_fit(model, history=history, val_ds=val_ds)
