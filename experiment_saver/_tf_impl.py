"""TensorFlow/Keras experiment saver implementation."""
from __future__ import annotations

import os
import platform
import random
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint

from ._common import (
    build_manifest,
    bundle_run_dir,
    make_json_safe,
    safe_write_json,
    save_git_commit,
    save_roc_artifacts,
    validate_binary_labels,
    validate_multiclass_labels,
)

__version__ = "0.2.0"


# ===========================================================================
# Config
# ===========================================================================

@dataclass
class ExperimentConfig:
    """
    Configuration for :class:`ExperimentSaver`.

    All boolean flags default to ``True`` — set any to ``False`` to skip
    saving that artifact.
    """

    run_dir: str = "runs/my_experiment_001"
    monitor: str = "val_loss"
    patience: int = 5
    save_best_only: bool = True
    verbose: int = 1

    # ROC / AUC
    roc_average: str = "macro"        # "macro", "micro", "weighted"
    roc_multi_class: str = "ovr"      # "ovr" or "ovo"
    positive_class_index: int = 1     # used for binary softmax (2-class) only

    # Evaluation & Metrics
    save_confusion_matrix: bool = True
    save_classification_report: bool = True
    save_val_predictions: bool = True
    save_val_scores: bool = True
    save_val_labels: bool = True

    # Model Metadata
    save_model_summary: bool = True
    save_param_count: bool = True
    save_model_architecture: bool = True

    # Reproducibility
    save_random_seeds: bool = True
    save_environment: bool = True
    save_git_commit: bool = True

    # Bundling
    bundle_artifacts: bool = False
    bundle_filename: str = "artifacts.zip"
    bundle_exclude_exts: Tuple[str, ...] = (".keras", ".h5", ".pt", ".pth")
    bundle_delete_originals: bool = False


# ===========================================================================
# ExperimentSaver
# ===========================================================================

class ExperimentSaver:
    """
    Comprehensive experiment saver for Keras classification models.

    Saves everything useful for reproducibility, debugging, and resuming:

    ── Core ──────────────────────────────────────────────────────────────────
      metrics.csv                per-epoch metrics via CSVLogger callback
      history.json               same as metrics.csv but as a dict
      best_model.keras           saved via ModelCheckpoint callback
      final_model.keras          saved at end of training
      config.json                ExperimentConfig + extra_config
      classes.json               class name list
      manifest.json              rich metadata + artifact file map

    ── Evaluation & Metrics ─────────────────────────────────────────────────
      roc_auc.json               AUC summary
      roc_fpr[_class_i].npy / roc_tpr[…].npy / roc_thresholds[…].npy
      confusion_matrix.npy       (C, C) int
      classification_report.json precision / recall / F1 per class
      val_predictions.npy        predicted class index  (N,)
      val_scores.npy             probability scores  (N,) or (N, C)
      val_labels.npy             ground-truth labels  (N,)

    ── Model Metadata ────────────────────────────────────────────────────────
      model_summary.txt          model.summary() string
      param_count.json           total / trainable / frozen parameter counts
      model_architecture.json    layer names + types + output shapes

    ── Reproducibility ───────────────────────────────────────────────────────
      random_seeds.json          numpy / python / tensorflow seeds
      environment.json           Python, TensorFlow, CUDA, sklearn versions
      git_commit.json            git hash, branch, dirty-flag

    Typical usage::

        saver = ExperimentSaver(config, class_names=["cat", "dog"])

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=20,
            callbacks=saver.callbacks(),
        )

        saved_paths = saver.save_after_fit(model, history, val_ds)
    """

    def __init__(
        self,
        config: ExperimentConfig,
        class_names: Optional[List[str]] = None,
    ) -> None:
        self.cfg = config
        self.class_names = class_names
        self._monitor_mode = "min" if "loss" in self.cfg.monitor else "max"
        self._best_monitor_value: Optional[float] = None

        os.makedirs(self.cfg.run_dir, exist_ok=True)

        self.paths: Dict[str, str] = {
            # Core
            "metrics_csv":              os.path.join(self.cfg.run_dir, "metrics.csv"),
            "history_json":             os.path.join(self.cfg.run_dir, "history.json"),
            "best_model":               os.path.join(self.cfg.run_dir, "best_model.keras"),
            "final_model":              os.path.join(self.cfg.run_dir, "final_model.keras"),
            "roc_auc_json":             os.path.join(self.cfg.run_dir, "roc_auc.json"),
            "config_json":              os.path.join(self.cfg.run_dir, "config.json"),
            "classes_json":             os.path.join(self.cfg.run_dir, "classes.json"),
            "manifest_json":            os.path.join(self.cfg.run_dir, "manifest.json"),
            # Evaluation
            "confusion_matrix":         os.path.join(self.cfg.run_dir, "confusion_matrix.npy"),
            "classification_report":    os.path.join(self.cfg.run_dir, "classification_report.json"),
            "val_predictions":          os.path.join(self.cfg.run_dir, "val_predictions.npy"),
            "val_scores":               os.path.join(self.cfg.run_dir, "val_scores.npy"),
            "val_labels":               os.path.join(self.cfg.run_dir, "val_labels.npy"),
            # Model Metadata
            "model_summary":            os.path.join(self.cfg.run_dir, "model_summary.txt"),
            "param_count_json":         os.path.join(self.cfg.run_dir, "param_count.json"),
            "model_architecture_json":  os.path.join(self.cfg.run_dir, "model_architecture.json"),
            # Reproducibility
            "random_seeds_json":        os.path.join(self.cfg.run_dir, "random_seeds.json"),
            "environment_json":         os.path.join(self.cfg.run_dir, "environment.json"),
            "git_commit_json":          os.path.join(self.cfg.run_dir, "git_commit.json"),
        }

        if self.class_names is not None:
            safe_write_json(self.paths["classes_json"], {"class_names": self.class_names})

    # -----------------------------------------------------------------------
    # Callbacks
    # -----------------------------------------------------------------------

    def callbacks(self) -> List[tf.keras.callbacks.Callback]:
        """Return the standard set of Keras callbacks for this experiment."""
        csv_logger = CSVLogger(self.paths["metrics_csv"], append=False)

        checkpoint = ModelCheckpoint(
            filepath=self.paths["best_model"],
            monitor=self.cfg.monitor,
            save_best_only=self.cfg.save_best_only,
            verbose=self.cfg.verbose,
        )

        early_stop = EarlyStopping(
            monitor=self.cfg.monitor,
            patience=self.cfg.patience,
            restore_best_weights=True,
            verbose=self.cfg.verbose,
        )

        return [csv_logger, checkpoint, early_stop]

    # -----------------------------------------------------------------------
    # Main save
    # -----------------------------------------------------------------------

    def save_after_fit(
        self,
        model: tf.keras.Model,
        history: tf.keras.callbacks.History,
        val_ds: tf.data.Dataset,
        extra_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        """
        Save all artifacts after ``model.fit()`` completes.

        Parameters
        ----------
        model:
            The trained Keras model.
        history:
            ``History`` object returned by ``model.fit()``.
        val_ds:
            ``tf.data.Dataset`` yielding ``(x, y)`` batches.
        extra_config:
            Optional extra key/value pairs merged into ``config.json``.

        Returns
        -------
        dict
            Mapping of artifact keys to their absolute file paths.
        """
        # ── History ───────────────────────────────────────────────────────
        hist_dict = getattr(history, "history", None)
        if not isinstance(hist_dict, dict):
            raise ValueError(
                "Invalid history object: history.history not found or not a dict. "
                "Pass the object returned directly by model.fit()."
            )
        safe_write_json(self.paths["history_json"], hist_dict)

        # Compute best monitor value from history
        monitor_vals = hist_dict.get(self.cfg.monitor, [])
        if monitor_vals:
            self._best_monitor_value = (
                float(min(monitor_vals))
                if self._monitor_mode == "min"
                else float(max(monitor_vals))
            )

        # ── Model ─────────────────────────────────────────────────────────
        self._safe_save_model(model, self.paths["final_model"])

        num_classes = self._infer_num_classes(model)

        if self.class_names is None:
            self.class_names = [f"class_{i}" for i in range(num_classes)]
            safe_write_json(self.paths["classes_json"], {"class_names": self.class_names})
        elif len(self.class_names) != num_classes:
            raise ValueError(
                f"class_names has {len(self.class_names)} entries but the model "
                f"has {num_classes} output classes. They must match."
            )

        cfg_payload: Dict[str, Any] = {
            "run_dir": self.cfg.run_dir,
            "monitor": self.cfg.monitor,
            "patience": self.cfg.patience,
            "save_best_only": self.cfg.save_best_only,
            "class_names": self.class_names,
            "num_classes": num_classes,
            "roc_average": self.cfg.roc_average,
            "roc_multi_class": self.cfg.roc_multi_class,
        }
        if extra_config:
            cfg_payload.update(make_json_safe(extra_config))
        safe_write_json(self.paths["config_json"], cfg_payload)

        # ── Collect validation predictions ────────────────────────────────
        y_true, y_score, y_pred = self._collect_predictions(model, val_ds, num_classes)

        # ── Evaluation ───────────────────────────────────────────────────
        if self.cfg.save_val_labels:
            np.save(self.paths["val_labels"], y_true)
        if self.cfg.save_val_scores:
            np.save(self.paths["val_scores"], y_score)
        if self.cfg.save_val_predictions:
            np.save(self.paths["val_predictions"], y_pred)

        roc_summary = save_roc_artifacts(
            y_true, y_score, num_classes,
            run_dir=self.cfg.run_dir,
            class_names=self.class_names,
            positive_class_index=self.cfg.positive_class_index,
            roc_multi_class=self.cfg.roc_multi_class,
            roc_average=self.cfg.roc_average,
        )
        safe_write_json(self.paths["roc_auc_json"], roc_summary)

        if self.cfg.save_confusion_matrix:
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_true, y_pred)
            np.save(self.paths["confusion_matrix"], cm)

        if self.cfg.save_classification_report:
            from sklearn.metrics import classification_report
            report = classification_report(
                y_true, y_pred,
                target_names=self.class_names,
                output_dict=True,
                zero_division=0,
            )
            safe_write_json(self.paths["classification_report"], report)

        # ── Model Metadata ────────────────────────────────────────────────
        if self.cfg.save_model_summary:
            lines: List[str] = []
            model.summary(print_fn=lambda line: lines.append(line))
            with open(self.paths["model_summary"], "w", encoding="utf-8") as f:
                f.write("\n".join(lines))

        if self.cfg.save_param_count:
            total = model.count_params()
            trainable = sum(tf.size(w).numpy() for w in model.trainable_weights)
            safe_write_json(
                self.paths["param_count_json"],
                {"total": int(total), "trainable": int(trainable), "frozen": int(total - trainable)},
            )

        if self.cfg.save_model_architecture:
            arch = {
                layer.name: {
                    "type": type(layer).__name__,
                    "output_shape": str(layer.output_shape),
                    "params": int(layer.count_params()),
                }
                for layer in model.layers
            }
            safe_write_json(self.paths["model_architecture_json"], arch)

        # ── Reproducibility ───────────────────────────────────────────────
        if self.cfg.save_random_seeds:
            safe_write_json(
                self.paths["random_seeds_json"],
                {
                    "numpy_seed":          int(np.random.get_state()[1][0]),
                    "python_random_state": random.getstate()[1][0],
                    "tf_global_seed":      str(tf.random.get_global_generator().state.numpy().tolist()),
                },
            )

        if self.cfg.save_environment:
            self._save_environment()

        if self.cfg.save_git_commit:
            save_git_commit(self.paths["git_commit_json"])

        # ── Manifest ──────────────────────────────────────────────────────
        manifest = build_manifest(
            paths=self.paths,
            framework="tensorflow",
            class_names=self.class_names,
            monitor=self.cfg.monitor,
            best_monitor_value=self._best_monitor_value,
            package_version=__version__,
        )
        safe_write_json(self.paths["manifest_json"], manifest)

        if self.cfg.verbose:
            print(f"\nAll artifacts saved to: {self.cfg.run_dir}")

        if self.cfg.bundle_artifacts:
            self.paths["bundle_zip"] = bundle_run_dir(
                run_dir=self.cfg.run_dir,
                zip_name=self.cfg.bundle_filename,
                exclude_exts=self.cfg.bundle_exclude_exts,
                delete_originals=self.cfg.bundle_delete_originals,
                verbose=self.cfg.verbose,
            )

        return dict(self.paths)

    # -----------------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------------

    def _infer_num_classes(self, model: tf.keras.Model) -> int:
        """Infer number of classes from the model's output shape."""
        out_shape = model.output_shape
        if isinstance(out_shape, list):
            raise ValueError(
                "Multi-output models are not supported. "
                "The model must have a single output tensor."
            )
        if len(out_shape) == 1:
            return 2
        if len(out_shape) == 2:
            c = out_shape[1]
            if c is None:
                raise ValueError(
                    "Model output dimension is None — cannot infer num_classes. "
                    "Pass class_names= explicitly to ExperimentSaver()."
                )
            return 2 if int(c) == 1 else int(c)
        raise ValueError(
            f"Unsupported model output shape: {out_shape}. "
            "Expected (batch,) or (batch, num_classes)."
        )

    def _collect_predictions(
        self,
        model: tf.keras.Model,
        dataset: tf.data.Dataset,
        num_classes: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Iterate *dataset* and return (y_true, y_score, y_pred).

        y_true : (N,) int32 ground-truth class indices
        y_score: (N,) or (N, C) float32 probabilities  [0, 1]
        y_pred : (N,) int32 predicted class indices
        """
        y_true_all: List[np.ndarray] = []
        y_score_all: List[np.ndarray] = []

        for batch in dataset:
            if not isinstance(batch, (tuple, list)) or len(batch) < 2:
                raise ValueError(
                    "val_ds must yield (x, y) batches. "
                    "Ensure your dataset is structured as (features, labels)."
                )
            x_batch, y_batch = batch[0], batch[1]

            y_true_all.append(self._labels_to_int(y_batch, num_classes))

            # model.predict applies the final activation (sigmoid/softmax)
            pred = model.predict(x_batch, verbose=0)
            y_score_all.append(self._pred_to_probs(pred, num_classes))

        y_true = np.concatenate(y_true_all, axis=0)
        y_score = np.concatenate(y_score_all, axis=0).astype(np.float32)
        y_score = np.clip(y_score, 0.0, 1.0)

        y_pred = (
            (y_score >= 0.5).astype(np.int32)
            if y_score.ndim == 1
            else np.argmax(y_score, axis=1).astype(np.int32)
        )

        if num_classes <= 2:
            validate_binary_labels(y_true)
        else:
            validate_multiclass_labels(y_true, num_classes)

        return y_true, y_score, y_pred

    def _labels_to_int(self, y: Any, num_classes: int) -> np.ndarray:
        y_np = y.numpy() if hasattr(y, "numpy") else np.asarray(y)
        if y_np.ndim == 2 and y_np.shape[1] > 1:
            return np.argmax(y_np, axis=1).astype(np.int32)
        y_np = y_np.reshape(-1)
        if np.issubdtype(y_np.dtype, np.floating):
            y_np = (
                (y_np >= 0.5).astype(np.int32)
                if num_classes <= 2
                else np.rint(y_np).astype(np.int32)
            )
        return y_np.astype(np.int32)

    def _pred_to_probs(self, pred: Any, num_classes: int) -> np.ndarray:
        """
        Convert raw model output (from ``model.predict()``) to probability array.

        For Keras models, ``model.predict()`` already applies the final activation,
        so this method just validates shapes.

        Returns (N,) for binary sigmoid or (N, C) for softmax.
        """
        pred_np = np.asarray(pred)

        if num_classes <= 2:
            if pred_np.ndim == 2 and pred_np.shape[1] == 1:
                return pred_np[:, 0].astype(np.float32)      # binary sigmoid → 1-D
            if pred_np.ndim == 2 and pred_np.shape[1] == 2:
                return pred_np.astype(np.float32)             # binary softmax → (N, 2)
            if pred_np.ndim == 1:
                return pred_np.astype(np.float32)
            raise ValueError(
                f"Binary model output shape {pred_np.shape} is not supported. "
                "Expected (batch, 1) for sigmoid or (batch, 2) for softmax."
            )

        if pred_np.ndim == 2 and pred_np.shape[1] == num_classes:
            return pred_np.astype(np.float32)

        raise ValueError(
            f"Multiclass model output shape {pred_np.shape} is not supported. "
            f"Expected (batch, {num_classes}) for {num_classes}-class softmax."
        )

    def _safe_save_model(self, model: tf.keras.Model, path: str) -> None:
        folder = os.path.dirname(path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        if not (path.endswith(".keras") or path.endswith(".h5")):
            raise ValueError(
                f"Model save path must end with .keras or .h5, got: '{path}'"
            )
        model.save(path, include_optimizer=False)

    def _save_environment(self) -> None:
        env: Dict[str, Any] = {
            "python_version":    sys.version,
            "platform":          platform.platform(),
            "tensorflow_version": tf.__version__,
            "cuda_available":    len(tf.config.list_physical_devices("GPU")) > 0,
            "gpu_devices":       [d.name for d in tf.config.list_physical_devices("GPU")],
        }
        for pkg in ("numpy", "sklearn", "cv2"):
            try:
                mod = __import__(pkg)
                env[f"{pkg}_version"] = mod.__version__
            except Exception:
                pass
        safe_write_json(self.paths["environment_json"], env)
