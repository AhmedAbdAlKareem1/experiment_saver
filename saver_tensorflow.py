import os
import sys
import json
import random
import platform
import subprocess
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
from sklearn.metrics import (
    roc_curve, auc, roc_auc_score,
    confusion_matrix, classification_report,
)
import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping


# ===========================================================================
# Config
# ===========================================================================

@dataclass
class ExperimentConfig:
    run_dir: str = "runs/my_experiment_001"
    monitor: str = "val_loss"
    patience: int = 5
    save_best_only: bool = True
    verbose: int = 1

    # ROC/AUC options
    roc_average: str = "macro"  # "macro", "micro", "weighted"
    roc_multi_class: str = "ovr"  # "ovr" or "ovo"
    positive_class_index: int = 1  # used only for binary softmax (2 classes)

    # -----------------------------------------------------------------------
    # Feature flags — all ON by default
    # -----------------------------------------------------------------------

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
      manifest.json              map of all saved file names

    ── Evaluation & Metrics ─────────────────────────────────────────────────
      roc_auc.json               AUC summary (binary or per-class + macro/micro/weighted)
      roc_fpr[_class_i].npy / roc_tpr[…].npy / roc_thresholds[…].npy
      confusion_matrix.npy       shape (C, C) int
      classification_report.json precision / recall / F1 per class
      val_predictions.npy        predicted class index per sample  (N,)
      val_scores.npy             probability scores per sample  (N,) or (N,C)
      val_labels.npy             ground-truth labels  (N,)

    ── Model Metadata ────────────────────────────────────────────────────────
      model_summary.txt          model.summary() string output
      param_count.json           total / trainable / frozen parameter counts
      model_architecture.json    layer names + types + output shapes

    ── Reproducibility ───────────────────────────────────────────────────────
      random_seeds.json          numpy / python / tensorflow seeds
      environment.json           Python, TensorFlow, CUDA, sklearn versions
      git_commit.json            git hash, branch, dirty-flag

    ─────────────────────────────────────────────────────────────────────────

    Typical usage
    ─────────────
        saver = ExperimentSaver(config, class_names=["cat", "dog"])

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=20,
            callbacks=saver.callbacks(),
        )

        saved_paths = saver.save_after_fit(
            model, history, val_ds,
            extra_config={"learning_rate": 1e-3},
        )
    """

    def __init__(self, config: ExperimentConfig, class_names: Optional[List[str]] = None):
        self.cfg = config
        self.class_names = class_names

        os.makedirs(self.cfg.run_dir, exist_ok=True)

        self.paths: Dict[str, str] = {
            # Core
            "metrics_csv": os.path.join(self.cfg.run_dir, "metrics.csv"),
            "history_json": os.path.join(self.cfg.run_dir, "history.json"),
            "best_model": os.path.join(self.cfg.run_dir, "best_model.keras"),
            "final_model": os.path.join(self.cfg.run_dir, "final_model.keras"),
            "roc_auc_json": os.path.join(self.cfg.run_dir, "roc_auc.json"),
            "config_json": os.path.join(self.cfg.run_dir, "config.json"),
            "classes_json": os.path.join(self.cfg.run_dir, "classes.json"),
            "manifest_json": os.path.join(self.cfg.run_dir, "manifest.json"),
            # Evaluation
            "confusion_matrix": os.path.join(self.cfg.run_dir, "confusion_matrix.npy"),
            "classification_report": os.path.join(self.cfg.run_dir, "classification_report.json"),
            "val_predictions": os.path.join(self.cfg.run_dir, "val_predictions.npy"),
            "val_scores": os.path.join(self.cfg.run_dir, "val_scores.npy"),
            "val_labels": os.path.join(self.cfg.run_dir, "val_labels.npy"),
            # Model Metadata
            "model_summary": os.path.join(self.cfg.run_dir, "model_summary.txt"),
            "param_count_json": os.path.join(self.cfg.run_dir, "param_count.json"),
            "model_architecture_json": os.path.join(self.cfg.run_dir, "model_architecture.json"),
            # Reproducibility
            "random_seeds_json": os.path.join(self.cfg.run_dir, "random_seeds.json"),
            "environment_json": os.path.join(self.cfg.run_dir, "environment.json"),
            "git_commit_json": os.path.join(self.cfg.run_dir, "git_commit.json"),
        }

        if self.class_names is not None:
            self._safe_write_json(self.paths["classes_json"], {"class_names": self.class_names})

    # -----------------------------------------------------------------------
    # Callbacks  (unchanged from original — Keras handles these natively)
    # -----------------------------------------------------------------------

    def callbacks(self) -> List[tf.keras.callbacks.Callback]:
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
    # Main save call
    # -----------------------------------------------------------------------

    def save_after_fit(
            self,
            model: tf.keras.Model,
            history: tf.keras.callbacks.History,
            val_ds: tf.data.Dataset,
            extra_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        """
        Call once after model.fit() is complete.

        Parameters
        ----------
        model       : trained Keras model
        history     : History object returned by model.fit()
        val_ds      : tf.data.Dataset yielding (x, y) batches
        extra_config: optional extra key-value pairs added to config.json
        """
        # ── Core ──────────────────────────────────────────────────────────
        hist_dict = getattr(history, "history", None)
        if not isinstance(hist_dict, dict):
            raise ValueError("Invalid history object: history.history not found or not a dict.")
        self._safe_write_json(self.paths["history_json"], hist_dict)

        self._safe_save_model(model, self.paths["final_model"])

        num_classes = self._infer_num_classes_from_model(model)

        if self.class_names is None:
            self.class_names = [f"class_{i}" for i in range(num_classes)]
            self._safe_write_json(self.paths["classes_json"], {"class_names": self.class_names})
        elif len(self.class_names) != num_classes:
            raise ValueError(
                f"class_names length ({len(self.class_names)}) does not match "
                f"model output classes ({num_classes})."
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
            cfg_payload.update(self._make_json_safe(extra_config))
        self._safe_write_json(self.paths["config_json"], cfg_payload)

        # ── Collect val predictions (shared by all eval saves below) ──────
        y_true, y_score, y_pred = self._collect_labels_scores_preds(model, val_ds, num_classes)

        # ── Evaluation & Metrics ──────────────────────────────────────────
        if self.cfg.save_val_labels:
            np.save(self.paths["val_labels"], y_true)

        if self.cfg.save_val_scores:
            np.save(self.paths["val_scores"], y_score)

        if self.cfg.save_val_predictions:
            np.save(self.paths["val_predictions"], y_pred)

        # ROC / AUC
        roc_summary = self._save_roc_artifacts(y_true, y_score, num_classes)
        self._safe_write_json(self.paths["roc_auc_json"], roc_summary)

        # Confusion matrix
        if self.cfg.save_confusion_matrix:
            cm = confusion_matrix(y_true, y_pred)
            np.save(self.paths["confusion_matrix"], cm)

        # Classification report
        if self.cfg.save_classification_report:
            report = classification_report(
                y_true, y_pred,
                target_names=self.class_names,
                output_dict=True,
                zero_division=0,
            )
            self._safe_write_json(self.paths["classification_report"], report)

        # ── Model Metadata ────────────────────────────────────────────────
        if self.cfg.save_model_summary:
            self._save_model_summary(model)

        if self.cfg.save_param_count:
            total = model.count_params()
            trainable = sum(tf.size(w).numpy() for w in model.trainable_weights)
            frozen = total - trainable
            self._safe_write_json(
                self.paths["param_count_json"],
                {"total": int(total), "trainable": int(trainable), "frozen": int(frozen)},
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
            self._safe_write_json(self.paths["model_architecture_json"], arch)

        # ── Reproducibility ───────────────────────────────────────────────
        if self.cfg.save_random_seeds:
            self._safe_write_json(
                self.paths["random_seeds_json"],
                {
                    "numpy_seed": int(np.random.get_state()[1][0]),
                    "python_random_state": random.getstate()[1][0],
                    "tf_global_seed": str(tf.random.get_global_generator().state.numpy().tolist()),
                },
            )

        if self.cfg.save_environment:
            self._save_environment()

        if self.cfg.save_git_commit:
            self._save_git_commit()

        # ── Manifest ──────────────────────────────────────────────────────
        manifest = {k: os.path.basename(v) for k, v in self.paths.items()}
        self._safe_write_json(self.paths["manifest_json"], manifest)

        if self.cfg.verbose:
            print(f"\nAll artifacts saved to: {self.cfg.run_dir}")

        return dict(self.paths)

    # -----------------------------------------------------------------------
    # Internals — model introspection
    # -----------------------------------------------------------------------

    def _infer_num_classes_from_model(self, model: tf.keras.Model) -> int:
        out_shape = model.output_shape
        if isinstance(out_shape, list):
            raise ValueError("Multi-output models are not supported.")
        if len(out_shape) == 1:
            return 2
        if len(out_shape) == 2:
            c = out_shape[1]
            if c is None:
                raise ValueError("Model output dimension is None; cannot infer classes.")
            return 2 if int(c) == 1 else int(c)
        raise ValueError(f"Unsupported model output shape: {out_shape}")

    def _save_model_summary(self, model: tf.keras.Model) -> None:
        lines: List[str] = []
        model.summary(print_fn=lambda line: lines.append(line))
        with open(self.paths["model_summary"], "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def _save_environment(self) -> None:
        env: Dict[str, Any] = {
            "python_version": sys.version,
            "platform": platform.platform(),
            "tensorflow_version": tf.__version__,
            "cuda_available": len(tf.config.list_physical_devices("GPU")) > 0,
            "gpu_devices": [d.name for d in tf.config.list_physical_devices("GPU")],
        }
        for pkg in ("numpy", "sklearn"):
            try:
                mod = __import__(pkg)
                env[f"{pkg}_version"] = mod.__version__
            except Exception:
                pass
        self._safe_write_json(self.paths["environment_json"], env)

    def _save_git_commit(self) -> None:
        info: Dict[str, Any] = {}
        try:
            info["commit"] = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            ).decode().strip()
            dirty = subprocess.check_output(
                ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
            ).decode().strip()
            info["dirty"] = bool(dirty)
            info["dirty_files"] = dirty.splitlines() if dirty else []
            info["branch"] = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL
            ).decode().strip()
        except Exception as e:
            info["error"] = str(e)
        self._safe_write_json(self.paths["git_commit_json"], info)

    # -----------------------------------------------------------------------
    # Internals — data collection
    # -----------------------------------------------------------------------

    def _collect_labels_scores_preds(
            self,
            model: tf.keras.Model,
            dataset: tf.data.Dataset,
            num_classes: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns
        -------
        y_true : (N,)        int32 ground-truth labels
        y_score: (N,) or (N,C)  float32 probabilities
        y_pred : (N,)        int32 predicted class indices
        """
        y_true_all: List[np.ndarray] = []
        y_score_all: List[np.ndarray] = []

        for batch in dataset:
            if not isinstance(batch, (tuple, list)) or len(batch) < 2:
                raise ValueError("val_ds must yield (x, y) batches.")
            x_batch, y_batch = batch[0], batch[1]

            y_true_all.append(self._labels_to_int_1d(y_batch, num_classes))

            pred = model.predict(x_batch, verbose=0)
            y_score_all.append(self._pred_to_scores(pred, num_classes))

        y_true = np.concatenate(y_true_all, axis=0)
        y_score = np.concatenate(y_score_all, axis=0).astype(np.float32)
        y_score = np.clip(y_score, 0.0, 1.0)

        # Hard predictions
        if y_score.ndim == 1:
            y_pred = (y_score >= 0.5).astype(np.int32)
        else:
            y_pred = np.argmax(y_score, axis=1).astype(np.int32)

        if num_classes <= 2:
            self._validate_binary_labels(y_true)
        else:
            self._validate_multiclass_labels(y_true, num_classes)

        return y_true, y_score, y_pred

    def _labels_to_int_1d(self, y: Any, num_classes: int) -> np.ndarray:
        y_np = y.numpy() if hasattr(y, "numpy") else np.asarray(y)
        if y_np.ndim == 2 and y_np.shape[1] > 1:
            return np.argmax(y_np, axis=1).astype(np.int32)
        y_np = y_np.reshape(-1)
        if np.issubdtype(y_np.dtype, np.floating):
            y_np = (y_np >= 0.5).astype(np.int32) if num_classes <= 2 else np.rint(y_np).astype(np.int32)
        else:
            y_np = y_np.astype(np.int32)
        return y_np

    def _pred_to_scores(self, pred: Any, num_classes: int) -> np.ndarray:
        pred_np = np.asarray(pred)
        if num_classes == 1:
            if pred_np.ndim == 1:
                return pred_np.astype(np.float32)
            if pred_np.ndim == 2 and pred_np.shape[1] == 1:
                return pred_np[:, 0].astype(np.float32)
            raise ValueError(f"Unsupported sigmoid shape {pred_np.shape}.")
        if num_classes == 2:
            if pred_np.ndim == 2 and pred_np.shape[1] == 2:
                return pred_np.astype(np.float32)
            if pred_np.ndim == 2 and pred_np.shape[1] == 1:
                return pred_np[:, 0].astype(np.float32)
            if pred_np.ndim == 1:
                return pred_np.astype(np.float32)
            raise ValueError(f"Unsupported binary shape {pred_np.shape}.")
        if pred_np.ndim == 2 and pred_np.shape[1] == num_classes:
            return pred_np.astype(np.float32)
        raise ValueError(f"Unsupported multiclass shape {pred_np.shape}. Expected (batch, {num_classes}).")

    # -----------------------------------------------------------------------
    # Internals — ROC / AUC
    # -----------------------------------------------------------------------

    def _save_roc_artifacts(
            self, y_true: np.ndarray, y_score: np.ndarray, num_classes: int
    ) -> Dict[str, Any]:
        if num_classes <= 2:
            score_1d = (
                y_score[:, self.cfg.positive_class_index]
                if y_score.ndim == 2
                else y_score
            )
            fpr, tpr, thresholds = roc_curve(y_true, score_1d)
            roc_auc_value = auc(fpr, tpr)
            np.save(os.path.join(self.cfg.run_dir, "roc_fpr.npy"), fpr)
            np.save(os.path.join(self.cfg.run_dir, "roc_tpr.npy"), tpr)
            np.save(os.path.join(self.cfg.run_dir, "roc_thresholds.npy"), thresholds)
            return {
                "task": "binary",
                "roc_auc": float(roc_auc_value),
                "positive_class_index": int(self.cfg.positive_class_index),
            }

        # Multiclass — one-vs-rest per class
        y_true_oh = self._one_hot(y_true, num_classes)
        per_class_auc: Dict[str, float] = {}

        for i in range(num_classes):
            fpr_i, tpr_i, thr_i = roc_curve(y_true_oh[:, i], y_score[:, i])
            auc_i = auc(fpr_i, tpr_i)
            np.save(os.path.join(self.cfg.run_dir, f"roc_fpr_class_{i}.npy"), fpr_i)
            np.save(os.path.join(self.cfg.run_dir, f"roc_tpr_class_{i}.npy"), tpr_i)
            np.save(os.path.join(self.cfg.run_dir, f"roc_thresholds_class_{i}.npy"), thr_i)
            name = self.class_names[i] if self.class_names else f"class_{i}"
            per_class_auc[name] = float(auc_i)

        summary: Dict[str, Any] = {
            "task": "multiclass",
            "multi_class": self.cfg.roc_multi_class,
            "average": self.cfg.roc_average,
            "per_class_auc": per_class_auc,
        }
        for avg in ("macro", "weighted"):
            try:
                summary[f"{avg}_auc"] = float(
                    roc_auc_score(y_true, y_score, multi_class=self.cfg.roc_multi_class, average=avg)
                )
            except Exception as e:
                summary[f"{avg}_auc_error"] = str(e)
        try:
            summary["micro_auc"] = float(
                roc_auc_score(y_true_oh, y_score, multi_class=self.cfg.roc_multi_class, average="micro")
            )
        except Exception as e:
            summary["micro_auc_error"] = str(e)

        return summary

    # -----------------------------------------------------------------------
    # Internals — utilities
    # -----------------------------------------------------------------------

    def _one_hot(self, y: np.ndarray, num_classes: int) -> np.ndarray:
        y = y.astype(np.int32).reshape(-1)
        oh = np.zeros((y.shape[0], num_classes), dtype=np.int32)
        oh[np.arange(y.shape[0]), y] = 1
        return oh

    def _validate_binary_labels(self, y_true: np.ndarray) -> None:
        unique = np.unique(y_true)
        if not np.all(np.isin(unique, [0, 1])):
            raise ValueError(f"Binary ROC expects labels in {{0,1}}. Got: {unique}")

    def _validate_multiclass_labels(self, y_true: np.ndarray, num_classes: int) -> None:
        unique = np.unique(y_true)
        if np.any(unique < 0) or np.any(unique >= num_classes):
            raise ValueError(
                f"Multiclass labels must be in [0, {num_classes - 1}]. Got: {unique}"
            )

    def _safe_save_model(self, model: tf.keras.Model, path: str) -> None:
        folder = os.path.dirname(path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        if not (path.endswith(".keras") or path.endswith(".h5")):
            raise ValueError("Model save path must end with .keras or .h5")
        model.save(path, include_optimizer=False)

    def _safe_write_json(self, path: str, obj: Any) -> None:
        folder = os.path.dirname(path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._make_json_safe(obj), f, indent=2, ensure_ascii=False)

    def _make_json_safe(self, obj: Any) -> Any:
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        if isinstance(obj, (list, tuple)):
            return [self._make_json_safe(x) for x in obj]
        if isinstance(obj, dict):
            return {str(k): self._make_json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        return str(obj)


from typing import Any  # noqa: F811
