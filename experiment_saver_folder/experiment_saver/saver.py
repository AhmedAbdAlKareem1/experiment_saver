import os
import json
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple, Union

import numpy as np
from sklearn.metrics import roc_curve, auc, roc_auc_score

import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping


@dataclass
class ExperimentConfig:
    run_dir: str = "runs/my_experiment_001"
    monitor: str = "val_loss"
    patience: int = 5
    save_best_only: bool = True
    verbose: int = 1

    # ROC/AUC options
    roc_average: str = "macro"          # "macro", "micro", "weighted"
    roc_multi_class: str = "ovr"        # "ovr" or "ovo" (roc_auc_score supports both)
    positive_class_index: int = 1       # used only for binary softmax (2 classes)


class ExperimentSaver:
    """
    A safe utility to save:
      - metrics.csv (per-epoch metrics)
      - history.json
      - best_model.keras (optional, via ModelCheckpoint)
      - final_model.keras
      - ROC curve arrays + AUC on a validation dataset

    Supports:
      - Binary classification:
          * sigmoid output: shape (batch, 1) or (batch,)
          * softmax output: shape (batch, 2)
        Saves: roc_fpr.npy, roc_tpr.npy, roc_thresholds.npy, roc_auc.json
      - Multi-class classification (N classes):
          * softmax output: shape (batch, N)
        Saves:
          * per-class ROC curves:
              roc_fpr_class_{i}.npy, roc_tpr_class_{i}.npy, roc_thresholds_class_{i}.npy
          * AUC summary:
              roc_auc.json with per-class AUC + macro/micro/weighted (when possible)
    """

    def __init__(self, config: ExperimentConfig, class_names: Optional[List[str]] = None):
        self.cfg = config
        self.class_names = class_names  # may be None until inferred

        os.makedirs(self.cfg.run_dir, exist_ok=True)

        self.paths = {
            "metrics_csv": os.path.join(self.cfg.run_dir, "metrics.csv"),
            "history_json": os.path.join(self.cfg.run_dir, "history.json"),
            "best_model": os.path.join(self.cfg.run_dir, "best_model.keras"),
            "final_model": os.path.join(self.cfg.run_dir, "final_model.keras"),
            "roc_auc_json": os.path.join(self.cfg.run_dir, "roc_auc.json"),
            "config_json": os.path.join(self.cfg.run_dir, "config.json"),
            "classes_json": os.path.join(self.cfg.run_dir, "classes.json"),
            "manifest_json": os.path.join(self.cfg.run_dir, "manifest.json"),
        }

        if self.class_names is not None:
            self._safe_write_json(self.paths["classes_json"], {"class_names": self.class_names})

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

    def save_after_fit(
        self,
        model: tf.keras.Model,
        history: tf.keras.callbacks.History,
        val_ds: tf.data.Dataset,
        extra_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        hist_dict = getattr(history, "history", None)
        if not isinstance(hist_dict, dict):
            raise ValueError("Invalid history object: history.history not found or not a dict.")
        self._safe_write_json(self.paths["history_json"], hist_dict)

        self._safe_save_model(model, self.paths["final_model"])

        # Determine number of classes from model output
        num_classes = self._infer_num_classes_from_model(model)

        # Ensure class names exist
        if self.class_names is None:
            self.class_names = [f"class_{i}" for i in range(num_classes)]
            self._safe_write_json(self.paths["classes_json"], {"class_names": self.class_names})
        else:
            if len(self.class_names) != num_classes:
                raise ValueError(
                    f"class_names length ({len(self.class_names)}) does not match "
                    f"model output classes ({num_classes})."
                )

        cfg_payload = {
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

        # Compute ROC/AUC
        y_true, y_score = self._collect_labels_and_scores(model, val_ds, num_classes)

        # Save ROC artifacts
        roc_summary = self._save_roc_artifacts(y_true, y_score, num_classes)

        self._safe_write_json(self.paths["roc_auc_json"], roc_summary)

        manifest = {
            "best_model": os.path.basename(self.paths["best_model"]),
            "final_model": os.path.basename(self.paths["final_model"]),
            "metrics_csv": os.path.basename(self.paths["metrics_csv"]),
            "history_json": os.path.basename(self.paths["history_json"]),
            "roc_auc_json": os.path.basename(self.paths["roc_auc_json"]),
            "classes_json": os.path.basename(self.paths["classes_json"]),
            "config_json": os.path.basename(self.paths["config_json"]),
        }
        self._safe_write_json(self.paths["manifest_json"], manifest)

        return dict(self.paths)

    # -----------------------
    # Internals
    # -----------------------

    def _infer_num_classes_from_model(self, model: tf.keras.Model) -> int:
        out_shape = model.output_shape
        if isinstance(out_shape, list):
            raise ValueError("Multi-output models are not supported.")

        # Examples:
        # (None,)          -> binary sigmoid (treated as 2 classes)
        # (None, 1)        -> binary sigmoid (treated as 2 classes)
        # (None, 2)        -> binary softmax (2 classes)
        # (None, C>2)      -> multiclass
        if len(out_shape) == 1:
            return 2  # (None,) -> binary

        if len(out_shape) == 2:
            c = out_shape[1]
            if c is None:
                raise ValueError("Model output dimension is None; cannot infer classes.")
            c = int(c)

            if c == 1:
                return 2  # sigmoid binary
            return c  # 2 -> binary softmax, >2 -> multiclass

        raise ValueError(f"Unsupported model output shape: {out_shape}")

    def _collect_labels_and_scores(
        self, model: tf.keras.Model, dataset: tf.data.Dataset, num_classes: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
          y_true:
            - binary: shape (N,) with values {0,1}
            - multiclass: shape (N,) with int labels {0..C-1}
          y_score:
            - binary: shape (N,) probabilities for positive class
            - multiclass: shape (N,C) probabilities for each class
        """
        y_true_all: List[np.ndarray] = []
        y_score_all: List[np.ndarray] = []

        for batch in dataset:
            if not isinstance(batch, (tuple, list)) or len(batch) < 2:
                raise ValueError("val_ds must yield (x, y) batches (optionally with sample_weight).")

            x_batch, y_batch = batch[0], batch[1]

            y_true_batch = self._labels_to_int_1d(y_batch, num_classes)
            y_true_all.append(y_true_batch)

            pred = model.predict(x_batch, verbose=0)
            score_batch = self._pred_to_scores(pred, num_classes)
            y_score_all.append(score_batch)

        y_true = np.concatenate(y_true_all, axis=0)

        if num_classes <= 2:
            y_score = np.concatenate(y_score_all, axis=0).astype(np.float32)  # (N,)
            y_score = np.clip(y_score, 0.0, 1.0)
            self._validate_binary_labels(y_true)
            return y_true, y_score

        # multiclass: concatenate (N,C)
        y_score = np.concatenate(y_score_all, axis=0).astype(np.float32)
        # defensive normalization: ensure row sums ~ 1, but do not force if model doesn't output probs
        y_score = np.clip(y_score, 0.0, 1.0)
        self._validate_multiclass_labels(y_true, num_classes)
        return y_true, y_score

    def _labels_to_int_1d(self, y: Any, num_classes: int) -> np.ndarray:
        y_np = y.numpy() if hasattr(y, "numpy") else np.asarray(y)
        y_np = np.asarray(y_np)

        # If one-hot (batch, C)
        if y_np.ndim == 2 and y_np.shape[1] > 1:
            y_int = np.argmax(y_np, axis=1).astype(np.int32)
            return y_int.reshape(-1)

        # If (batch,1) or (batch,)
        y_np = y_np.reshape(-1)

        # If float labels for binary (e.g., 0.0/1.0)
        if np.issubdtype(y_np.dtype, np.floating):
            if num_classes <= 2:
                y_np = (y_np >= 0.5).astype(np.int32)
            else:
                # floats for multiclass labels are suspicious; attempt safe cast
                y_np = np.rint(y_np).astype(np.int32)
        else:
            y_np = y_np.astype(np.int32)

        return y_np

    def _pred_to_scores(self, pred: Any, num_classes: int) -> np.ndarray:
        pred_np = np.asarray(pred)

        # Binary: sigmoid (batch,) or (batch,1)
        if num_classes == 1:
            if pred_np.ndim == 1:
                return pred_np.astype(np.float32)
            if pred_np.ndim == 2 and pred_np.shape[1] == 1:
                return pred_np[:, 0].astype(np.float32)
            raise ValueError(
                f"Unsupported sigmoid prediction shape {pred_np.shape}. Expected (batch,) or (batch,1)."
            )

        # Binary: softmax (batch,2)
        if num_classes == 2:
            if pred_np.ndim == 2 and pred_np.shape[1] == 2:
                idx = int(self.cfg.positive_class_index)
                if idx not in (0, 1):
                    raise ValueError("positive_class_index must be 0 or 1 for binary softmax.")
                return pred_np[:, idx].astype(np.float32)
            if pred_np.ndim == 2 and pred_np.shape[1] == 1:
                return pred_np[:, 0].astype(np.float32)
            if pred_np.ndim == 1:
                return pred_np.astype(np.float32)
            raise ValueError(
                f"Unsupported binary prediction shape {pred_np.shape}. "
                "Expected (batch,), (batch,1) sigmoid, or (batch,2) softmax."
            )

        # Multiclass: (batch,C)
        if pred_np.ndim == 2 and pred_np.shape[1] == num_classes:
            return pred_np.astype(np.float32)

        raise ValueError(
            f"Unsupported multiclass prediction shape {pred_np.shape}. Expected (batch,{num_classes})."
        )

    def _save_roc_artifacts(self, y_true: np.ndarray, y_score: np.ndarray, num_classes: int) -> Dict[str, Any]:
        """
        Binary:
          Save roc_fpr.npy, roc_tpr.npy, roc_thresholds.npy
        Multiclass:
          Save per-class ROC arrays as roc_fpr_class_{i}.npy, etc.
        Also returns a JSON-safe summary of AUC values.
        """
        if num_classes <= 2:
            fpr, tpr, thresholds = roc_curve(y_true, y_score)
            roc_auc_value = auc(fpr, tpr)

            np.save(os.path.join(self.cfg.run_dir, "roc_fpr.npy"), fpr)
            np.save(os.path.join(self.cfg.run_dir, "roc_tpr.npy"), tpr)
            np.save(os.path.join(self.cfg.run_dir, "roc_thresholds.npy"), thresholds)

            return {
                "task": "binary",
                "roc_auc": float(roc_auc_value),
                "positive_class_index": int(self.cfg.positive_class_index),
            }

        # One-vs-Rest ROC per class
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

        # Global AUC using sklearn roc_auc_score
        summary: Dict[str, Any] = {
            "task": "multiclass",
            "multi_class": self.cfg.roc_multi_class,
            "average": self.cfg.roc_average,
            "per_class_auc": per_class_auc,
        }

        # macro/micro/weighted AUC (try; can fail if only one class present in y_true)
        try:
            macro_auc = roc_auc_score(
                y_true, y_score,
                multi_class=self.cfg.roc_multi_class,
                average="macro",
            )
            summary["macro_auc"] = float(macro_auc)
        except Exception as e:
            summary["macro_auc_error"] = str(e)

        try:
            weighted_auc = roc_auc_score(
                y_true, y_score,
                multi_class=self.cfg.roc_multi_class,
                average="weighted",
            )
            summary["weighted_auc"] = float(weighted_auc)
        except Exception as e:
            summary["weighted_auc_error"] = str(e)

        try:
            # micro for multiclass is supported for OVR with one-hot y_true in some sklearn versions,
            # but to be safe, compute micro using one-hot
            micro_auc = roc_auc_score(
                y_true_oh, y_score,
                multi_class=self.cfg.roc_multi_class,
                average="micro",
            )
            summary["micro_auc"] = float(micro_auc)
        except Exception as e:
            summary["micro_auc_error"] = str(e)

        return summary

    def _one_hot(self, y: np.ndarray, num_classes: int) -> np.ndarray:
        y = y.astype(np.int32).reshape(-1)
        oh = np.zeros((y.shape[0], num_classes), dtype=np.int32)
        oh[np.arange(y.shape[0]), y] = 1
        return oh

    def _validate_binary_labels(self, y_true: np.ndarray) -> None:
        unique = np.unique(y_true)
        if not np.all(np.isin(unique, [0, 1])):
            raise ValueError(f"Binary ROC expects labels in {{0,1}}. Got unique labels: {unique}")

    def _validate_multiclass_labels(self, y_true: np.ndarray, num_classes: int) -> None:
        unique = np.unique(y_true)
        if np.any(unique < 0) or np.any(unique >= num_classes):
            raise ValueError(
                f"Multiclass labels must be in [0, {num_classes-1}]. Got unique labels: {unique}"
            )

    def _safe_save_model(self, model: tf.keras.Model, path: str) -> None:
        folder = os.path.dirname(path)
        if folder:
            os.makedirs(folder, exist_ok=True)

        if not (path.endswith(".keras") or path.endswith(".h5")):
            raise ValueError("Model save path must end with .keras or .h5")

        # More portable loads across machines
        model.save(path, include_optimizer=False)

    def _safe_write_json(self, path: str, obj: Any) -> None:
        folder = os.path.dirname(path)
        if folder:
            os.makedirs(folder, exist_ok=True)

        safe_obj = self._make_json_safe(obj)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(safe_obj, f, indent=2, ensure_ascii=False)

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
