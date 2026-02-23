import os
import json
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
from sklearn.metrics import roc_curve, auc

import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping


@dataclass
class ExperimentConfig:
    run_dir: str = "runs/my_experiment_001"
    monitor: str = "val_loss"
    patience: int = 5
    save_best_only: bool = True
    verbose: int = 1


class ExperimentSaver:
    """
    A safe utility to save:
      - metrics.csv (per-epoch metrics)
      - history.json
      - best_model.keras (optional, via ModelCheckpoint)
      - final_model.keras
      - ROC curve arrays + AUC on a validation dataset

    Designed for binary classification models.
    Supports:
      - sigmoid output: shape (batch, 1)
      - softmax output: shape (batch, 2) (we use positive-class probability)
    """

    def __init__(self, config: ExperimentConfig, class_names: Optional[List[str]] = None):
        self.cfg = config
        self.class_names = class_names or ["class_0", "class_1"]

        os.makedirs(self.cfg.run_dir, exist_ok=True)

        self.paths = {
            "metrics_csv": os.path.join(self.cfg.run_dir, "metrics.csv"),
            "history_json": os.path.join(self.cfg.run_dir, "history.json"),
            "best_model": os.path.join(self.cfg.run_dir, "best_model.keras"),
            "final_model": os.path.join(self.cfg.run_dir, "final_model.keras"),
            "roc_fpr": os.path.join(self.cfg.run_dir, "roc_fpr.npy"),
            "roc_tpr": os.path.join(self.cfg.run_dir, "roc_tpr.npy"),
            "roc_thresholds": os.path.join(self.cfg.run_dir, "roc_thresholds.npy"),
            "roc_auc_json": os.path.join(self.cfg.run_dir, "roc_auc.json"),
            "config_json": os.path.join(self.cfg.run_dir, "config.json"),
            "classes_json": os.path.join(self.cfg.run_dir, "classes.json"),
        }

        # Save class names early (helpful for later plotting/eval tools)
        self._safe_write_json(self.paths["classes_json"], {"class_names": self.class_names})

    def callbacks(self) -> List[tf.keras.callbacks.Callback]:
        """
        Returns safe callbacks to pass into model.fit(...).
        """
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
        """
        Call after training finishes.
        Saves history, final model, ROC curve, and config.
        Returns dict of saved file paths.
        """
        # 1) Save history.json
        hist_dict = getattr(history, "history", None)
        if not isinstance(hist_dict, dict):
            raise ValueError("Invalid history object: history.history not found or not a dict.")
        self._safe_write_json(self.paths["history_json"], hist_dict)

        # 2) Save final model
        self._safe_save_model(model, self.paths["final_model"])

        # 3) Save config metadata (optional)
        cfg_payload = {
            "run_dir": self.cfg.run_dir,
            "monitor": self.cfg.monitor,
            "patience": self.cfg.patience,
            "save_best_only": self.cfg.save_best_only,
            "class_names": self.class_names,
        }
        if extra_config:
            # only merge JSON-serializable values (best effort)
            cfg_payload.update(self._make_json_safe(extra_config))
        self._safe_write_json(self.paths["config_json"], cfg_payload)

        # 4) Compute & save ROC
        y_true, y_prob = self._collect_labels_and_probs(model, val_ds)
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        roc_auc_value = auc(fpr, tpr)

        np.save(self.paths["roc_fpr"], fpr)
        np.save(self.paths["roc_tpr"], tpr)
        np.save(self.paths["roc_thresholds"], thresholds)
        self._safe_write_json(self.paths["roc_auc_json"], {"roc_auc": float(roc_auc_value)})

        return dict(self.paths)

    # -----------------------
    # Internals (safe helpers)
    # -----------------------

    def _collect_labels_and_probs(
            self, model: tf.keras.Model, dataset: tf.data.Dataset
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runs inference on a dataset yielding (x, y).
        Returns:
          y_true: shape (N,) values 0/1
          y_prob: shape (N,) probabilities for positive class
        """
        y_true_all: List[np.ndarray] = []
        y_prob_all: List[np.ndarray] = []

        for batch in dataset:
            if not isinstance(batch, (tuple, list)) or len(batch) < 2:
                raise ValueError("val_ds must yield (x, y) batches.")

            x_batch, y_batch = batch[0], batch[1]

            # Convert labels to numpy 1D
            y_np = self._to_1d_numpy(y_batch)
            y_true_all.append(y_np)

            # Predict probabilities
            pred = model.predict(x_batch, verbose=0)
            prob = self._extract_positive_probability(pred)
            y_prob_all.append(prob)

        y_true = np.concatenate(y_true_all, axis=0)
        y_prob = np.concatenate(y_prob_all, axis=0)

        # Defensive cleanup: clip probabilities
        y_prob = np.clip(y_prob, 0.0, 1.0)

        return y_true, y_prob

    def _extract_positive_probability(self, pred: Any) -> np.ndarray:
        """
        Accepts model.predict output and returns P(positive) as 1D numpy array.
        Supports:
          - sigmoid: (batch, 1) or (batch,)
          - softmax: (batch, 2) -> use column 1
        """
        pred_np = np.asarray(pred)

        if pred_np.ndim == 1:
            # (batch,)
            return pred_np.astype(np.float32)

        if pred_np.ndim == 2:
            if pred_np.shape[1] == 1:
                # (batch, 1)
                return pred_np[:, 0].astype(np.float32)
            if pred_np.shape[1] == 2:
                # (batch, 2) softmax
                return pred_np[:, 1].astype(np.float32)

        raise ValueError(
            f"Unsupported prediction shape {pred_np.shape}. "
            "Expected (batch,), (batch,1) sigmoid, or (batch,2) softmax for binary."
        )

    def _to_1d_numpy(self, y: Any) -> np.ndarray:
        """
        Converts labels tensor/array to 1D numpy array of 0/1.
        """
        y_np = y.numpy() if hasattr(y, "numpy") else np.asarray(y)

        # If labels come as (batch, 1), flatten them
        y_np = np.asarray(y_np).reshape(-1)

        # If labels are float, round safely to 0/1
        if np.issubdtype(y_np.dtype, np.floating):
            y_np = (y_np >= 0.5).astype(np.int32)
        else:
            y_np = y_np.astype(np.int32)

        return y_np

    def _safe_save_model(self, model: tf.keras.Model, path: str) -> None:
        """
        Saves Keras model safely. Raises clear error if path is invalid.
        """
        folder = os.path.dirname(path)
        if folder:
            os.makedirs(folder, exist_ok=True)

        # Prefer .keras
        if not (path.endswith(".keras") or path.endswith(".h5")):
            raise ValueError("Model save path must end with .keras or .h5")

        model.save(path)

    def _safe_write_json(self, path: str, obj: Any) -> None:
        """
        Writes JSON with UTF-8 and indentation. Ensures directory exists.
        """
        folder = os.path.dirname(path)
        if folder:
            os.makedirs(folder, exist_ok=True)

        safe_obj = self._make_json_safe(obj)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(safe_obj, f, indent=2, ensure_ascii=False)

    def _make_json_safe(self, obj: Any) -> Any:
        """
        Best-effort conversion to JSON-serializable types.
        """
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj

        if isinstance(obj, (list, tuple)):
            return [self._make_json_safe(x) for x in obj]

        if isinstance(obj, dict):
            return {str(k): self._make_json_safe(v) for k, v in obj.items()}

        # numpy types
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()

        # fallback
        return str(obj)


