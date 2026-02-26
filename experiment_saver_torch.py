import os
import sys
import json
import time
import random
import platform
import subprocess
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
from sklearn.metrics import (
    roc_curve, auc, roc_auc_score,
    confusion_matrix, classification_report,
)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Optional dependency: torchinfo (for model summary)
# ---------------------------------------------------------------------------
try:
    from torchinfo import summary as torchinfo_summary
    _TORCHINFO_AVAILABLE = True
except ImportError:
    _TORCHINFO_AVAILABLE = False


# ===========================================================================
# Config
# ===========================================================================

@dataclass
class ExperimentConfig:
    run_dir: str = "runs/my_experiment_001"
    monitor: str = "val_loss"           # metric name to monitor
    patience: int = 5                   # early stopping patience
    save_best_only: bool = True
    verbose: int = 1

    # ROC / AUC
    roc_average: str = "macro"          # "macro", "micro", "weighted"
    roc_multi_class: str = "ovr"        # "ovr" or "ovo"
    positive_class_index: int = 1       # binary softmax only

    # -----------------------------------------------------------------------
    # Feature flags — all ON by default
    # -----------------------------------------------------------------------

    # Optimizer & Training State
    save_optimizer_state: bool = True
    save_scheduler_state: bool = True
    save_last_epoch: bool = True

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

    # Training Diagnostics
    save_grad_norms: bool = True
    save_lr_history: bool = True
    save_epoch_times: bool = True

    # torchinfo input shape — required for detailed model summary
    # e.g. (1, 3, 224, 224) for a batch of one 3-channel 224x224 image
    model_input_size: Optional[Tuple] = None


# ===========================================================================
# Early Stopping
# ===========================================================================

class EarlyStopping:
    """Simple early stopping tracker."""

    def __init__(self, patience: int = 5, mode: str = "min", verbose: int = 1):
        self.patience = patience
        self.mode = mode
        self.verbose = verbose
        self.best: Optional[float] = None
        self.counter: int = 0
        self.should_stop: bool = False
        self._is_better = (lambda a, b: a < b) if mode == "min" else (lambda a, b: a > b)

    def step(self, value: float) -> bool:
        """Returns True if training should stop."""
        if self.best is None or self._is_better(value, self.best):
            self.best = value
            self.counter = 0
            if self.verbose:
                print(f"EarlyStopping: new best {self.best:.6f}")
            return False
        self.counter += 1
        if self.verbose:
            print(f"EarlyStopping: no improvement ({self.counter}/{self.patience})")
        if self.counter >= self.patience:
            self.should_stop = True
            if self.verbose:
                print("EarlyStopping: stopping.")
            return True
        return False


# ===========================================================================
# ExperimentSaver
# ===========================================================================

class ExperimentSaver:
    """
    Comprehensive experiment saver for PyTorch classification models.

    Saves everything useful for reproducibility, debugging, and resuming:

    ── Core ──────────────────────────────────────────────────────────────────
      metrics.csv                per-epoch metrics (train/val loss, acc, …)
      history.json               same as metrics.csv but as a dict
      best_model.pt              state_dict at best monitored epoch
      final_model.pt             state_dict at end of training
      config.json                ExperimentConfig + extra_config
      classes.json               class name list
      manifest.json              map of all saved file names

    ── Optimizer & Training State ────────────────────────────────────────────
      optimizer_state.pt         optimizer.state_dict()  (resume training)
      scheduler_state.pt         scheduler.state_dict()  (resume training)
      last_epoch.json            epoch index when training ended

    ── Evaluation & Metrics ─────────────────────────────────────────────────
      roc_auc.json               AUC summary (binary or per-class + macro/micro/weighted)
      roc_fpr[_class_i].npy / roc_tpr[…].npy / roc_thresholds[…].npy
      confusion_matrix.npy       shape (C, C) int
      classification_report.json precision / recall / F1 per class
      val_predictions.npy        predicted class index per sample  (N,)
      val_scores.npy             probability scores per sample  (N,) or (N,C)
      val_labels.npy             ground-truth labels  (N,)

    ── Model Metadata ────────────────────────────────────────────────────────
      model_summary.txt          torchinfo or str(model) output
      param_count.json           total / trainable / frozen parameter counts
      model_architecture.json    layer names + types + param counts

    ── Reproducibility ───────────────────────────────────────────────────────
      random_seeds.json          torch / numpy / python random seeds
      environment.json           Python, PyTorch, CUDA, sklearn versions
      git_commit.json            git hash, branch, dirty-flag

    ── Training Diagnostics ─────────────────────────────────────────────────
      grad_norms.csv             gradient norm logged per epoch
      lr_history.csv             learning rate per epoch (all param groups)
      epoch_times.csv            wall-clock seconds per epoch

    ─────────────────────────────────────────────────────────────────────────

    Typical training loop
    ─────────────────────
        saver = ExperimentSaver(config, class_names=["cat", "dog"])
        early_stop = saver.make_early_stopping()

        for epoch in range(max_epochs):
            t0 = time.time()

            train_loss = train_one_epoch(model, train_loader, optimizer)
            val_loss, val_acc = validate(model, val_loader)

            saver.log_lr(optimizer, epoch=epoch)      # before or after scheduler.step()
            scheduler.step()

            metrics = {"train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc}
            saver.log_epoch(epoch, metrics, model=model, epoch_time=time.time() - t0)

            if saver.should_save_best(metrics[saver.cfg.monitor]):
                saver.save_best_checkpoint(model)

            if early_stop.step(metrics[saver.cfg.monitor]):
                break

        saved_paths = saver.save_after_fit(
            model,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            last_epoch=epoch,
        )
    """

    def __init__(
        self,
        config: ExperimentConfig,
        class_names: Optional[List[str]] = None,
    ):
        self.cfg = config
        self.class_names = class_names
        self._csv_header_written = False
        self._history: Dict[str, List[float]] = {}
        self._grad_norm_rows: List[str] = []
        self._lr_rows: List[str] = []
        self._epoch_time_rows: List[str] = []
        self._best_monitor_value: Optional[float] = None
        self._monitor_mode = "min" if "loss" in self.cfg.monitor else "max"

        os.makedirs(self.cfg.run_dir, exist_ok=True)

        self.paths: Dict[str, str] = {
            # Core
            "metrics_csv":              os.path.join(self.cfg.run_dir, "metrics.csv"),
            "history_json":             os.path.join(self.cfg.run_dir, "history.json"),
            "best_model":               os.path.join(self.cfg.run_dir, "best_model.pt"),
            "final_model":              os.path.join(self.cfg.run_dir, "final_model.pt"),
            "roc_auc_json":             os.path.join(self.cfg.run_dir, "roc_auc.json"),
            "config_json":              os.path.join(self.cfg.run_dir, "config.json"),
            "classes_json":             os.path.join(self.cfg.run_dir, "classes.json"),
            "manifest_json":            os.path.join(self.cfg.run_dir, "manifest.json"),
            # Optimizer & Training State
            "optimizer_state":          os.path.join(self.cfg.run_dir, "optimizer_state.pt"),
            "scheduler_state":          os.path.join(self.cfg.run_dir, "scheduler_state.pt"),
            "last_epoch_json":          os.path.join(self.cfg.run_dir, "last_epoch.json"),
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
            # Training Diagnostics
            "grad_norms_csv":           os.path.join(self.cfg.run_dir, "grad_norms.csv"),
            "lr_history_csv":           os.path.join(self.cfg.run_dir, "lr_history.csv"),
            "epoch_times_csv":          os.path.join(self.cfg.run_dir, "epoch_times.csv"),
        }

        if self.class_names is not None:
            self._safe_write_json(self.paths["classes_json"], {"class_names": self.class_names})

    # -----------------------------------------------------------------------
    # Public helpers  (call during training loop)
    # -----------------------------------------------------------------------

    def make_early_stopping(self) -> EarlyStopping:
        """Returns an EarlyStopping tracker configured from cfg."""
        return EarlyStopping(
            patience=self.cfg.patience,
            mode=self._monitor_mode,
            verbose=self.cfg.verbose,
        )

    def should_save_best(self, monitor_value: float) -> bool:
        """
        Returns True when monitor_value is a new best.
        Tracks the best value internally — call once per epoch.
        """
        is_better = (
            self._best_monitor_value is None
            or (self._monitor_mode == "min" and monitor_value < self._best_monitor_value)
            or (self._monitor_mode == "max" and monitor_value > self._best_monitor_value)
        )
        if is_better:
            self._best_monitor_value = monitor_value
        return is_better

    def log_epoch(
        self,
        epoch: int,
        metrics: Dict[str, float],
        model: Optional[nn.Module] = None,
        epoch_time: Optional[float] = None,
    ) -> None:
        """
        Call once per epoch after computing train/val metrics.
          - Appends a row to metrics.csv
          - Accumulates internal history dict
          - Logs gradient norm  (if model given and save_grad_norms=True)
          - Logs epoch wall-clock time  (if epoch_time given and save_epoch_times=True)
        """
        row = {"epoch": epoch, **metrics}

        # metrics.csv
        mode = "a" if self._csv_header_written else "w"
        with open(self.paths["metrics_csv"], mode, encoding="utf-8") as f:
            if not self._csv_header_written:
                f.write(",".join(str(k) for k in row.keys()) + "\n")
                self._csv_header_written = True
            f.write(",".join(str(v) for v in row.values()) + "\n")

        # internal history
        for k, v in metrics.items():
            self._history.setdefault(k, []).append(float(v))

        # gradient norm (computed from whatever .grad is currently populated)
        if self.cfg.save_grad_norms and model is not None:
            gnorm = self._compute_grad_norm(model)
            if not self._grad_norm_rows:
                self._grad_norm_rows.append("epoch,grad_norm")
            self._grad_norm_rows.append(f"{epoch},{gnorm:.6f}")

        # epoch time
        if self.cfg.save_epoch_times and epoch_time is not None:
            if not self._epoch_time_rows:
                self._epoch_time_rows.append("epoch,seconds")
            self._epoch_time_rows.append(f"{epoch},{epoch_time:.3f}")

        if self.cfg.verbose:
            metric_str = "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
            print(f"Epoch {epoch:04d}  {metric_str}")

    def log_lr(
        self,
        optimizer: torch.optim.Optimizer,
        epoch: Optional[int] = None,
    ) -> None:
        """
        Record the current learning rate(s) from all param groups.
        Call once per epoch, ideally before scheduler.step().
        If epoch is None, uses the current row count as the epoch index.
        """
        if not self.cfg.save_lr_history:
            return
        if not self._lr_rows:
            self._lr_rows.append("epoch,lr")
        ep = epoch if epoch is not None else len(self._lr_rows) - 1
        lrs = [pg["lr"] for pg in optimizer.param_groups]
        lr_str = ",".join(f"{lr:.8f}" for lr in lrs)
        self._lr_rows.append(f"{ep},{lr_str}")

    def save_best_checkpoint(self, model: nn.Module) -> None:
        """Save model state_dict as best_model.pt."""
        torch.save(model.state_dict(), self.paths["best_model"])
        if self.cfg.verbose:
            print(f"Saved best checkpoint → {self.paths['best_model']}")

    # -----------------------------------------------------------------------
    # Main save call
    # -----------------------------------------------------------------------

    def save_after_fit(
        self,
        model: nn.Module,
        val_loader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler=None,
        last_epoch: Optional[int] = None,
        extra_config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
    ) -> Dict[str, str]:
        """
        Call once after training is complete.

        Parameters
        ----------
        model        : trained PyTorch model
        val_loader   : DataLoader yielding (x, y) — needed for all eval artifacts
        optimizer    : to save optimizer state
        scheduler    : to save LR scheduler state
        last_epoch   : int epoch index when training stopped
        extra_config : extra key-value pairs added to config.json
        device       : inference device (defaults to model's first param device)
        """
        # Resolve device
        if device is None:
            try:
                device = next(model.parameters()).device
            except StopIteration:
                device = torch.device("cpu")

        # ── Core ──────────────────────────────────────────────────────────
        self._safe_write_json(self.paths["history_json"], self._history)
        torch.save(model.state_dict(), self.paths["final_model"])

        num_classes = self._infer_num_classes_from_model(model)

        if self.class_names is None:
            self.class_names = [f"class_{i}" for i in range(num_classes)]
            self._safe_write_json(self.paths["classes_json"], {"class_names": self.class_names})
        elif len(self.class_names) != num_classes:
            raise ValueError(
                f"class_names length ({len(self.class_names)}) != "
                f"inferred num_classes ({num_classes})."
            )

        cfg_payload: Dict[str, Any] = {
            "run_dir":          self.cfg.run_dir,
            "monitor":          self.cfg.monitor,
            "patience":         self.cfg.patience,
            "save_best_only":   self.cfg.save_best_only,
            "class_names":      self.class_names,
            "num_classes":      num_classes,
            "roc_average":      self.cfg.roc_average,
            "roc_multi_class":  self.cfg.roc_multi_class,
        }
        if extra_config:
            cfg_payload.update(self._make_json_safe(extra_config))
        self._safe_write_json(self.paths["config_json"], cfg_payload)

        # ── Optimizer & Training State ─────────────────────────────────────
        if self.cfg.save_optimizer_state and optimizer is not None:
            torch.save(optimizer.state_dict(), self.paths["optimizer_state"])

        if self.cfg.save_scheduler_state and scheduler is not None:
            torch.save(scheduler.state_dict(), self.paths["scheduler_state"])

        if self.cfg.save_last_epoch and last_epoch is not None:
            self._safe_write_json(
                self.paths["last_epoch_json"], {"last_epoch": int(last_epoch)}
            )

        # ── Collect val predictions (shared by all eval saves below) ──────
        y_true, y_score, y_pred = None, None, None
        if val_loader is not None:
            y_true, y_score, y_pred = self._collect_labels_scores_preds(
                model, val_loader, num_classes, device
            )

        # ── Evaluation & Metrics ──────────────────────────────────────────
        if y_true is not None:
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
        else:
            self._safe_write_json(
                self.paths["roc_auc_json"],
                {"note": "val_loader not provided; ROC/AUC and eval metrics not computed."},
            )

        # ── Model Metadata ────────────────────────────────────────────────
        if self.cfg.save_model_summary:
            self._save_model_summary(model)

        if self.cfg.save_param_count:
            total = sum(p.numel() for p in model.parameters())
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            self._safe_write_json(
                self.paths["param_count_json"],
                {"total": total, "trainable": trainable, "frozen": total - trainable},
            )

        if self.cfg.save_model_architecture:
            arch = {
                name: {
                    "type":   type(module).__name__,
                    "params": sum(p.numel() for p in module.parameters()),
                }
                for name, module in model.named_modules()
            }
            self._safe_write_json(self.paths["model_architecture_json"], arch)

        # ── Reproducibility ───────────────────────────────────────────────
        if self.cfg.save_random_seeds:
            self._safe_write_json(
                self.paths["random_seeds_json"],
                {
                    "torch_initial_seed":   torch.initial_seed(),
                    "numpy_seed":           int(np.random.get_state()[1][0]),
                    "python_random_state":  random.getstate()[1][0],
                    "cuda_available":       torch.cuda.is_available(),
                    "cudnn_deterministic":  torch.backends.cudnn.deterministic,
                    "cudnn_benchmark":      torch.backends.cudnn.benchmark,
                },
            )

        if self.cfg.save_environment:
            self._save_environment()

        if self.cfg.save_git_commit:
            self._save_git_commit()

        # ── Training Diagnostics ──────────────────────────────────────────
        if self.cfg.save_grad_norms and self._grad_norm_rows:
            with open(self.paths["grad_norms_csv"], "w", encoding="utf-8") as f:
                f.write("\n".join(self._grad_norm_rows) + "\n")

        if self.cfg.save_lr_history and self._lr_rows:
            with open(self.paths["lr_history_csv"], "w", encoding="utf-8") as f:
                f.write("\n".join(self._lr_rows) + "\n")

        if self.cfg.save_epoch_times and self._epoch_time_rows:
            with open(self.paths["epoch_times_csv"], "w", encoding="utf-8") as f:
                f.write("\n".join(self._epoch_time_rows) + "\n")

        # ── Manifest ──────────────────────────────────────────────────────
        manifest = {k: os.path.basename(v) for k, v in self.paths.items()}
        self._safe_write_json(self.paths["manifest_json"], manifest)

        if self.cfg.verbose:
            print(f"\nAll artifacts saved to: {self.cfg.run_dir}")

        return dict(self.paths)

    # -----------------------------------------------------------------------
    # Internals — model introspection
    # -----------------------------------------------------------------------

    def _infer_num_classes_from_model(self, model: nn.Module) -> int:
        """Walks layers in reverse to find the last Linear/Conv and reads its output size."""
        for layer in reversed(list(model.modules())):
            if isinstance(layer, nn.Linear):
                out = layer.out_features
                return 2 if out == 1 else int(out)   # sigmoid (1) → treat as binary (2)
            if isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                out = layer.out_channels
                return 2 if out == 1 else int(out)
        raise ValueError(
            "Could not infer num_classes from model layers. "
            "Pass class_names= explicitly to ExperimentSaver()."
        )

    def _save_model_summary(self, model: nn.Module) -> None:
        if _TORCHINFO_AVAILABLE and self.cfg.model_input_size is not None:
            try:
                s = torchinfo_summary(
                    model,
                    input_size=self.cfg.model_input_size,
                    verbose=0,
                )
                summary_str = str(s)
            except Exception:
                summary_str = str(model)
        else:
            summary_str = str(model)
            if _TORCHINFO_AVAILABLE and self.cfg.model_input_size is None:
                summary_str = (
                    "# torchinfo is available but model_input_size was not set.\n"
                    "# Set config.model_input_size = (1, C, H, W) for a detailed summary.\n\n"
                ) + summary_str
        with open(self.paths["model_summary"], "w", encoding="utf-8") as f:
            f.write(summary_str)

    def _save_environment(self) -> None:
        env: Dict[str, Any] = {
            "python_version": sys.version,
            "platform":       platform.platform(),
            "torch_version":  torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version":   torch.version.cuda if torch.cuda.is_available() else None,
            "cudnn_version":  str(torch.backends.cudnn.version()) if torch.cuda.is_available() else None,
        }
        for pkg in ("numpy", "sklearn", "torchinfo"):
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

    @torch.no_grad()
    def _collect_labels_scores_preds(
        self,
        model: nn.Module,
        loader: DataLoader,
        num_classes: int,
        device: torch.device,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns
        -------
        y_true : (N,)        int32 ground-truth labels
        y_score: (N,) or (N,C)  float32 probabilities
        y_pred : (N,)        int32 predicted class indices
        """
        model.eval()
        y_true_all, y_score_all = [], []

        for batch in loader:
            if not isinstance(batch, (tuple, list)) or len(batch) < 2:
                raise ValueError("val_loader must yield (x, y) batches.")
            x_batch, y_batch = batch[0], batch[1]
            if isinstance(x_batch, torch.Tensor):
                x_batch = x_batch.to(device)

            logits = model(x_batch)
            pred_np = logits.cpu().numpy() if isinstance(logits, torch.Tensor) else np.asarray(logits)

            y_true_all.append(self._labels_to_int_1d(y_batch, num_classes))
            y_score_all.append(self._pred_to_scores(pred_np, num_classes))

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
        y_np = y.cpu().numpy() if isinstance(y, torch.Tensor) else np.asarray(y)
        if y_np.ndim == 2 and y_np.shape[1] > 1:
            return np.argmax(y_np, axis=1).astype(np.int32)
        y_np = y_np.reshape(-1)
        if np.issubdtype(y_np.dtype, np.floating):
            y_np = (y_np >= 0.5).astype(np.int32) if num_classes <= 2 else np.rint(y_np).astype(np.int32)
        else:
            y_np = y_np.astype(np.int32)
        return y_np

    def _pred_to_scores(self, pred_np: np.ndarray, num_classes: int) -> np.ndarray:
        if num_classes == 2:
            if pred_np.ndim == 1:
                return pred_np.astype(np.float32)
            if pred_np.ndim == 2 and pred_np.shape[1] == 1:
                return pred_np[:, 0].astype(np.float32)
            if pred_np.ndim == 2 and pred_np.shape[1] == 2:
                return pred_np.astype(np.float32)       # keep both; argmax → y_pred
            raise ValueError(f"Binary prediction shape {pred_np.shape} not supported.")
        if pred_np.ndim == 2 and pred_np.shape[1] == num_classes:
            return pred_np.astype(np.float32)
        raise ValueError(
            f"Multiclass prediction shape {pred_np.shape} not supported. "
            f"Expected (batch, {num_classes})."
        )

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
                    roc_auc_score(
                        y_true, y_score,
                        multi_class=self.cfg.roc_multi_class,
                        average=avg,
                    )
                )
            except Exception as e:
                summary[f"{avg}_auc_error"] = str(e)
        try:
            summary["micro_auc"] = float(
                roc_auc_score(
                    y_true_oh, y_score,
                    multi_class=self.cfg.roc_multi_class,
                    average="micro",
                )
            )
        except Exception as e:
            summary["micro_auc_error"] = str(e)

        return summary

    # -----------------------------------------------------------------------
    # Internals — diagnostics
    # -----------------------------------------------------------------------

    def _compute_grad_norm(self, model: nn.Module) -> float:
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.detach().norm(2).item() ** 2
        return total_norm ** 0.5

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
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        return str(obj)


from typing import Any  # noqa: F811
