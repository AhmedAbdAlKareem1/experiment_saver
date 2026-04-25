"""PyTorch experiment saver implementation."""
from __future__ import annotations

import os
import platform
import random
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ._common import (
    build_manifest,
    bundle_run_dir,
    make_json_safe as _base_make_json_safe,
    safe_write_json,
    save_git_commit,
    save_roc_artifacts,
    validate_binary_labels,
    validate_multiclass_labels,
)

__version__ = "0.2.0"


# ---------------------------------------------------------------------------
# Torch-aware JSON serialiser (extends the base version)
# ---------------------------------------------------------------------------

def _make_json_safe(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    return _base_make_json_safe(obj)


def _safe_write_json(path: str, obj: Any) -> None:
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    import json
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_make_json_safe(obj), f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Optional torchinfo
# ---------------------------------------------------------------------------

try:
    from torchinfo import summary as _torchinfo_summary
    _TORCHINFO_AVAILABLE = True
except ImportError:
    _TORCHINFO_AVAILABLE = False


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
    roc_average: str = "macro"
    roc_multi_class: str = "ovr"
    positive_class_index: int = 1

    # Model outputs
    model_outputs_logits: bool = False
    """
    Set to ``True`` if the model returns raw logits (no final activation).

    When ``True``, sigmoid is applied for binary (1-output) models and
    softmax for multiclass / binary-softmax models before computing scores.

    The PyTorch convention is to train with logits (``BCEWithLogitsLoss``,
    ``CrossEntropyLoss``) and add the activation only at inference.  If your
    model *already* contains a ``Sigmoid`` or ``Softmax`` layer, leave this
    as ``False`` (the default).
    """

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

    # torchinfo input shape for detailed model summary, e.g. (1, 3, 224, 224)
    model_input_size: Optional[Tuple] = None

    # Bundling
    bundle_artifacts: bool = False
    bundle_filename: str = "artifacts.zip"
    bundle_exclude_exts: Tuple[str, ...] = (".keras", ".h5", ".pt", ".pth")
    bundle_delete_originals: bool = False


# ===========================================================================
# Early Stopping
# ===========================================================================

class EarlyStopping:
    """Lightweight early-stopping tracker for PyTorch training loops."""

    def __init__(self, patience: int = 5, mode: str = "min", verbose: int = 1) -> None:
        self.patience = patience
        self.mode = mode
        self.verbose = verbose
        self.best: Optional[float] = None
        self.counter: int = 0
        self.should_stop: bool = False
        self._is_better = (lambda a, b: a < b) if mode == "min" else (lambda a, b: a > b)

    def step(self, value: float) -> bool:
        """Return ``True`` when training should stop."""
        if self.best is None or self._is_better(value, self.best):
            self.best = value
            self.counter = 0
            if self.verbose:
                print(f"EarlyStopping: new best = {self.best:.6f}")
            return False
        self.counter += 1
        if self.verbose:
            print(f"EarlyStopping: no improvement ({self.counter}/{self.patience})")
        if self.counter >= self.patience:
            self.should_stop = True
            if self.verbose:
                print("EarlyStopping: stopping training.")
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
      metrics.csv                per-epoch train/val metrics
      history.json               same as metrics.csv but as a dict
      best_model.pt              state_dict at best monitored epoch
      final_model.pt             state_dict at end of training
      config.json                ExperimentConfig + extra_config
      classes.json               class name list
      manifest.json              rich metadata + artifact file map

    ── Optimizer & Training State ────────────────────────────────────────────
      optimizer_state.pt         optimizer.state_dict()
      scheduler_state.pt         scheduler.state_dict()
      last_epoch.json            epoch index when training ended

    ── Evaluation & Metrics ─────────────────────────────────────────────────
      roc_auc.json               AUC summary
      roc_fpr[_class_i].npy / roc_tpr[…].npy / roc_thresholds[…].npy
      confusion_matrix.npy       (C, C) int
      classification_report.json precision / recall / F1 per class
      val_predictions.npy        predicted class index (N,)
      val_scores.npy             probability scores (N,) or (N, C)
      val_labels.npy             ground-truth labels (N,)

    ── Model Metadata ────────────────────────────────────────────────────────
      model_summary.txt          torchinfo or str(model)
      param_count.json           total / trainable / frozen parameter counts
      model_architecture.json    module names + types + param counts

    ── Reproducibility ───────────────────────────────────────────────────────
      random_seeds.json          torch / numpy / python random seeds
      environment.json           Python, PyTorch, CUDA, sklearn versions
      git_commit.json            git hash, branch, dirty-flag

    ── Training Diagnostics ─────────────────────────────────────────────────
      grad_norms.csv             gradient norm per epoch
      lr_history.csv             learning rate(s) per epoch
      epoch_times.csv            wall-clock seconds per epoch

    Typical training loop::

        saver = ExperimentSaver(config, class_names=["cat", "dog"])
        early_stop = saver.make_early_stopping()

        for epoch in range(max_epochs):
            t0 = time.time()
            train_loss = train_one_epoch(model, train_loader, optimizer)
            val_loss, val_acc = validate(model, val_loader)

            saver.log_lr(optimizer, epoch=epoch)
            scheduler.step()

            metrics = {"train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc}
            saver.log_epoch(epoch, metrics, model=model, epoch_time=time.time() - t0)

            if saver.should_save_best(metrics[saver.cfg.monitor]):
                saver.save_best_checkpoint(model)

            if early_stop.step(metrics[saver.cfg.monitor]):
                break

        saved_paths = saver.save_after_fit(
            model, val_loader=val_loader,
            optimizer=optimizer, scheduler=scheduler, last_epoch=epoch,
        )
    """

    def __init__(
        self,
        config: ExperimentConfig,
        class_names: Optional[List[str]] = None,
    ) -> None:
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
            safe_write_json(self.paths["classes_json"], {"class_names": self.class_names})

    # -----------------------------------------------------------------------
    # Training-loop helpers
    # -----------------------------------------------------------------------

    def make_early_stopping(self) -> EarlyStopping:
        """Return an :class:`EarlyStopping` tracker configured from ``cfg``."""
        return EarlyStopping(
            patience=self.cfg.patience,
            mode=self._monitor_mode,
            verbose=self.cfg.verbose,
        )

    def should_save_best(self, monitor_value: float) -> bool:
        """
        Return ``True`` when *monitor_value* is a new best.

        Call once per epoch after computing validation metrics.
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
        Record one epoch of metrics.

        Appends a row to ``metrics.csv``, accumulates the internal history
        dict, optionally logs the gradient norm, and optionally records the
        epoch wall-clock time.

        Parameters
        ----------
        epoch:
            Zero-based epoch index.
        metrics:
            Dict of metric names to float values (e.g. ``{"val_loss": 0.5}``).
        model:
            Provide to log gradient norm (requires ``save_grad_norms=True``).
        epoch_time:
            Wall-clock seconds for this epoch (requires ``save_epoch_times=True``).
        """
        row = {"epoch": epoch, **metrics}

        mode = "a" if self._csv_header_written else "w"
        with open(self.paths["metrics_csv"], mode, encoding="utf-8") as f:
            if not self._csv_header_written:
                f.write(",".join(str(k) for k in row.keys()) + "\n")
                self._csv_header_written = True
            f.write(",".join(str(v) for v in row.values()) + "\n")

        for k, v in metrics.items():
            self._history.setdefault(k, []).append(float(v))

        if self.cfg.save_grad_norms and model is not None:
            gnorm = self._compute_grad_norm(model)
            if not self._grad_norm_rows:
                self._grad_norm_rows.append("epoch,grad_norm")
            self._grad_norm_rows.append(f"{epoch},{gnorm:.6f}")

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

        Call once per epoch, ideally *before* ``scheduler.step()``.
        """
        if not self.cfg.save_lr_history:
            return
        ep = epoch if epoch is not None else max(len(self._lr_rows) - 1, 0)
        lrs = [pg["lr"] for pg in optimizer.param_groups]
        if not self._lr_rows:
            header = ["epoch"] + [f"lr_group{i}" for i in range(len(lrs))]
            self._lr_rows.append(",".join(header))
        lr_str = ",".join(f"{float(lr):.8f}" for lr in lrs)
        self._lr_rows.append(f"{ep},{lr_str}")

    def save_best_checkpoint(self, model: nn.Module) -> None:
        """Save ``model.state_dict()`` as ``best_model.pt``."""
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "monitor": self.cfg.monitor,
            "best_monitor_value": self._best_monitor_value,
            "class_names": self.class_names,
        }
        torch.save(checkpoint, self.paths["best_model"])
        if self.cfg.verbose:
            print(f"Saved best checkpoint → {self.paths['best_model']}")

    # -----------------------------------------------------------------------
    # Main save
    # -----------------------------------------------------------------------

    def save_after_fit(
        self,
        model: nn.Module,
        val_loader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Any = None,
        last_epoch: Optional[int] = None,
        extra_config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
    ) -> Dict[str, str]:
        """
        Save all artifacts after training is complete.

        Parameters
        ----------
        model:
            The trained PyTorch model.
        val_loader:
            ``DataLoader`` yielding ``(x, y)`` — required for eval artifacts.
        optimizer:
            To save optimizer state for resuming.
        scheduler:
            To save LR scheduler state for resuming.
        last_epoch:
            Epoch index when training stopped.
        extra_config:
            Extra key/value pairs merged into ``config.json``.
        device:
            Inference device; auto-detected from model parameters if ``None``.

        Returns
        -------
        dict
            Mapping of artifact keys to their absolute file paths.
        """
        if device is None:
            try:
                device = next(model.parameters()).device
            except StopIteration:
                device = torch.device("cpu")

        # ── History ───────────────────────────────────────────────────────
        safe_write_json(self.paths["history_json"], self._history)

        # ── Final model checkpoint ────────────────────────────────────────
        final_ckpt = {
            "model_state_dict": model.state_dict(),
            "class_names": self.class_names,
            "monitor": self.cfg.monitor,
            "best_monitor_value": self._best_monitor_value,
            "last_epoch": last_epoch,
        }
        torch.save(final_ckpt, self.paths["final_model"])

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
            "run_dir":           self.cfg.run_dir,
            "monitor":           self.cfg.monitor,
            "patience":          self.cfg.patience,
            "save_best_only":    self.cfg.save_best_only,
            "class_names":       self.class_names,
            "num_classes":       num_classes,
            "roc_average":       self.cfg.roc_average,
            "roc_multi_class":   self.cfg.roc_multi_class,
            "model_outputs_logits": self.cfg.model_outputs_logits,
        }
        if extra_config:
            cfg_payload.update(_make_json_safe(extra_config))
        safe_write_json(self.paths["config_json"], cfg_payload)

        # ── Optimizer & scheduler state ───────────────────────────────────
        if self.cfg.save_optimizer_state and optimizer is not None:
            torch.save(optimizer.state_dict(), self.paths["optimizer_state"])

        if self.cfg.save_scheduler_state and scheduler is not None:
            torch.save(scheduler.state_dict(), self.paths["scheduler_state"])

        if self.cfg.save_last_epoch and last_epoch is not None:
            safe_write_json(self.paths["last_epoch_json"], {"last_epoch": int(last_epoch)})

        # ── Validation predictions ────────────────────────────────────────
        y_true, y_score, y_pred = None, None, None
        if val_loader is not None:
            y_true, y_score, y_pred = self._collect_predictions(
                model, val_loader, num_classes, device
            )

        # ── Evaluation ───────────────────────────────────────────────────
        if y_true is not None:
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
        else:
            safe_write_json(
                self.paths["roc_auc_json"],
                {"note": "val_loader not provided — ROC/AUC not computed."},
            )

        # ── Model Metadata ────────────────────────────────────────────────
        if self.cfg.save_model_summary:
            self._save_model_summary(model)

        if self.cfg.save_param_count:
            total = sum(p.numel() for p in model.parameters())
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            safe_write_json(
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
            safe_write_json(self.paths["model_architecture_json"], arch)

        # ── Reproducibility ───────────────────────────────────────────────
        if self.cfg.save_random_seeds:
            safe_write_json(
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
            save_git_commit(self.paths["git_commit_json"])

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
        manifest = build_manifest(
            paths=self.paths,
            framework="pytorch",
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

    def _infer_num_classes(self, model: nn.Module) -> int:
        """Walk modules in reverse to find the last Linear / Conv output size."""
        for layer in reversed(list(model.modules())):
            if isinstance(layer, nn.Linear):
                out = layer.out_features
                return 2 if out == 1 else int(out)
            if isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                out = layer.out_channels
                return 2 if out == 1 else int(out)
        raise ValueError(
            "Could not infer num_classes from the model's layers. "
            "Pass class_names= explicitly to ExperimentSaver()."
        )

    @torch.no_grad()
    def _collect_predictions(
        self,
        model: nn.Module,
        loader: DataLoader,
        num_classes: int,
        device: torch.device,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Iterate *loader* and return (y_true, y_score, y_pred).

        y_true : (N,) int32
        y_score: (N,) or (N, C) float32 probabilities
        y_pred : (N,) int32
        """
        model.eval()
        y_true_all: List[np.ndarray] = []
        y_score_all: List[np.ndarray] = []

        for batch in loader:
            if not isinstance(batch, (tuple, list)) or len(batch) < 2:
                raise ValueError(
                    "val_loader must yield (x, y) batches. "
                    "Ensure your DataLoader is structured as (features, labels)."
                )
            x_batch, y_batch = batch[0], batch[1]
            if isinstance(x_batch, torch.Tensor):
                x_batch = x_batch.to(device)

            out = model(x_batch)
            pred_np = out.cpu().numpy() if isinstance(out, torch.Tensor) else np.asarray(out)

            y_true_all.append(self._labels_to_int(y_batch, num_classes))
            y_score_all.append(self._pred_to_probs(pred_np, num_classes))

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
        y_np = y.cpu().numpy() if isinstance(y, torch.Tensor) else np.asarray(y)
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

    def _pred_to_probs(self, pred_np: np.ndarray, num_classes: int) -> np.ndarray:
        """
        Convert model output to probability scores.

        If ``cfg.model_outputs_logits=True``, applies sigmoid (binary) or softmax
        (multiclass) before returning.  Otherwise, treats the output as already
        containing probabilities.
        """
        if self.cfg.model_outputs_logits:
            t = torch.from_numpy(pred_np).float()
            if num_classes <= 2 and pred_np.shape[-1] == 1:
                probs = torch.sigmoid(t).squeeze(-1).numpy()      # (N,)
            else:
                probs = torch.softmax(t, dim=-1).numpy()           # (N, C)
            return probs.astype(np.float32)

        # Already probabilities — validate shape and return
        if num_classes <= 2:
            if pred_np.ndim == 2 and pred_np.shape[1] == 1:
                return pred_np[:, 0].astype(np.float32)
            if pred_np.ndim == 2 and pred_np.shape[1] == 2:
                return pred_np.astype(np.float32)
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
            f"Expected (batch, {num_classes})."
        )

    def _compute_grad_norm(self, model: nn.Module) -> float:
        total = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total += p.grad.detach().norm(2).item() ** 2
        return total ** 0.5

    def _save_model_summary(self, model: nn.Module) -> None:
        if _TORCHINFO_AVAILABLE and self.cfg.model_input_size is not None:
            try:
                summary_str = str(_torchinfo_summary(model, input_size=self.cfg.model_input_size, verbose=0))
            except Exception:
                summary_str = str(model)
        else:
            summary_str = str(model)
            if _TORCHINFO_AVAILABLE and self.cfg.model_input_size is None:
                summary_str = (
                    "# torchinfo is available but config.model_input_size is not set.\n"
                    "# Set model_input_size=(1, C, H, W) for a detailed summary.\n\n"
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
        safe_write_json(self.paths["environment_json"], env)
