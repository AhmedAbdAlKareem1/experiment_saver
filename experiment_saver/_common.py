"""
Shared utilities used by both TF and PyTorch experiment savers.

This module has **no framework imports** — only the Python standard library,
NumPy, and scikit-learn (imported lazily inside functions that need it).
"""
from __future__ import annotations

import datetime
import json
import os
import platform
import subprocess
import sys
import zipfile
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def make_json_safe(obj: Any) -> Any:
    """Recursively convert *obj* to a JSON-serialisable value."""
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, (list, tuple)):
        return [make_json_safe(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def safe_write_json(path: str, obj: Any) -> None:
    """Write *obj* as pretty-printed JSON to *path*, creating directories as needed."""
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(make_json_safe(obj), f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Manifest builder
# ---------------------------------------------------------------------------

def build_manifest(
    paths: Dict[str, str],
    framework: str,
    class_names: Optional[List[str]],
    monitor: str,
    best_monitor_value: Optional[float],
    package_version: str,
) -> Dict[str, Any]:
    """
    Build a rich manifest dict that captures run metadata.

    Parameters
    ----------
    paths:
        The saver's ``self.paths`` dict (keys → absolute paths).
    framework:
        ``"tensorflow"`` or ``"pytorch"``.
    class_names:
        List of class label strings, or ``None``.
    monitor:
        The metric being monitored (e.g. ``"val_loss"``).
    best_monitor_value:
        Best value seen for *monitor* during training, or ``None``.
    package_version:
        Version string of experiment-saver.
    """
    return {
        "package_version": package_version,
        "framework": framework,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "python_version": sys.version,
        "monitor": monitor,
        "best_monitor_value": best_monitor_value,
        "class_names": class_names,
        "artifacts": {k: os.path.basename(v) for k, v in paths.items()},
    }


# ---------------------------------------------------------------------------
# Label / score helpers
# ---------------------------------------------------------------------------

def one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    """Convert integer label array (N,) to one-hot (N, C)."""
    y = y.astype(np.int32).reshape(-1)
    oh = np.zeros((y.shape[0], num_classes), dtype=np.int32)
    oh[np.arange(y.shape[0]), y] = 1
    return oh


def validate_binary_labels(y_true: np.ndarray) -> None:
    unique = np.unique(y_true)
    if not np.all(np.isin(unique, [0, 1])):
        raise ValueError(
            f"Binary ROC/AUC expects labels in {{0, 1}}. "
            f"Got unique values: {unique.tolist()}. "
            "Check that your dataset is returning integer class indices."
        )


def validate_multiclass_labels(y_true: np.ndarray, num_classes: int) -> None:
    unique = np.unique(y_true)
    if np.any(unique < 0) or np.any(unique >= num_classes):
        raise ValueError(
            f"Multiclass labels must be in [0, {num_classes - 1}]. "
            f"Got unique values: {unique.tolist()}."
        )


# ---------------------------------------------------------------------------
# ROC / AUC
# ---------------------------------------------------------------------------

def save_roc_artifacts(
    y_true: np.ndarray,
    y_score: np.ndarray,
    num_classes: int,
    run_dir: str,
    class_names: Optional[List[str]],
    positive_class_index: int,
    roc_multi_class: str,
    roc_average: str,
) -> Dict[str, Any]:
    """
    Compute and save ROC curve arrays; return a summary dict.

    For binary classification, saves:
        roc_fpr.npy, roc_tpr.npy, roc_thresholds.npy

    For multiclass, saves per-class files:
        roc_fpr_class_{i}.npy, roc_tpr_class_{i}.npy, roc_thresholds_class_{i}.npy

    Parameters
    ----------
    y_score:
        - Binary: 1-D array (N,) of P(positive) **or** 2-D (N, 2)
        - Multiclass: 2-D array (N, C)
    """
    from sklearn.metrics import auc, roc_auc_score, roc_curve

    if num_classes <= 2:
        score_1d = (
            y_score[:, positive_class_index] if y_score.ndim == 2 else y_score
        )
        fpr, tpr, thresholds = roc_curve(y_true, score_1d)
        roc_auc_value = auc(fpr, tpr)
        np.save(os.path.join(run_dir, "roc_fpr.npy"), fpr)
        np.save(os.path.join(run_dir, "roc_tpr.npy"), tpr)
        np.save(os.path.join(run_dir, "roc_thresholds.npy"), thresholds)
        return {
            "task": "binary",
            "roc_auc": float(roc_auc_value),
            "positive_class_index": int(positive_class_index),
        }

    # Multiclass — one-vs-rest per class
    if y_score.ndim != 2 or y_score.shape[1] != num_classes:
        raise ValueError(
            f"Multiclass ROC expects y_score of shape (N, {num_classes}), "
            f"got {y_score.shape}. "
            "Make sure model outputs softmax probabilities for all classes."
        )

    y_true_oh = one_hot(y_true, num_classes)
    per_class_auc: Dict[str, float] = {}

    for i in range(num_classes):
        fpr_i, tpr_i, thr_i = roc_curve(y_true_oh[:, i], y_score[:, i])
        auc_i = auc(fpr_i, tpr_i)
        np.save(os.path.join(run_dir, f"roc_fpr_class_{i}.npy"), fpr_i)
        np.save(os.path.join(run_dir, f"roc_tpr_class_{i}.npy"), tpr_i)
        np.save(os.path.join(run_dir, f"roc_thresholds_class_{i}.npy"), thr_i)
        name = class_names[i] if class_names else f"class_{i}"
        per_class_auc[name] = float(auc_i)

    summary: Dict[str, Any] = {
        "task": "multiclass",
        "multi_class": roc_multi_class,
        "average": roc_average,
        "per_class_auc": per_class_auc,
    }
    for avg in ("macro", "weighted"):
        try:
            summary[f"{avg}_auc"] = float(
                roc_auc_score(y_true, y_score, multi_class=roc_multi_class, average=avg)
            )
        except Exception as exc:
            summary[f"{avg}_auc_error"] = str(exc)
    try:
        summary["micro_auc"] = float(
            roc_auc_score(y_true_oh, y_score, multi_class=roc_multi_class, average="micro")
        )
    except Exception as exc:
        summary["micro_auc_error"] = str(exc)

    return summary


# ---------------------------------------------------------------------------
# Git & environment
# ---------------------------------------------------------------------------

def save_git_commit(path: str) -> None:
    """Write git metadata to *path*; records an error key if not in a git repo."""
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
    except subprocess.CalledProcessError:
        info["error"] = (
            "Not inside a git repository, or git is not installed. "
            "Run from a git repo root to capture commit info."
        )
    except FileNotFoundError:
        info["error"] = "git executable not found on PATH."
    except Exception as exc:
        info["error"] = str(exc)
    safe_write_json(path, info)


# ---------------------------------------------------------------------------
# Artifact bundling
# ---------------------------------------------------------------------------

def bundle_run_dir(
    run_dir: str,
    zip_name: str,
    exclude_exts: Tuple[str, ...],
    delete_originals: bool,
    verbose: int = 1,
) -> str:
    """
    Zip all non-excluded files in *run_dir* into ``{run_dir}/{zip_name}``.

    Returns the full path of the created zip file.
    """
    os.makedirs(run_dir, exist_ok=True)
    zip_path = os.path.join(run_dir, zip_name)
    ex = tuple(e.lower() for e in (exclude_exts or ()))

    def _include(p: str) -> bool:
        if os.path.abspath(p) == os.path.abspath(zip_path):
            return False
        return os.path.splitext(p)[1].lower() not in ex

    files: List[str] = []
    for root, _, filenames in os.walk(run_dir):
        for fn in filenames:
            full = os.path.join(root, fn)
            if _include(full):
                files.append(full)

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for full in files:
            zf.write(full, arcname=os.path.relpath(full, run_dir))

    if verbose:
        print(f"Bundled artifacts → {zip_path}")

    if delete_originals:
        for full in files:
            try:
                os.remove(full)
            except Exception:
                pass

    return zip_path
