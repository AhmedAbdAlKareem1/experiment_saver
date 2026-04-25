"""
Backward-compatibility shim — do not import this directly in new code.

Use instead::

    from experiment_saver import TorchExperimentSaver, TorchExperimentConfig
"""
from __future__ import annotations

from experiment_saver._torch_impl import (
    EarlyStopping,
    ExperimentConfig,
    ExperimentSaver,
)

__all__ = ["ExperimentConfig", "ExperimentSaver", "EarlyStopping"]
