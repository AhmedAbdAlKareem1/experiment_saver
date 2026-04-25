"""
Backward-compatibility shim — do not import this directly in new code.

Use instead::

    from experiment_saver import TFExperimentSaver, TFExperimentConfig
"""
from __future__ import annotations

from experiment_saver._tf_impl import ExperimentConfig, ExperimentSaver

__all__ = ["ExperimentConfig", "ExperimentSaver"]
