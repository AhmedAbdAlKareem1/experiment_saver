"""
experiment_saver

Unified experiment saving utilities for:
- TensorFlow / Keras
- PyTorch
"""

from __future__ import annotations

from typing import Any


def _missing_backend(backend: str, extra: str, err: Exception) -> ImportError:
    return ImportError(
        f"{backend} backend is not available. Install with `pip install \"experiment-saver[{extra}]\"`."
    ).with_traceback(err.__traceback__)


class TFExperimentSaver:  # pragma: no cover
    def __new__(cls, *args: Any, **kwargs: Any):
        try:
            from .saver_tensorflow import ExperimentSaver as _Saver
        except Exception as e:
            raise ImportError(
                "TensorFlow backend is not available. Install with `pip install \"experiment-saver[tf]\"`."
            ) from e
        return _Saver(*args, **kwargs)


class TFExperimentConfig:  # pragma: no cover
    def __new__(cls, *args: Any, **kwargs: Any):
        try:
            from .saver_tensorflow import ExperimentConfig as _Cfg
        except Exception as e:
            raise ImportError(
                "TensorFlow backend is not available. Install with `pip install \"experiment-saver[tf]\"`."
            ) from e
        return _Cfg(*args, **kwargs)


class TorchExperimentSaver:  # pragma: no cover
    def __new__(cls, *args: Any, **kwargs: Any):
        try:
            from .experiment_saver_torch import ExperimentSaver as _Saver
        except Exception as e:
            raise ImportError(
                "PyTorch backend is not available. Install with `pip install \"experiment-saver[torch]\"`."
            ) from e
        return _Saver(*args, **kwargs)


class TorchExperimentConfig:  # pragma: no cover
    def __new__(cls, *args: Any, **kwargs: Any):
        try:
            from .experiment_saver_torch import ExperimentConfig as _Cfg
        except Exception as e:
            raise ImportError(
                "PyTorch backend is not available. Install with `pip install \"experiment-saver[torch]\"`."
            ) from e
        return _Cfg(*args, **kwargs)

__all__ = [
    "TFExperimentSaver",
    "TFExperimentConfig",
    "TorchExperimentSaver",
    "TorchExperimentConfig",
]
