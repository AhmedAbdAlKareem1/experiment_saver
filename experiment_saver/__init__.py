"""
experiment_saver
================
Lightweight local experiment saver for TensorFlow/Keras and PyTorch
classification runs.

Importing this package does **not** require TensorFlow or PyTorch to be
installed — the framework-specific code is loaded only when you instantiate
one of the classes below.

Install framework support::

    pip install "experiment-saver[tf]"      # TensorFlow / Keras
    pip install "experiment-saver[torch]"   # PyTorch

Quick start::

    # TensorFlow
    from experiment_saver import TFExperimentSaver, TFExperimentConfig

    # PyTorch
    from experiment_saver import TorchExperimentSaver, TorchExperimentConfig
"""
from __future__ import annotations

from typing import Any

__version__ = "0.2.0"


class TFExperimentSaver:  # pragma: no cover
    """
    Lazy proxy for the TensorFlow/Keras :class:`~experiment_saver._tf_impl.ExperimentSaver`.

    Install with ``pip install "experiment-saver[tf]"``.
    """

    def __new__(cls, *args: Any, **kwargs: Any):
        try:
            from .tf import TFExperimentSaver as _Saver
        except Exception as e:
            raise ImportError(
                "TensorFlow backend is not available. "
                'Install it with:  pip install "experiment-saver[tf]"'
            ) from e
        return _Saver(*args, **kwargs)


class TFExperimentConfig:  # pragma: no cover
    """
    Lazy proxy for the TensorFlow/Keras :class:`~experiment_saver._tf_impl.ExperimentConfig`.

    Install with ``pip install "experiment-saver[tf]"``.
    """

    def __new__(cls, *args: Any, **kwargs: Any):
        try:
            from .tf import TFExperimentConfig as _Cfg
        except Exception as e:
            raise ImportError(
                "TensorFlow backend is not available. "
                'Install it with:  pip install "experiment-saver[tf]"'
            ) from e
        return _Cfg(*args, **kwargs)


class TorchExperimentSaver:  # pragma: no cover
    """
    Lazy proxy for the PyTorch :class:`~experiment_saver._torch_impl.ExperimentSaver`.

    Install with ``pip install "experiment-saver[torch]"``.
    """

    def __new__(cls, *args: Any, **kwargs: Any):
        try:
            from .torch import TorchExperimentSaver as _Saver
        except Exception as e:
            raise ImportError(
                "PyTorch backend is not available. "
                'Install it with:  pip install "experiment-saver[torch]"'
            ) from e
        return _Saver(*args, **kwargs)


class TorchExperimentConfig:  # pragma: no cover
    """
    Lazy proxy for the PyTorch :class:`~experiment_saver._torch_impl.ExperimentConfig`.

    Install with ``pip install "experiment-saver[torch]"``.
    """

    def __new__(cls, *args: Any, **kwargs: Any):
        try:
            from .torch import TorchExperimentConfig as _Cfg
        except Exception as e:
            raise ImportError(
                "PyTorch backend is not available. "
                'Install it with:  pip install "experiment-saver[torch]"'
            ) from e
        return _Cfg(*args, **kwargs)


__all__ = [
    "TFExperimentSaver",
    "TFExperimentConfig",
    "TorchExperimentSaver",
    "TorchExperimentConfig",
    "__version__",
]
