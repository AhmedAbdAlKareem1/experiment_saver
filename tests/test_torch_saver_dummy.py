"""
PyTorch ExperimentSaver tests with small dummy models and synthetic data.

All tests run offline — no pretrained weights are downloaded.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _binary_model():
    torch = pytest.importorskip("torch")
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(4, 8), nn.ReLU(),
        nn.Linear(8, 1), nn.Sigmoid(),
    )


def _multiclass_model(num_classes: int = 3):
    torch = pytest.importorskip("torch")
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(4, 8), nn.ReLU(),
        nn.Linear(8, num_classes), nn.Softmax(dim=1),
    )


def _binary_loader(n: int = 40, seed: int = 0):
    torch = pytest.importorskip("torch")
    from torch.utils.data import DataLoader, TensorDataset
    torch.manual_seed(seed)
    X = torch.randn(n, 4)
    y = torch.randint(0, 2, (n,))
    return DataLoader(TensorDataset(X, y), batch_size=10)


def _multiclass_loader(num_classes: int = 3, n: int = 60, seed: int = 0):
    torch = pytest.importorskip("torch")
    from torch.utils.data import DataLoader, TensorDataset
    torch.manual_seed(seed)
    X = torch.randn(n, 4)
    y = torch.randint(0, num_classes, (n,))
    return DataLoader(TensorDataset(X, y), batch_size=10)


# ---------------------------------------------------------------------------
# Tests — binary classification
# ---------------------------------------------------------------------------

class TestTorchSaverBinary:
    def test_core_artifacts_exist(self, tmp_path):
        pytest.importorskip("torch")
        from experiment_saver import TorchExperimentConfig, TorchExperimentSaver

        cfg = TorchExperimentConfig(
            run_dir=str(tmp_path / "binary"),
            verbose=0, save_environment=False, save_git_commit=False,
        )
        saver = TorchExperimentSaver(cfg, class_names=["neg", "pos"])

        for epoch in range(3):
            saver.log_epoch(epoch, {"train_loss": 0.6 - epoch * 0.1, "val_loss": 0.7 - epoch * 0.1})

        paths = saver.save_after_fit(_binary_model(), val_loader=_binary_loader())

        for key in ("final_model", "manifest_json", "history_json", "roc_auc_json",
                    "confusion_matrix", "classification_report", "val_labels",
                    "val_scores", "val_predictions", "metrics_csv"):
            assert os.path.isfile(paths[key]), f"Missing: {key} at {paths[key]}"

    def test_roc_is_binary(self, tmp_path):
        pytest.importorskip("torch")
        from experiment_saver import TorchExperimentConfig, TorchExperimentSaver

        cfg = TorchExperimentConfig(
            run_dir=str(tmp_path / "roc_binary"),
            verbose=0, save_environment=False, save_git_commit=False,
        )
        saver = TorchExperimentSaver(cfg, class_names=["neg", "pos"])
        paths = saver.save_after_fit(_binary_model(), val_loader=_binary_loader())

        with open(paths["roc_auc_json"]) as f:
            roc = json.load(f)
        assert roc["task"] == "binary"
        assert "roc_auc" in roc
        assert 0.0 <= roc["roc_auc"] <= 1.0

    def test_manifest_has_required_fields(self, tmp_path):
        pytest.importorskip("torch")
        from experiment_saver import TorchExperimentConfig, TorchExperimentSaver

        cfg = TorchExperimentConfig(
            run_dir=str(tmp_path / "manifest"),
            verbose=0, save_environment=False, save_git_commit=False,
        )
        saver = TorchExperimentSaver(cfg, class_names=["neg", "pos"])
        paths = saver.save_after_fit(_binary_model(), val_loader=_binary_loader())

        with open(paths["manifest_json"]) as f:
            m = json.load(f)

        for field in ("package_version", "framework", "timestamp", "python_version",
                      "monitor", "class_names", "artifacts"):
            assert field in m, f"Missing manifest field: {field}"

        assert m["framework"] == "pytorch"
        assert m["class_names"] == ["neg", "pos"]
        assert isinstance(m["artifacts"], dict)

    def test_best_checkpoint_contains_class_names(self, tmp_path):
        torch = pytest.importorskip("torch")
        from experiment_saver import TorchExperimentConfig, TorchExperimentSaver

        cfg = TorchExperimentConfig(
            run_dir=str(tmp_path / "best_ckpt"),
            verbose=0, save_environment=False, save_git_commit=False,
        )
        saver = TorchExperimentSaver(cfg, class_names=["neg", "pos"])
        model = _binary_model()

        for epoch in range(3):
            val_loss = 0.8 - epoch * 0.1
            saver.log_epoch(epoch, {"val_loss": val_loss})
            if saver.should_save_best(val_loss):
                saver.save_best_checkpoint(model)

        ckpt = torch.load(saver.paths["best_model"], map_location="cpu")
        assert "class_names" in ckpt
        assert ckpt["class_names"] == ["neg", "pos"]

    def test_early_stopping_triggers(self, tmp_path):
        pytest.importorskip("torch")
        from experiment_saver import TorchExperimentConfig, TorchExperimentSaver

        cfg = TorchExperimentConfig(
            run_dir=str(tmp_path / "es"),
            patience=2, verbose=0,
            save_environment=False, save_git_commit=False,
        )
        saver = TorchExperimentSaver(cfg, class_names=["a", "b"])
        early_stop = saver.make_early_stopping()

        val_losses = [0.5, 0.4, 0.45, 0.50, 0.55]
        stopped_at = None
        for epoch, loss in enumerate(val_losses):
            saver.log_epoch(epoch, {"val_loss": loss})
            if early_stop.step(loss):
                stopped_at = epoch
                break

        assert stopped_at is not None, "EarlyStopping did not trigger"
        assert stopped_at < len(val_losses) - 1

    def test_lr_history_rows(self, tmp_path):
        torch = pytest.importorskip("torch")
        from experiment_saver import TorchExperimentConfig, TorchExperimentSaver

        model = torch.nn.Linear(4, 2)
        opt = torch.optim.Adam(model.parameters(), lr=0.001)

        cfg = TorchExperimentConfig(
            run_dir=str(tmp_path / "lr"),
            verbose=0, save_environment=False, save_git_commit=False,
        )
        saver = TorchExperimentSaver(cfg)
        saver.log_lr(opt, epoch=0)
        saver.log_lr(opt, epoch=1)

        assert len(saver._lr_rows) == 3   # header + 2 data rows
        assert saver._lr_rows[0].startswith("epoch,")

    def test_no_val_loader_roc_note(self, tmp_path):
        pytest.importorskip("torch")
        from experiment_saver import TorchExperimentConfig, TorchExperimentSaver

        cfg = TorchExperimentConfig(
            run_dir=str(tmp_path / "no_val"),
            verbose=0, save_environment=False, save_git_commit=False,
        )
        saver = TorchExperimentSaver(cfg, class_names=["a", "b"])
        paths = saver.save_after_fit(_binary_model())   # no val_loader

        with open(paths["roc_auc_json"]) as f:
            roc = json.load(f)
        assert "note" in roc


# ---------------------------------------------------------------------------
# Tests — multiclass classification
# ---------------------------------------------------------------------------

class TestTorchSaverMulticlass:
    def test_roc_is_multiclass(self, tmp_path):
        pytest.importorskip("torch")
        from experiment_saver import TorchExperimentConfig, TorchExperimentSaver

        cfg = TorchExperimentConfig(
            run_dir=str(tmp_path / "mc"),
            verbose=0, save_environment=False, save_git_commit=False,
        )
        saver = TorchExperimentSaver(cfg, class_names=["a", "b", "c"])
        paths = saver.save_after_fit(
            _multiclass_model(3), val_loader=_multiclass_loader(3)
        )

        with open(paths["roc_auc_json"]) as f:
            roc = json.load(f)
        assert roc["task"] == "multiclass"
        assert "per_class_auc" in roc

    def test_confusion_matrix_shape(self, tmp_path):
        pytest.importorskip("torch")
        from experiment_saver import TorchExperimentConfig, TorchExperimentSaver

        cfg = TorchExperimentConfig(
            run_dir=str(tmp_path / "cm"),
            verbose=0, save_environment=False, save_git_commit=False,
        )
        saver = TorchExperimentSaver(cfg, class_names=["a", "b", "c"])
        paths = saver.save_after_fit(
            _multiclass_model(3), val_loader=_multiclass_loader(3)
        )
        cm = np.load(paths["confusion_matrix"])
        assert cm.shape == (3, 3)

    def test_class_names_length_mismatch_raises(self, tmp_path):
        pytest.importorskip("torch")
        from experiment_saver import TorchExperimentConfig, TorchExperimentSaver

        cfg = TorchExperimentConfig(
            run_dir=str(tmp_path / "mismatch"),
            verbose=0, save_environment=False, save_git_commit=False,
        )
        saver = TorchExperimentSaver(cfg, class_names=["a", "b"])  # wrong: model has 3 classes
        with pytest.raises(ValueError, match="class_names"):
            saver.save_after_fit(_multiclass_model(3), val_loader=_multiclass_loader(3))


# ---------------------------------------------------------------------------
# Tests — base import (no framework required)
# ---------------------------------------------------------------------------

class TestBaseImport:
    def test_import_works(self):
        import experiment_saver  # noqa: F401

    def test_version_exported(self):
        import experiment_saver
        assert hasattr(experiment_saver, "__version__")
        assert experiment_saver.__version__

    def test_missing_torch_raises_importerror(self):
        """When torch is not installed, instantiation must raise ImportError."""
        import experiment_saver
        try:
            import torch  # noqa: F401
        except ImportError:
            with pytest.raises(ImportError, match="PyTorch"):
                experiment_saver.TorchExperimentSaver(config=None)
