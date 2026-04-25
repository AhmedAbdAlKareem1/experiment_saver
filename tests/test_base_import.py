import unittest
import os
import sys

# Ensure we import the local checkout, not an installed package.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class TestBaseImport(unittest.TestCase):
    def test_import_experiment_saver_base(self):
        import experiment_saver  # noqa: F401

    def test_missing_backends_raise_importerror(self):
        import experiment_saver

        # In a base install (no extras), these should raise ImportError when instantiated.
        # We can't guarantee torch/tf are absent on every machine, so we only assert
        # ImportError if the backend is actually missing.
        try:
            import tensorflow  # noqa: F401
            tf_available = True
        except Exception:
            tf_available = False

        if not tf_available:
            with self.assertRaises(ImportError):
                experiment_saver.TFExperimentSaver(config=None)  # type: ignore[arg-type]

        try:
            import torch  # noqa: F401
            torch_available = True
        except Exception:
            torch_available = False

        if not torch_available:
            with self.assertRaises(ImportError):
                experiment_saver.TorchExperimentSaver(config=None)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()

