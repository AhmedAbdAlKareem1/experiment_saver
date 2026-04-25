"""Test that LR-history CSV header has the right param-group columns."""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class TestTorchLRHeader(unittest.TestCase):
    def test_log_lr_header_has_groups(self):
        try:
            import torch
        except ImportError as e:
            raise unittest.SkipTest("torch not installed") from e

        from experiment_saver import TorchExperimentConfig, TorchExperimentSaver

        model = torch.nn.Linear(4, 2)
        opt = torch.optim.SGD(
            [
                {"params": [model.weight], "lr": 0.1},
                {"params": [model.bias],   "lr": 0.01},
            ]
        )

        saver = TorchExperimentSaver(
            TorchExperimentConfig(
                run_dir="runs/_test_lr_header",
                save_lr_history=True,
                verbose=0,
            )
        )
        saver.log_lr(opt, epoch=0)
        saver.log_lr(opt, epoch=1)

        header = saver._lr_rows[0]
        self.assertEqual(header, "epoch,lr_group0,lr_group1")
        self.assertTrue(saver._lr_rows[1].startswith("0,"))
        self.assertTrue(saver._lr_rows[2].startswith("1,"))


if __name__ == "__main__":
    unittest.main()
