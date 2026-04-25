"""
Example: PyTorch custom training loop — multiclass classification (4 classes).

Run:
    pip install "experiment-saver[torch]"
    python examples/torch_training_loop_multiclass.py
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from experiment_saver import TorchExperimentConfig, TorchExperimentSaver


# ---------------------------------------------------------------------------
# Tiny demo dataset (replace with real data)
# ---------------------------------------------------------------------------

NUM_CLASSES = 4


def make_loaders(n: int = 400, val_split: float = 0.2, seed: int = 42):
    torch.manual_seed(seed)
    X = torch.randn(n, 32)
    y = torch.randint(0, NUM_CLASSES, (n,))
    n_val = int(n * val_split)
    train_loader = DataLoader(TensorDataset(X[n_val:], y[n_val:]), batch_size=32, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X[:n_val], y[:n_val]),  batch_size=32)
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(input_dim: int = 32, num_classes: int = NUM_CLASSES) -> nn.Module:
    return nn.Sequential(
        nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(128, 64), nn.ReLU(),
        nn.Linear(64, num_classes),
        # No Softmax here — CrossEntropyLoss expects raw logits
    )


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(X)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct = 0.0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        out = model(X)
        total_loss += criterion(out, y).item() * len(X)
        correct += (out.argmax(dim=1) == y).sum().item()
    n = len(loader.dataset)
    return total_loss / n, correct / n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = make_loaders()

    model = build_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    class_names = ["alpha", "beta", "gamma", "delta"]

    cfg = TorchExperimentConfig(
        run_dir="runs/torch_multiclass_001",
        monitor="val_loss",
        patience=5,
        verbose=1,
        model_input_size=(1, 32),           # for torchinfo summary (optional)
        model_outputs_logits=True,          # model returns raw logits, not softmax probs
        bundle_artifacts=False,
    )

    saver = TorchExperimentSaver(cfg, class_names=class_names)
    early_stop = saver.make_early_stopping()

    for epoch in range(50):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        saver.log_lr(optimizer, epoch=epoch)
        scheduler.step()

        metrics = {
            "train_loss": train_loss,
            "val_loss":   val_loss,
            "val_acc":    val_acc,
        }
        saver.log_epoch(epoch, metrics, model=model, epoch_time=time.time() - t0)

        if saver.should_save_best(val_loss):
            saver.save_best_checkpoint(model)

        if early_stop.step(val_loss):
            break

    saved_paths = saver.save_after_fit(
        model=model,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        last_epoch=epoch,
        device=device,
        extra_config={
            "learning_rate": 1e-3,
            "architecture": "dense_3layer",
            "num_classes": NUM_CLASSES,
        },
    )

    print("\nSaved artifacts:")
    for k, v in saved_paths.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
