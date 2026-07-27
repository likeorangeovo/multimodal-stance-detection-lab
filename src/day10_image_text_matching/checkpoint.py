"""Checkpoint save/load helpers for Day10.

TODO:
- save_checkpoint()
- load_checkpoint()
"""
import os

import torch


def save_checkpoint(path, model, optimizer=None, epoch=None, metric=None):
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "metric": metric,
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    torch.save(checkpoint, path)


def load_checkpoint(path, model, optimizer=None, device="cpu"):
    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint.get("epoch")
    metric = checkpoint.get("metric")

    return epoch, metric
