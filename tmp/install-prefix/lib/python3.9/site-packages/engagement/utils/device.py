"""Portable torch device selection."""

import os


def select_device(requested: str = "auto"):
    import torch

    requested = os.environ.get("ENGAGEMENT_DEVICE", requested).lower()
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

