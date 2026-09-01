"""Shared helpers for DL-AG paper-protocol training / eval."""
from __future__ import annotations

import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Best-effort reproducibility within a seed (does not remove cross-seed variance)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass
    # Soft: avoid hard use_deterministic_algorithms (can crash on some CUDA ops)


def make_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def get_device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Shannon entropy over class dimension. logits: [B, C] -> [B]."""
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log(probs.clamp_min(1e-12))
    return -(probs * log_probs).sum(dim=-1)
