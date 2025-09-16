"""Metric helpers for training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch


def topk_accuracy(logits: torch.Tensor, targets: torch.Tensor, k: int = 1) -> torch.Tensor:
    """Compute top-k accuracy for one-hot ``targets``."""

    with torch.no_grad():
        topk = logits.topk(k, dim=-1).indices
        target_indices = targets.argmax(dim=-1, keepdim=True)
        correct = (topk == target_indices).any(dim=-1).float()
        return correct.mean()


@dataclass
class AverageMeter:
    value: float = 0.0
    count: int = 0

    def update(self, val: float, n: int = 1) -> None:
        self.value += val * n
        self.count += n

    @property
    def average(self) -> float:
        if self.count == 0:
            return 0.0
        return self.value / self.count


__all__ = ["topk_accuracy", "AverageMeter"]
