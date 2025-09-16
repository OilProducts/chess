"""Minimal distributed training helpers."""

from __future__ import annotations

import torch


def is_distributed_available() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def get_world_size(default: int = 1) -> int:
    if is_distributed_available():
        return torch.distributed.get_world_size()
    return default


def get_rank(default: int = 0) -> int:
    if is_distributed_available():
        return torch.distributed.get_rank()
    return default


def barrier() -> None:
    if is_distributed_available():  # pragma: no cover - depends on dist backend
        torch.distributed.barrier()


def setup_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover
        torch.cuda.manual_seed_all(seed)


__all__ = ["is_distributed_available", "get_world_size", "get_rank", "barrier", "setup_seed"]
