"""Lightweight logging utilities (TensorBoard friendly)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover - optional dependency
    SummaryWriter = None  # type: ignore


@dataclass
class LoggingConfig:
    enabled: bool = False
    log_dir: str = "runs"


class TrainLogger:
    """Tiny wrapper around TensorBoard's ``SummaryWriter``.

    The logger is optional—if TensorBoard is unavailable or logging is disabled,
    all methods become no-ops. This keeps training loops simple without littering
    them with guard clauses.
    """

    def __init__(self, config: LoggingConfig) -> None:
        self._enabled = config.enabled and SummaryWriter is not None
        self._writer: Optional[SummaryWriter] = None
        if self._enabled:
            Path(config.log_dir).mkdir(parents=True, exist_ok=True)
            self._writer = SummaryWriter(log_dir=config.log_dir)

    @property
    def enabled(self) -> bool:
        return self._enabled and self._writer is not None

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        if self.enabled:
            self._writer.add_scalar(tag, value, step)

    def log_scalars(self, metrics: Dict[str, float], step: int) -> None:
        if self.enabled:
            for key, val in metrics.items():
                self._writer.add_scalar(key, val, step)

    def flush(self) -> None:
        if self.enabled:
            self._writer.flush()

    def close(self) -> None:
        if self.enabled:
            self._writer.close()
            self._writer = None
            self._enabled = False


__all__ = ["LoggingConfig", "TrainLogger"]
