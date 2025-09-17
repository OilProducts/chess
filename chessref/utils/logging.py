"""Lightweight logging utilities backed by MLflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

try:  # pragma: no cover - import side effect tested elsewhere
    import mlflow
except ImportError:  # pragma: no cover - optional dependency
    mlflow = None  # type: ignore


@dataclass
class LoggingConfig:
    enabled: bool = False
    tracking_uri: Optional[str] = None
    experiment: Optional[str] = "Default"
    run_name: Optional[str] = None


class TrainLogger:
    """Minimal MLflow-backed metric logger.

    Logging is optional—if MLflow is unavailable or logging is disabled, all
    methods degrade to no-ops so training loops remain uncluttered.
    """

    def __init__(self, config: LoggingConfig) -> None:
        self._enabled = config.enabled and mlflow is not None
        self._active_run = False
        self._config = config
        if self._enabled:
            self._start_run()

    @property
    def enabled(self) -> bool:
        return self._enabled and self._active_run

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        if self.enabled:
            mlflow.log_metric(tag, float(value), step=step)

    def log_scalars(self, metrics: Dict[str, float], step: int) -> None:
        if self.enabled:
            mlflow.log_metrics({key: float(val) for key, val in metrics.items()}, step=step)

    def flush(self) -> None:
        # MLflow writes metrics immediately; nothing to flush.
        return None

    def close(self) -> None:
        if self.enabled:
            mlflow.end_run()
            self._active_run = False
            self._enabled = False

    def _start_run(self) -> None:
        assert mlflow is not None  # for type-checkers
        if self._config.tracking_uri:
            mlflow.set_tracking_uri(self._config.tracking_uri)
        if self._config.experiment:
            mlflow.set_experiment(self._config.experiment)
        mlflow.start_run(run_name=self._config.run_name)
        self._active_run = True


__all__ = ["LoggingConfig", "TrainLogger"]
