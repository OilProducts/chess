from pathlib import Path

import pytest

try:
    import mlflow  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    mlflow = None  # type: ignore

from chessref.utils.logging import LoggingConfig, TrainLogger


@pytest.mark.skipif(mlflow is None, reason="mlflow not available")
def test_train_logger_creates_mlflow_run(tmp_path: Path) -> None:
    tracking_dir = tmp_path / "mlruns"
    cfg = LoggingConfig(
        enabled=True,
        tracking_uri=tracking_dir.as_uri(),
        experiment="test-exp",
        run_name="test-run",
    )
    logger = TrainLogger(cfg)
    assert logger.enabled
    logger.log_scalar("loss", 0.5, step=1)
    logger.flush()
    logger.close()
    assert tracking_dir.exists()
    # Expect experiment directory to be created (ID starts at 0)
    assert any(tracking_dir.iterdir())
