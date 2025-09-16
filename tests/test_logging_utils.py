from pathlib import Path

from chessref.utils.logging import LoggingConfig, TrainLogger


def test_train_logger_creates_event(tmp_path: Path) -> None:
    log_dir = tmp_path / "runs"
    logger = TrainLogger(LoggingConfig(enabled=True, log_dir=str(log_dir)))
    assert log_dir.exists()
    logger.log_scalar("loss", 0.5, step=1)
    logger.flush()
    logger.close()
    # Writer may create subdirectories; ensure something was written.
    assert any(log_dir.iterdir())
