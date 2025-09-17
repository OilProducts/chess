from pathlib import Path

import torch

from chessref.moves.encoding import NUM_MOVES
import chessref.train.train_supervised as train_module
from chessref.train.train_supervised import (
    CheckpointConfig,
    DataConfig,
    ModelConfig,
    OptimConfig,
    TrainConfig,
    TrainSummary,
    TrainingConfig,
    train,
    _save_iteration_checkpoint,
    _resume_from_latest_iteration,
)
from chessref.model.refiner import IterativeRefiner

PGN_SAMPLE = """
[Event "Test"]
[White "WhitePlayer"]
[Black "BlackPlayer"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 O-O 8. c3 d5 9. exd5 Nxd5 10. Nxe5 Nxe5 11. Rxe5 c6 12. d4 Bd6 13. Re1 Qh4 14. g3 Qh3 15. Re4 g5 16. Bxg5 Qf5 17. Bc2 Qxg5 18. Rh4 Bf5 19. Rh5 Qxh5 20. Qxh5 Bxc2 21. Nd2 Rae8 22. Nf3 Re6 23. Ne5 Bxe5 24. dxe5 Rfe8 25. f4 f5 26. Kf2 R8e7 27. Rg1 Rg6 28. Qh4 Rd7 29. h3 Be4 30. g4 Nxf4 31. gxf5 Rxg1 32. Kxg1 Rd1+ 33. Kf2 Nd3+ 34. Ke3 Re1+ 35. Kd2 Rh1 36. Qxe4 Rh2+ 37. Ke3 Rxh3+ 38. Kd4 c5+ 39. Kd5 Nf2 40. Qg2+ Kf7 41. Qxf2 Rh6 42. e6+ Ke8 43. Kd6 Rf6 44. Qxc5 Kf8 45. Ke5+ 1-0
"""


def _write_sample_pgn(tmp_path: Path) -> Path:
    path = tmp_path / "sample.pgn"
    path.write_text(PGN_SAMPLE, encoding="utf-8")
    return path


def test_train_loop_smoke(tmp_path: Path) -> None:
    pgn_path = _write_sample_pgn(tmp_path)

    model_cfg = ModelConfig(num_moves=NUM_MOVES, d_model=64, nhead=4, depth=1, dim_feedforward=128, dropout=0.0, use_act=False)
    data_cfg = DataConfig(
        pgn_paths=[str(pgn_path)],
        max_games=1,
        transforms=["identity"],
        target={"type": "selfplay"},
        shuffle=False,
    )
    optim_cfg = OptimConfig(lr=1e-3, weight_decay=0.0, betas=(0.9, 0.999), grad_accum_steps=1, use_amp=False)
    training_cfg = TrainingConfig(
        epochs=1,
        batch_size=4,
        k_train=2,
        detach_prev_policy=True,
        act_weight=0.0,
        value_loss="mse",
        device="cpu",
        log_interval=0,
    )
    checkpoint_cfg = CheckpointConfig(directory=str(tmp_path / "ckpt"), save_interval=0)

    cfg = TrainConfig(
        model=model_cfg,
        data=data_cfg,
        optim=optim_cfg,
        training=training_cfg,
        checkpoint=checkpoint_cfg,
    )

    summary = train(cfg)
    assert isinstance(summary, TrainSummary)
    assert summary.epochs_completed == 1
    assert summary.steps > 0
    assert summary.last_loss == summary.last_loss  # not NaN

    iteration_ckpts = list((tmp_path / "ckpt").glob("iteration_*.pt"))
    assert len(iteration_ckpts) == 1


def test_iteration_checkpoint_rotation(tmp_path: Path) -> None:
    directory = tmp_path / "ckpt"
    model = IterativeRefiner(num_moves=8, d_model=16, nhead=2, depth=1, dim_feedforward=32, dropout=0.0, use_act=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    for step in range(12):
        _save_iteration_checkpoint(directory, model, optimizer, step, k_train=1, keep=10)

    files = sorted(directory.glob("iteration_*.pt"))
    assert len(files) == 10
    indices = sorted(int(path.stem.split("_")[1]) for path in files)
    assert indices == list(range(2, 12))


def test_resume_skips_incompatible_checkpoint(tmp_path: Path) -> None:
    directory = tmp_path / "ckpt"
    directory.mkdir(parents=True, exist_ok=True)

    model = IterativeRefiner(num_moves=8, d_model=16, nhead=2, depth=1, dim_feedforward=32, dropout=0.0, use_act=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    bad_path = directory / "iteration_00000.pt"
    torch.save({"model": {}}, bad_path)

    step = _resume_from_latest_iteration(directory, model, optimizer, torch.device("cpu"), k_train=1)
    assert step == 0
    assert bad_path.exists()


def test_train_resumes_from_latest_iteration(tmp_path: Path, monkeypatch) -> None:
    pgn_path = _write_sample_pgn(tmp_path)
    checkpoint_dir = tmp_path / "ckpt"

    model_cfg = ModelConfig(num_moves=NUM_MOVES, d_model=64, nhead=4, depth=1, dim_feedforward=128, dropout=0.0, use_act=False)
    data_cfg = DataConfig(
        pgn_paths=[str(pgn_path)],
        max_games=1,
        transforms=["identity"],
        target={"type": "selfplay"},
        shuffle=False,
    )
    optim_cfg = OptimConfig(lr=1e-3, weight_decay=0.0, betas=(0.9, 0.999), grad_accum_steps=1, use_amp=False)
    training_cfg = TrainingConfig(
        epochs=1,
        batch_size=4,
        k_train=1,
        detach_prev_policy=True,
        act_weight=0.0,
        value_loss="mse",
        device="cpu",
        log_interval=0,
    )
    checkpoint_cfg = CheckpointConfig(directory=str(checkpoint_dir), save_interval=0)

    cfg = TrainConfig(
        model=model_cfg,
        data=data_cfg,
        optim=optim_cfg,
        training=training_cfg,
        checkpoint=checkpoint_cfg,
    )

    summary1 = train(cfg)
    assert summary1.steps > 0
    first_ckpt = list(checkpoint_dir.glob("iteration_*.pt"))
    assert len(first_ckpt) == 1

    calls = {}
    original_load = train_module.load_checkpoint

    def wrapped_load(*args, **kwargs):
        calls["path"] = Path(args[0])
        return original_load(*args, **kwargs)

    monkeypatch.setattr(train_module, "load_checkpoint", wrapped_load)

    summary2 = train(cfg)
    assert summary2.steps >= summary1.steps
    assert calls["path"].name.startswith("iteration_00000_")
    ckpts = sorted(checkpoint_dir.glob("iteration_*.pt"))
    assert len(ckpts) == 2
