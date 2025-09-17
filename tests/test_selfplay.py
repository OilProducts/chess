from pathlib import Path
from typing import List

import chess
import chess.engine
import torch

from chessref.train.selfplay import SelfPlayConfig, generate_selfplay_games
from chessref.train.train_supervised import (
    ModelConfig,
    TrainConfig,
    ModelConfig as TrainModelConfig,
    DataConfig,
    OptimConfig,
    TrainingConfig,
    CheckpointConfig,
    LoggingConfig,
    TrainSummary,
)
from chessref.moves.encoding import NUM_MOVES


def test_generate_selfplay_games(tmp_path: Path) -> None:
    output = tmp_path / "out.pgn"
    dataset_path = tmp_path / "data.pt"
    cfg = SelfPlayConfig(
        model=ModelConfig(num_moves=4672, d_model=32, nhead=4, depth=1, dim_feedforward=64, dropout=0.0, use_act=False),
        checkpoint=None,
        num_games=1,
        max_moves=10,
        output_pgn=str(output),
        output_dataset=str(dataset_path),
        device="cpu",
        inference_max_loops=1,
        temperature=1.0,
        mcts_num_simulations=8,
        train_after=False,
        max_datasets=None,
        train_every_batches=1,
        run_forever=False,
        max_batches=None,
    )

    path = generate_selfplay_games(cfg)
    assert path.exists()
    content = path.read_text().strip()
    assert "Result" in content
    assert dataset_path.exists()
    samples = torch.load(dataset_path, weights_only=False)
    assert samples


def test_generate_selfplay_games_triggers_training(monkeypatch, tmp_path: Path) -> None:
    train_called = {}

    def fake_train(cfg: TrainConfig) -> TrainSummary:
        train_called["cfg"] = cfg
        return TrainSummary(epochs_completed=0, steps=0, last_loss=0.0)

    def fake_load_config(path: Path) -> TrainConfig:
        data_cfg = DataConfig(
            pgn_paths=[],
            max_games=1,
            transforms=["identity"],
            target={"type": "selfplay"},
            selfplay_datasets=[],
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )
        model_cfg = TrainModelConfig(num_moves=NUM_MOVES, d_model=32, nhead=4, depth=1, dim_feedforward=64, dropout=0.0, use_act=False)
        return TrainConfig(
            model=model_cfg,
            data=data_cfg,
            optim=OptimConfig(),
            training=TrainingConfig(epochs=0, batch_size=1, k_train=1, detach_prev_policy=True, act_weight=0.0, value_loss="mse", device="cpu", log_interval=0),
            checkpoint=CheckpointConfig(directory=str(tmp_path / "ckpt"), save_interval=0),
            logging=LoggingConfig(enabled=False),
        )

    monkeypatch.setattr("chessref.train.selfplay.train", fake_train)
    monkeypatch.setattr("chessref.train.selfplay.load_train_config", fake_load_config)

    output = tmp_path / "out.pgn"
    dataset_path = tmp_path / "data.pt"
    cfg = SelfPlayConfig(
        model=ModelConfig(num_moves=4672, d_model=32, nhead=4, depth=1, dim_feedforward=64, dropout=0.0, use_act=False),
        checkpoint=None,
        num_games=1,
        max_moves=5,
        output_pgn=str(output),
        output_dataset=str(dataset_path),
        device="cpu",
        inference_max_loops=1,
        temperature=1.0,
        mcts_num_simulations=4,
        train_after=True,
        train_config=str(tmp_path / "train.yaml"),
        max_datasets=None,
        train_every_batches=1,
        run_forever=False,
        max_batches=None,
    )

    generate_selfplay_games(cfg)
    assert "cfg" in train_called
    assert dataset_path.exists()
    assert dataset_path.as_posix() in train_called["cfg"].data.selfplay_datasets


def test_generate_selfplay_games_stockfish_eval(monkeypatch, tmp_path: Path) -> None:
    calls: List[int] = []

    class FakeEngine:
        def __init__(self) -> None:
            self.closed = False

        def analyse(self, board: chess.Board, limit, multipv: int = 1):
            calls.append(1)
            moves = list(board.legal_moves)
            move1 = moves[0]
            score1 = chess.engine.PovScore(chess.engine.Cp(100), chess.WHITE if board.turn else chess.BLACK)
            entries = [{"pv": [move1], "score": score1}]
            if multipv > 1 and len(moves) > 1:
                move2 = moves[1]
                score2 = chess.engine.PovScore(chess.engine.Cp(50), chess.WHITE if board.turn else chess.BLACK)
                entries.append({"pv": [move2], "score": score2})
            return entries

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr("chess.engine.SimpleEngine.popen_uci", lambda path: FakeEngine())

    output = tmp_path / "out.pgn"
    dataset_path = tmp_path / "data.pt"
    cfg = SelfPlayConfig(
        model=ModelConfig(num_moves=4672, d_model=32, nhead=4, depth=1, dim_feedforward=64, dropout=0.0, use_act=False),
        checkpoint=None,
        num_games=1,
        max_moves=3,
        output_pgn=str(output),
        output_dataset=str(dataset_path),
        device="cpu",
        inference_max_loops=1,
        temperature=1.0,
        mcts_num_simulations=4,
        train_after=False,
        eval_engine_path="fake",
        eval_depth=1,
        eval_multipv=2,
        max_datasets=None,
        train_every_batches=1,
        run_forever=False,
        max_batches=None,
    )

    generate_selfplay_games(cfg)
    assert calls  # engine was queried


def test_selfplay_respects_dataset_limit(monkeypatch, tmp_path: Path) -> None:
    train_calls: List[List[str]] = []

    def fake_train(cfg: TrainConfig) -> TrainSummary:
        train_calls.append(list(cfg.data.selfplay_datasets))
        return TrainSummary(epochs_completed=0, steps=0, last_loss=0.0)

    def fake_load_config(path: Path) -> TrainConfig:
        data_cfg = DataConfig(
            pgn_paths=[],
            max_games=1,
            transforms=["identity"],
            target={"type": "selfplay"},
            selfplay_datasets=[],
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )
        model_cfg = TrainModelConfig(num_moves=NUM_MOVES, d_model=32, nhead=4, depth=1, dim_feedforward=64, dropout=0.0, use_act=False)
        return TrainConfig(
            model=model_cfg,
            data=data_cfg,
            optim=OptimConfig(),
            training=TrainingConfig(epochs=0, batch_size=1, k_train=1, detach_prev_policy=True, act_weight=0.0, value_loss="mse", device="cpu", log_interval=0),
            checkpoint=CheckpointConfig(directory=str(tmp_path / "ckpt"), save_interval=0),
            logging=LoggingConfig(enabled=False),
        )

    monkeypatch.setattr("chessref.train.selfplay.train", fake_train)
    monkeypatch.setattr("chessref.train.selfplay.load_train_config", fake_load_config)

    output = tmp_path / "out.pgn"
    dataset = tmp_path / "data.pt"
    cfg = SelfPlayConfig(
        model=ModelConfig(num_moves=4672, d_model=32, nhead=4, depth=1, dim_feedforward=64, dropout=0.0, use_act=False),
        checkpoint=None,
        num_games=1,
        max_moves=3,
        output_pgn=str(output),
        output_dataset=str(dataset),
        device="cpu",
        inference_max_loops=1,
        temperature=1.0,
        mcts_num_simulations=4,
        train_after=True,
        train_config=str(tmp_path / "train.yaml"),
        max_datasets=1,
        train_every_batches=1,
        run_forever=True,
        max_batches=2,
    )

    generate_selfplay_games(cfg)

    assert len(train_calls) == 2
    assert all(len(call) == 1 for call in train_calls)
    old_dataset = dataset.with_name(f"{dataset.stem}_b0{dataset.suffix}")
    new_dataset = dataset.with_name(f"{dataset.stem}_b1{dataset.suffix}")
    assert not old_dataset.exists()
    assert new_dataset.exists()


def test_selfplay_prunes_old_pgns(tmp_path: Path) -> None:
    output = tmp_path / "selfplay.pgn"
    dataset_path = tmp_path / "data.pt"
    cfg = SelfPlayConfig(
        model=ModelConfig(num_moves=4672, d_model=32, nhead=4, depth=1, dim_feedforward=64, dropout=0.0, use_act=False),
        checkpoint=None,
        num_games=1,
        max_moves=5,
        output_pgn=str(output),
        output_dataset=str(dataset_path),
        device="cpu",
        inference_max_loops=1,
        temperature=1.0,
        mcts_num_simulations=4,
        train_after=False,
        max_datasets=None,
        train_every_batches=1,
        run_forever=True,
        max_batches=2,
        max_pgn_games=1,
    )

    generate_selfplay_games(cfg)

    pgn_files = sorted(tmp_path.glob("selfplay_*.pgn"))
    assert len(pgn_files) == 1
    assert pgn_files[0].name.endswith("_b1.pgn")
