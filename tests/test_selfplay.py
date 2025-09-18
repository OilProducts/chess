import json
from pathlib import Path
from typing import Dict, List

import chess
import chess.engine
import torch

from chessref.train.selfplay import SelfPlayConfig, StockfishConfig, generate_selfplay_games
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


def _load_manifest(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    assert isinstance(data, list)
    return data


def test_generate_selfplay_games(tmp_path: Path) -> None:
    output = tmp_path / "pgn"
    dataset_dir = tmp_path / "datasets"
    manifest_path = tmp_path / "manifest.json"
    cfg = SelfPlayConfig(
        model=ModelConfig(num_moves=4672, d_model=32, nhead=4, depth=1, dim_feedforward=64, dropout=0.0, use_act=False),
        checkpoint=None,
        num_games=1,
        max_moves=10,
        output_pgn=str(output),
        output_dataset=str(dataset_dir),
        output_manifest=str(manifest_path),
        device="cpu",
        inference_max_loops=1,
        temperature=1.0,
        mcts_num_simulations=8,
        train_after=False,
        train_every_batches=1,
        run_forever=False,
        max_batches=None,
    )

    path = generate_selfplay_games(cfg)
    assert path.exists()
    content = path.read_text().strip()
    assert "Result" in content
    manifest_entries = _load_manifest(manifest_path)
    assert len(manifest_entries) == 1
    dataset_path = Path(manifest_entries[0]["dataset"])
    assert dataset_path.exists()
    samples = torch.load(dataset_path, weights_only=False)
    assert samples
    pgn_entry = Path(manifest_entries[0]["pgn"])
    assert pgn_entry.exists()
    assert manifest_entries[0]["result"] in {"1-0", "0-1", "1/2-1/2"}
    assert "line_hash" in manifest_entries[0]


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

    output = tmp_path / "out"
    dataset_dir = tmp_path / "data"
    manifest_path = tmp_path / "manifest.json"
    cfg = SelfPlayConfig(
        model=ModelConfig(num_moves=4672, d_model=32, nhead=4, depth=1, dim_feedforward=64, dropout=0.0, use_act=False),
        checkpoint=None,
        num_games=1,
        max_moves=1,
        output_pgn=str(output),
        output_dataset=str(dataset_dir),
        output_manifest=str(manifest_path),
        device="cpu",
        inference_max_loops=1,
        temperature=1.0,
        mcts_num_simulations=4,
        train_after=True,
        train_config=str(tmp_path / "train.yaml"),
        train_every_batches=1,
        run_forever=False,
        max_batches=None,
    )

    result = generate_selfplay_games(cfg)
    assert "cfg" in train_called
    manifest_entries = _load_manifest(manifest_path)
    datasets_in_manifest = [entry["dataset"] for entry in manifest_entries]
    assert datasets_in_manifest
    for dataset_path in datasets_in_manifest:
        assert Path(dataset_path).exists()
    # ensure training consumed the most recent dataset(s)
    assert set(datasets_in_manifest).issubset(set(train_called["cfg"].data.selfplay_datasets))
    assert Path(result).exists()


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

    output = tmp_path / "out"
    dataset_dir = tmp_path / "data"
    cfg = SelfPlayConfig(
        model=ModelConfig(num_moves=4672, d_model=32, nhead=4, depth=1, dim_feedforward=64, dropout=0.0, use_act=False),
        checkpoint=None,
        num_games=1,
        max_moves=3,
        output_pgn=str(output),
        output_dataset=str(dataset_dir),
        device="cpu",
        inference_max_loops=1,
        temperature=1.0,
        mcts_num_simulations=4,
        train_after=False,
        eval_engine_path="fake",
        eval_depth=1,
        eval_multipv=2,
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

    output = tmp_path / "pgn"
    dataset_dir = tmp_path / "data"
    manifest_path = tmp_path / "manifest.json"
    cfg = SelfPlayConfig(
        model=ModelConfig(num_moves=4672, d_model=32, nhead=4, depth=1, dim_feedforward=64, dropout=0.0, use_act=False),
        checkpoint=None,
        num_games=1,
        max_moves=3,
        output_pgn=str(output),
        output_dataset=str(dataset_dir),
        output_manifest=str(manifest_path),
        device="cpu",
        inference_max_loops=1,
        temperature=1.0,
        mcts_num_simulations=4,
        train_after=True,
        train_config=str(tmp_path / "train.yaml"),
        train_every_batches=1,
        run_forever=True,
        max_batches=2,
    )

    generate_selfplay_games(cfg)

    assert len(train_calls) == 2
    manifest_entries = _load_manifest(manifest_path)
    datasets = [entry["dataset"] for entry in manifest_entries]
    assert len(datasets) == 2
    # First training pass sees only the first dataset
    assert train_calls[0] == [datasets[0]]
    # Second training pass includes both existing datasets (newest last)
    assert train_calls[1] == datasets
    for path_str in datasets:
        assert Path(path_str).exists()


def test_selfplay_prunes_old_pgns(tmp_path: Path) -> None:
    output = tmp_path / "selfplay"
    dataset_dir = tmp_path / "data"
    manifest_path = tmp_path / "manifest.json"
    cfg = SelfPlayConfig(
        model=ModelConfig(num_moves=4672, d_model=32, nhead=4, depth=1, dim_feedforward=64, dropout=0.0, use_act=False),
        checkpoint=None,
        num_games=1,
        max_moves=1,
        output_pgn=str(output),
        output_dataset=str(dataset_dir),
        output_manifest=str(manifest_path),
        device="cpu",
        inference_max_loops=1,
        temperature=1.0,
        mcts_num_simulations=4,
        train_after=False,
        train_every_batches=1,
        run_forever=True,
        max_batches=2,
        max_positions=1,
    )

    generate_selfplay_games(cfg)

    manifest_entries = _load_manifest(manifest_path)
    assert len(manifest_entries) == 1
    remaining_pgn = Path(manifest_entries[0]["pgn"])
    assert remaining_pgn.exists()
    assert "g00002" in remaining_pgn.name
    assert manifest_entries[0]["result"] in {"1-0", "0-1", "1/2-1/2"}
    assert "line_hash" in manifest_entries[0]


def test_selfplay_deduplicates_identical_sequences(monkeypatch, tmp_path: Path) -> None:
    def fake_search(self, board: chess.Board, *, add_noise: bool = True) -> Dict[chess.Move, int]:
        move = next(iter(board.legal_moves))
        return {move: 1}

    monkeypatch.setattr("chessref.train.selfplay.MCTS.search", fake_search, raising=False)

    output = tmp_path / "pgn"
    dataset_dir = tmp_path / "data"
    manifest_path = tmp_path / "manifest.json"
    cfg = SelfPlayConfig(
        model=ModelConfig(num_moves=4672, d_model=32, nhead=4, depth=1, dim_feedforward=64, dropout=0.0, use_act=False),
        checkpoint=None,
        num_games=1,
        max_moves=3,
        output_pgn=str(output),
        output_dataset=str(dataset_dir),
        output_manifest=str(manifest_path),
        device="cpu",
        inference_max_loops=1,
        temperature=0.0,
        mcts_num_simulations=4,
        mcts_dirichlet_epsilon=0.0,
        train_after=False,
        train_every_batches=1,
        run_forever=False,
        max_batches=None,
    )

    generate_selfplay_games(cfg)
    manifest_entries = _load_manifest(manifest_path)
    assert len(manifest_entries) == 1
    first_dataset = Path(manifest_entries[0]["dataset"])
    assert first_dataset.exists()
    first_line_hash = manifest_entries[0]["line_hash"]

    generate_selfplay_games(cfg)
    manifest_entries = _load_manifest(manifest_path)
    assert len(manifest_entries) == 1
    second_dataset = Path(manifest_entries[0]["dataset"])
    assert second_dataset.exists()
    assert second_dataset != first_dataset
    assert not first_dataset.exists()
    assert manifest_entries[0]["line_hash"] == first_line_hash


def test_selfplay_with_stockfish_opponent(monkeypatch, tmp_path: Path) -> None:
    class FakeEngine:
        def __init__(self) -> None:
            self.closed = False

        def configure(self, _options: Dict[str, int]) -> None:
            return None

        def play(self, board: chess.Board, _limit) -> chess.engine.PlayResult:
            move = next(iter(board.legal_moves))
            return chess.engine.PlayResult(move=move, ponder=None)

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr("chess.engine.SimpleEngine.popen_uci", lambda path: FakeEngine())
    monkeypatch.setattr("random.random", lambda: 0.0)

    output = tmp_path / "pgn"
    dataset_dir = tmp_path / "data"
    manifest_path = tmp_path / "manifest.json"
    cfg = SelfPlayConfig(
        model=ModelConfig(num_moves=4672, d_model=32, nhead=4, depth=1, dim_feedforward=64, dropout=0.0, use_act=False),
        checkpoint=None,
        num_games=1,
        max_moves=4,
        output_pgn=str(output),
        output_dataset=str(dataset_dir),
        output_manifest=str(manifest_path),
        device="cpu",
        inference_max_loops=1,
        temperature=0.1,
        opening_moves=0,
        mcts_num_simulations=4,
        stockfish=StockfishConfig(
            enabled=True,
            ratio=1.0,
            engine_path="fake",
            depth=1,
            skill_level=5,
            play_as_white=True,
            alternate_colors=False,
        ),
        train_after=False,
        train_every_batches=1,
        run_forever=False,
        max_batches=None,
    )

    generate_selfplay_games(cfg)

    manifest_entries = _load_manifest(manifest_path)
    assert len(manifest_entries) == 1
    entry = manifest_entries[0]
    assert entry["result"] in {"1-0", "0-1", "1/2-1/2"}
    assert "line_hash" in entry
    dataset_path = Path(entry["dataset"])
    assert dataset_path.exists()
    samples = torch.load(dataset_path, weights_only=False)
    assert len(samples) == entry["num_samples"]
    assert entry["num_samples"] <= entry["num_plies"]
