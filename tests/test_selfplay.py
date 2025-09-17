from pathlib import Path

import torch

from chessref.train.selfplay import SelfPlayConfig, generate_selfplay_games
from chessref.train.train_supervised import ModelConfig


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
    )

    path = generate_selfplay_games(cfg)
    assert path.exists()
    content = path.read_text().strip()
    assert "Result" in content
    assert dataset_path.exists()
    samples = torch.load(dataset_path, weights_only=False)
    assert samples
