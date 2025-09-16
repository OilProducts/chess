import chess
import pytest
import torch

from chessref.data.engine_targets import SelfPlayTargetGenerator
from chessref.moves.encoding import encode_move


def test_selfplay_generator_creates_one_hot_policy() -> None:
    board = chess.Board()
    move = chess.Move.from_uci("e2e4")

    generator = SelfPlayTargetGenerator()
    default_value = torch.tensor(0.75, dtype=torch.float32)
    policy, value = generator.generate(board, move, default_value)

    assert pytest.approx(1.0) == policy.sum().item()
    assert torch.argmax(policy).item() == encode_move(move, board)
    assert pytest.approx(default_value.item()) == value.item()
