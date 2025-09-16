import chess
import torch

from chessref.data.planes import board_to_planes, NUM_PLANES
from chessref.inference.predict import refine_predict
from chessref.model.refiner import IterativeRefiner
from chessref.moves.legal_mask import legal_move_mask
from chessref.moves.encoding import NUM_MOVES


def test_refine_predict_masks_illegal_moves() -> None:
    model = IterativeRefiner(num_moves=NUM_MOVES, d_model=32, nhead=4, depth=1, dim_feedforward=64, dropout=0.0, use_act=False)
    board = chess.Board()
    planes = board_to_planes(board).unsqueeze(0)
    legal = legal_move_mask(board).unsqueeze(0)

    result = refine_predict(model, planes, legal, max_loops=2, min_loops=1)
    final_policy = result.final_policy

    assert final_policy.shape == (1, NUM_MOVES)
    assert torch.all(final_policy[:, ~legal[0]] == 0)
    assert torch.allclose(final_policy.sum(dim=-1), torch.ones(1))
