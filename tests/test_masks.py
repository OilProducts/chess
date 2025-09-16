import chess
import torch

from chessref.moves.encoding import encode_move
from chessref.moves.legal_mask import legal_move_indices, legal_move_mask


_TEST_FENS = [
    chess.STARTING_FEN,
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1",
    "r1bq1rk1/pp1nbppp/2n1p3/2ppP3/3P1P2/2PB1N2/PP1N2PP/R1BQ1RK1 w - - 0 10",
]


def test_legal_move_mask_matches_legal_moves() -> None:
    for fen in _TEST_FENS:
        board = chess.Board(fen)
        mask = legal_move_mask(board)
        assert mask.dtype == torch.bool
        true_indices = mask.nonzero(as_tuple=False).flatten().tolist()

        legal_moves = list(board.legal_moves)
        encoded_moves = sorted(encode_move(move, board) for move in legal_moves)
        assert sorted(true_indices) == encoded_moves


def test_legal_move_mask_dtype_cast() -> None:
    board = chess.Board()
    mask_float = legal_move_mask(board, dtype=torch.float32)
    assert mask_float.dtype == torch.float32
    assert mask_float.sum().item() == len(list(board.legal_moves))


def test_legal_move_indices_generator() -> None:
    board = chess.Board()
    indices_from_generator = sorted(legal_move_indices(board))
    indices_from_mask = sorted(mask_idx.item() for mask_idx in legal_move_mask(board).nonzero(as_tuple=True)[0])
    assert indices_from_generator == indices_from_mask
