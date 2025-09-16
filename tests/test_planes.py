import chess
import torch

from chessref.data.planes import NUM_PLANES, PIECE_TO_PLANE, board_to_planes


def test_board_to_planes_shape_and_dtype() -> None:
    board = chess.Board()
    tensor = board_to_planes(board)
    assert tensor.shape == (NUM_PLANES, 8, 8)
    assert tensor.dtype == torch.float32


def test_piece_planes_and_metadata() -> None:
    board = chess.Board()
    tensor = board_to_planes(board)

    wp_plane = PIECE_TO_PLANE[(chess.WHITE, chess.PAWN)]
    bp_plane = PIECE_TO_PLANE[(chess.BLACK, chess.PAWN)]
    wk_plane = PIECE_TO_PLANE[(chess.WHITE, chess.KING)]

    assert tensor[wp_plane].sum().item() == 8
    assert tensor[bp_plane].sum().item() == 8
    # Verify a couple of specific squares.
    assert tensor[wp_plane, chess.square_rank(chess.A2), chess.square_file(chess.A2)] == 1
    assert tensor[bp_plane, chess.square_rank(chess.A7), chess.square_file(chess.A7)] == 1
    assert tensor[wk_plane, chess.square_rank(chess.E1), chess.square_file(chess.E1)] == 1

    # Side to move plane is filled with ones for white on move.
    side_to_move_plane = PIECE_TO_PLANE[(chess.BLACK, chess.KING)] + 1
    assert torch.all(tensor[side_to_move_plane] == 1)

    # Castling rights planes.
    castling_offset = side_to_move_plane + 1
    for offset in range(4):
        assert torch.all(tensor[castling_offset + offset] == 1)

    # En passant plane should be all zeros at the starting position.
    ep_plane = castling_offset + 4
    assert torch.count_nonzero(tensor[ep_plane]) == 0

    # Halfmove clock zero; fullmove plane records 1/400.
    halfmove_plane = ep_plane + 1
    fullmove_plane = halfmove_plane + 1
    assert torch.all(tensor[halfmove_plane] == 0)
    assert torch.allclose(tensor[fullmove_plane], torch.full((8, 8), 1 / 400, dtype=torch.float32))


def test_en_passant_and_move_counters() -> None:
    fen = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 3 2"
    board = chess.Board(fen)
    tensor = board_to_planes(board)
    side_to_move_plane = PIECE_TO_PLANE[(chess.BLACK, chess.KING)] + 1
    assert torch.all(tensor[side_to_move_plane] == 1)

    castling_offset = side_to_move_plane + 1
    ep_plane = castling_offset + 4
    # En passant marker along the e-file (file index 4).
    assert torch.allclose(tensor[ep_plane][:, 4], torch.ones(8))
    assert torch.count_nonzero(tensor[ep_plane][:, 4]) == 8

    halfmove_plane = ep_plane + 1
    fullmove_plane = halfmove_plane + 1
    assert torch.all(tensor[halfmove_plane] == 3 / 99)
    assert torch.all(tensor[fullmove_plane] == 2 / 400)
