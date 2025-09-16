import chess
import pytest

from chessref.data.augment import TRANSFORMS, augment_position, transform_move


@pytest.fixture
def complex_board() -> chess.Board:
    fen = "r3k2r/pppq1ppp/2npbn2/3Np3/2P5/1P2PN2/PB3PPP/R2Q1RK1 w kq - 4 12"
    return chess.Board(fen)


def test_augment_position_returns_all_transforms(complex_board: chess.Board) -> None:
    seen = {tf.name for _, tf in augment_position(complex_board)}
    expected = {tf.name for tf in TRANSFORMS}
    assert seen == expected


def test_transform_preserves_legality(complex_board: chess.Board) -> None:
    legal_moves = list(complex_board.legal_moves)
    for transformed_board, tf in augment_position(complex_board):
        for move in legal_moves:
            mapped = transform_move(move, tf)
            assert transformed_board.is_legal(mapped)


def test_transform_handles_promotions() -> None:
    board = chess.Board()
    board.clear()
    board.set_piece_at(chess.A1, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(chess.H8, chess.Piece(chess.KING, chess.BLACK))
    board.set_piece_at(chess.G7, chess.Piece(chess.PAWN, chess.WHITE))
    board.turn = chess.WHITE

    move = chess.Move.from_uci("g7g8q")
    assert board.is_legal(move)

    for transformed_board, tf in augment_position(board):
        mapped = transform_move(move, tf)
        assert transformed_board.is_legal(mapped)


def test_transform_handles_en_passant() -> None:
    board = chess.Board()
    board.clear()
    board.set_piece_at(chess.A1, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(chess.H8, chess.Piece(chess.KING, chess.BLACK))
    board.set_piece_at(chess.A4, chess.Piece(chess.PAWN, chess.BLACK))
    board.set_piece_at(chess.B4, chess.Piece(chess.PAWN, chess.WHITE))
    board.turn = chess.BLACK
    board.ep_square = chess.B3

    move = chess.Move.from_uci("a4b3")
    assert board.is_legal(move)

    for transformed_board, tf in augment_position(board):
        mapped = transform_move(move, tf)
        assert transformed_board.is_legal(mapped)
