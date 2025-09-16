import chess
import pytest

from chessref.moves.encoding import NUM_MOVES, decode_index, encode_move


@pytest.mark.parametrize(
    "fen",
    [
        chess.STARTING_FEN,
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1",
        "r1bq1rk1/pp1nbppp/2n1p3/2ppP3/3P1P2/2PB1N2/PP1N2PP/R1BQ1RK1 w - - 0 10",
    ],
)
def test_encode_decode_round_trip(fen: str) -> None:
    board = chess.Board(fen)
    legal_moves = list(board.legal_moves)
    assert legal_moves, "Test position should have legal moves"

    for move in legal_moves:
        index = encode_move(move, board)
        assert 0 <= index < NUM_MOVES
        recovered = decode_index(index, board)
        assert recovered == move


def _underpromotion_board(color: chess.Color) -> chess.Board:
    board = chess.Board()
    board.clear()
    board.turn = color
    board.castling_rights = 0

    # Place kings to keep the position legal.
    board.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))

    if color == chess.WHITE:
        board.set_piece_at(chess.G7, chess.Piece(chess.PAWN, chess.WHITE))
        board.set_piece_at(chess.F8, chess.Piece(chess.ROOK, chess.BLACK))
        board.set_piece_at(chess.H8, chess.Piece(chess.BISHOP, chess.BLACK))
    else:
        board.set_piece_at(chess.C2, chess.Piece(chess.PAWN, chess.BLACK))
        board.set_piece_at(chess.B1, chess.Piece(chess.BISHOP, chess.WHITE))
        board.set_piece_at(chess.D1, chess.Piece(chess.ROOK, chess.WHITE))

    return board


@pytest.mark.parametrize("color", [chess.WHITE, chess.BLACK])
def test_encode_underpromotion_moves(color: chess.Color) -> None:
    board = _underpromotion_board(color)
    moves = list(board.legal_moves)

    # Ensure the three underpromotion lanes are present.
    uci_moves = {move.uci() for move in moves}
    if color == chess.WHITE:
        expected = {
            "g7g8n",
            "g7g8b",
            "g7g8r",
            "g7f8n",
            "g7f8b",
            "g7f8r",
            "g7h8n",
            "g7h8b",
            "g7h8r",
        }
    else:
        expected = {
            "c2c1n",
            "c2c1b",
            "c2c1r",
            "c2b1n",
            "c2b1b",
            "c2b1r",
            "c2d1n",
            "c2d1b",
            "c2d1r",
        }

    assert expected.issubset(uci_moves)

    for move in moves:
        idx = encode_move(move, board)
        recovered = decode_index(idx, board)
        assert recovered == move


def test_decode_invalid_index_raises() -> None:
    board = chess.Board()
    with pytest.raises(ValueError):
        decode_index(NUM_MOVES, board)


def test_decode_offboard_index_raises() -> None:
    board = chess.Board()
    # From square a1 (0), direction north-west (index 5) one step is off-board.
    offboard_index = 0 * 73 + (5 * 7)
    with pytest.raises(ValueError):
        decode_index(offboard_index, board)
