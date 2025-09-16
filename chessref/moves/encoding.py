"""AlphaZero-style move indexing utilities.

The move space follows the 8x8x73 layout popularised by AlphaZero:

* 56 queen-like ray moves (8 directions x up to 7 squares).
* 8 knight jumps.
* 9 pawn underpromotion moves (3 piece types x 3 directions).

Moves are indexed from the perspective of the side to move. When the board
turn is black we rotate the move by 180° (using :func:`chess.square_mirror`).
This keeps the encoding symmetric for both players and matches common
iterative-refinement setups.
"""

from __future__ import annotations

from typing import Dict, Iterator, List, Optional, Tuple

import chess

NUM_SQUARES = 64
PLANES_PER_SQUARE = 73
NUM_MOVES = NUM_SQUARES * PLANES_PER_SQUARE  # 4672

# Direction vectors expressed as (file delta, rank delta) in white's orientation.
_RAY_DIRECTIONS: Tuple[Tuple[int, int], ...] = (
    (0, 1),   # north
    (0, -1),  # south
    (1, 0),   # east
    (-1, 0),  # west
    (1, 1),   # north-east
    (-1, 1),  # north-west
    (1, -1),  # south-east
    (-1, -1), # south-west
)

_KNIGHT_OFFSETS: Tuple[Tuple[int, int], ...] = (
    (1, 2),
    (2, 1),
    (2, -1),
    (1, -2),
    (-1, -2),
    (-2, -1),
    (-2, 1),
    (-1, 2),
)

# Underpromotion directions in white's orientation: forward, capture-left, capture-right.
_PROMOTION_OFFSETS: Tuple[Tuple[int, int], ...] = (
    (0, 1),
    (-1, 1),
    (1, 1),
)
_PROMOTION_PIECES: Tuple[chess.PieceType, ...] = (
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
)

MoveKey = Tuple[int, int, Optional[chess.PieceType]]

UCI2IDX: Dict[MoveKey, int]
IDX2UCI: List[Optional[MoveKey]]


def _target_square(square: int, df: int, dr: int, step: int = 1) -> Optional[int]:
    """Return the square reached from ``square`` by moving ``step`` times ``(df, dr)``."""

    file = chess.square_file(square) + df * step
    rank = chess.square_rank(square) + dr * step
    if 0 <= file < 8 and 0 <= rank < 8:
        return chess.square(file, rank)
    return None


def _register_move(mapping: Dict[MoveKey, int], inverse: List[Optional[MoveKey]], key: MoveKey, idx: int) -> None:
    """Store the move key -> index mapping and populate the inverse table."""

    mapping[key] = idx
    if inverse[idx] is None:
        inverse[idx] = key


def _build_maps() -> Tuple[Dict[MoveKey, int], List[Optional[MoveKey]]]:
    mapping: Dict[MoveKey, int] = {}
    inverse: List[Optional[MoveKey]] = [None] * NUM_MOVES

    for from_sq in range(NUM_SQUARES):
        base = from_sq * PLANES_PER_SQUARE

        # 56 queen-like moves.
        for direction, (df, dr) in enumerate(_RAY_DIRECTIONS):
            for step in range(1, 8):
                plane = direction * 7 + (step - 1)
                idx = base + plane
                target = _target_square(from_sq, df, dr, step)
                if target is None:
                    continue

                key = (from_sq, target, None)
                _register_move(mapping, inverse, key, idx)

                # Queen promotions (single-step pawn moves) share the same plane.
                if step == 1:
                    promotion_key = (from_sq, target, chess.QUEEN)
                    mapping[promotion_key] = idx

        # 8 knight moves.
        for knight_idx, (df, dr) in enumerate(_KNIGHT_OFFSETS):
            plane = 56 + knight_idx
            idx = base + plane
            target = _target_square(from_sq, df, dr)
            if target is None:
                continue
            key = (from_sq, target, None)
            _register_move(mapping, inverse, key, idx)

        # 9 underpromotion moves (only meaningful on the seventh rank in white's view).
        if chess.square_rank(from_sq) == 6:
            for promo_idx, piece in enumerate(_PROMOTION_PIECES):
                for move_idx, (df, dr) in enumerate(_PROMOTION_OFFSETS):
                    plane = 56 + 8 + promo_idx * 3 + move_idx
                    idx = base + plane
                    target = _target_square(from_sq, df, dr)
                    if target is None:
                        continue
                    key = (from_sq, target, piece)
                    _register_move(mapping, inverse, key, idx)

    return mapping, inverse


UCI2IDX, IDX2UCI = _build_maps()


def _canonicalise_move(move: chess.Move, color: chess.Color) -> MoveKey:
    """Return the canonical (from, to, promotion) tuple for ``move``.

    The board is viewed from the perspective of ``color``; if ``color`` is black we
    mirror the coordinates so that forward always increases the rank.
    """

    from_sq, to_sq, promotion = move.from_square, move.to_square, move.promotion
    if color == chess.BLACK:
        from_sq = chess.square_mirror(from_sq)
        to_sq = chess.square_mirror(to_sq)
    return from_sq, to_sq, promotion


def encode_move(move: chess.Move, board: chess.Board) -> int:
    """Encode a legal move into the 0..4671 index space.

    Parameters
    ----------
    move:
        The move to encode. It must be legal in ``board``.
    board:
        The board before the move is played. Only the side-to-move flag is
        inspected; the board is not modified.
    """

    key = _canonicalise_move(move, board.turn)
    try:
        return UCI2IDX[key]
    except KeyError as exc:  # pragma: no cover - defensive: should not occur for legal moves
        raise ValueError(f"Move {move.uci()} cannot be encoded in the 4672 move space") from exc


def decode_index(index: int, board: chess.Board) -> chess.Move:
    """Decode an index back into a :class:`chess.Move` for ``board``'s side to move."""

    if not 0 <= index < NUM_MOVES:
        raise ValueError(f"Index {index} outside valid range 0..{NUM_MOVES - 1}")

    entry = IDX2UCI[index]
    if entry is None:
        raise ValueError(f"Index {index} does not correspond to a legal move target")

    from_sq, to_sq, promotion = entry
    if board.turn == chess.BLACK:
        from_sq = chess.square_mirror(from_sq)
        to_sq = chess.square_mirror(to_sq)

    move = chess.Move(from_sq, to_sq, promotion=promotion)

    # Promotions to queens share the same plane as the corresponding single-step
    # queen move. Recover the promotion flag when required so that equality tests
    # and legality checks succeed.
    if promotion is None:
        piece = board.piece_at(move.from_square)
        if piece is not None and piece.piece_type == chess.PAWN:
            rank = chess.square_rank(move.to_square)
            if rank in (0, 7):
                move = chess.Move(move.from_square, move.to_square, promotion=chess.QUEEN)

    return move


def iter_legal_indices(board: chess.Board) -> Iterator[int]:
    """Yield the indices for all legal moves in ``board``."""

    for move in board.legal_moves:
        yield encode_move(move, board)


__all__ = [
    "NUM_MOVES",
    "PLANES_PER_SQUARE",
    "encode_move",
    "decode_index",
    "iter_legal_indices",
    "UCI2IDX",
    "IDX2UCI",
]
