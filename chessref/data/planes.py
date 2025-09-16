"""Board-to-tensor encodings for chess positions."""

from __future__ import annotations

from typing import Dict

import chess
import torch

PIECE_TYPES = (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING)
NUM_PIECE_PLANES = len(PIECE_TYPES) * 2  # white + black
EXTRA_PLANES = 8  # side to move, castling rights (4), en passant file (8 one-hot), move counters (2)
NUM_PLANES = NUM_PIECE_PLANES + EXTRA_PLANES

PIECE_TO_PLANE: Dict[tuple[chess.Color, chess.PieceType], int] = {}
for color_index, color in enumerate((chess.WHITE, chess.BLACK)):
    for offset, piece_type in enumerate(PIECE_TYPES):
        PIECE_TO_PLANE[(color, piece_type)] = color_index * len(PIECE_TYPES) + offset


def board_to_planes(board: chess.Board) -> torch.Tensor:
    """Convert ``board`` into a tensor of shape ``[NUM_PLANES, 8, 8]``."""

    tensor = torch.zeros(NUM_PLANES, 8, 8, dtype=torch.float32)

    for square, piece in board.piece_map().items():
        plane = PIECE_TO_PLANE[(piece.color, piece.piece_type)]
        rank = chess.square_rank(square)
        file = chess.square_file(square)
        tensor[plane, rank, file] = 1.0

    # Side to move plane.
    tensor[NUM_PIECE_PLANES, :, :] = 1.0 if board.turn == chess.WHITE else 0.0

    # Castling rights planes (white king-side, white queen-side, black king-side, black queen-side).
    castling_offset = NUM_PIECE_PLANES + 1
    tensor[castling_offset + 0, :, :] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    tensor[castling_offset + 1, :, :] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    tensor[castling_offset + 2, :, :] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    tensor[castling_offset + 3, :, :] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0

    # En passant file encoded as one-hot across a plane.
    ep_plane = castling_offset + 4
    if board.ep_square is not None:
        file = chess.square_file(board.ep_square)
        tensor[ep_plane, :, file] = 1.0

    # Half-move clock (scaled to [0,1] over a typical range of 0-99).
    halfmove_plane = ep_plane + 1
    tensor[halfmove_plane, :, :] = min(board.halfmove_clock, 99) / 99.0

    # Full-move number (scaled similarly; the absolute value is useful for chronological context).
    fullmove_plane = halfmove_plane + 1
    tensor[fullmove_plane, :, :] = min(board.fullmove_number, 400) / 400.0

    return tensor


__all__ = [
    "board_to_planes",
    "NUM_PLANES",
    "PIECE_TYPES",
    "PIECE_TO_PLANE",
]
