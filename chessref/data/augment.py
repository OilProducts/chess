"""Symmetry-preserving chess augmentations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, Tuple

import chess

SquareTransform = Callable[[int], int]
BitboardTransform = Callable[[int], int]


def _mirror_file(square: int) -> int:
    file = 7 - chess.square_file(square)
    rank = chess.square_rank(square)
    return chess.square(file, rank)


@dataclass(frozen=True)
class BoardTransform:
    """A board-space symmetry transform that preserves move legality."""

    name: str
    bitboard_fn: BitboardTransform
    square_fn: SquareTransform

    def apply(self, board: chess.Board) -> chess.Board:
        return board.transform(self.bitboard_fn)

    def map_square(self, square: int) -> int:
        return self.square_fn(square)


TRANSFORMS: Tuple[BoardTransform, ...] = (
    BoardTransform("identity", lambda bb: bb, lambda sq: sq),
    BoardTransform("flip_horizontal", chess.flip_horizontal, _mirror_file),
)


def transform_move(move: chess.Move, transform: BoardTransform) -> chess.Move:
    """Map a move through ``transform``."""

    from_sq = transform.map_square(move.from_square)
    to_sq = transform.map_square(move.to_square)
    return chess.Move(from_sq, to_sq, promotion=move.promotion)


def augment_position(board: chess.Board) -> Iterator[Tuple[chess.Board, BoardTransform]]:
    """Yield symmetrically transformed boards with metadata about the transform."""

    for tf in TRANSFORMS:
        yield tf.apply(board), tf


__all__ = ["BoardTransform", "TRANSFORMS", "augment_position", "transform_move"]
