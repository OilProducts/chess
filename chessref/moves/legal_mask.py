"""Utilities for building legal move masks over the 4672-action space."""

from __future__ import annotations

from typing import Iterable, Optional

import chess
import torch

from .encoding import NUM_MOVES, encode_move


def legal_move_mask(board: chess.Board, *, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """Return a tensor mask with ``True`` entries for legal moves.

    Parameters
    ----------
    board:
        The chess position whose legal moves should be encoded.
    device:
        Optional target device for the returned tensor.
    dtype:
        Optional dtype. If omitted a boolean tensor is returned; otherwise the
        tensor is cast using ``mask.to(dtype=dtype)`` which is useful for loss
        masking.
    """

    mask = torch.zeros(NUM_MOVES, dtype=torch.bool, device=device)
    for move in board.legal_moves:
        idx = encode_move(move, board)
        mask[idx] = True

    if dtype is None:
        return mask
    return mask.to(dtype=dtype)


def legal_move_indices(board: chess.Board) -> Iterable[int]:
    """Yield the encoded indices for all legal moves in ``board``."""

    for move in board.legal_moves:
        yield encode_move(move, board)


__all__ = ["legal_move_mask", "legal_move_indices"]
