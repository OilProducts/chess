"""Utilities for extracting training samples from PGN files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence

import chess
import chess.pgn
import torch

from chessref.data.planes import board_to_planes
from chessref.moves.encoding import NUM_MOVES


@dataclass
class PositionRecord:
    """Metadata for a single board position extracted from a PGN game."""

    fen: str
    move_uci: str
    white_result: float
    ply: int
    game_index: int

    def board(self) -> chess.Board:
        return chess.Board(self.fen)

    def move(self) -> chess.Move:
        return chess.Move.from_uci(self.move_uci)


@dataclass
class TrainingSample:
    """Fully materialised tensors ready for dataset ingestion."""

    planes: torch.Tensor  # [NUM_PLANES, 8, 8]
    policy: torch.Tensor  # [NUM_MOVES]
    value: torch.Tensor   # scalar tensor
    metadata: PositionRecord


def _result_to_float(result: str) -> Optional[float]:
    if result == "1-0":
        return 1.0
    if result == "0-1":
        return 0.0
    if result in {"1/2-1/2", "1/2"}:
        return 0.5
    return None


def iter_position_records(
    pgn_path: Path,
    *,
    max_games: Optional[int] = None,
) -> Iterator[PositionRecord]:
    """Yield :class:`PositionRecord` instances from ``pgn_path``."""

    with pgn_path.open("r", encoding="utf-8") as handle:
        game_index = 0
        while True:
            game = chess.pgn.read_game(handle)
            if game is None:
                break
            if max_games is not None and game_index >= max_games:
                break

            result_str = game.headers.get("Result", "*")
            result = _result_to_float(result_str)
            if result is None:
                game_index += 1
                continue  # skip games without a decisive/draw result

            board = game.board()
            for ply, move in enumerate(game.mainline_moves()):
                record = PositionRecord(
                    fen=board.fen(),
                    move_uci=move.uci(),
                    white_result=result,
                    ply=ply,
                    game_index=game_index,
                )
                yield record
                board.push(move)

            game_index += 1


def extract_position_records(paths: Sequence[Path], *, max_games: Optional[int] = None) -> List[PositionRecord]:
    """Collect position records from multiple PGN files."""

    records: List[PositionRecord] = []
    for path in paths:
        records.extend(list(iter_position_records(path, max_games=max_games)))
    return records


def build_training_samples(
    records: Iterable[PositionRecord],
    target_generator: "TargetGenerator",
) -> List[TrainingSample]:
    """Convert position records into tensors using ``target_generator``."""

    samples: List[TrainingSample] = []
    for record in records:
        board = record.board()
        move = record.move()

        planes = board_to_planes(board)
        value = torch.tensor(
            record.white_result if board.turn == chess.WHITE else 1.0 - record.white_result,
            dtype=torch.float32,
        )
        policy, teacher_value = target_generator.generate(board, move, value)

        if policy.shape != (NUM_MOVES,):
            raise ValueError("Policy tensor must have shape (NUM_MOVES,)")
        if not torch.isclose(policy.sum(), policy.new_tensor(1.0), atol=1e-4):
            raise ValueError("Policy distribution must sum to 1.0")

        samples.append(
            TrainingSample(
                planes=planes,
                policy=policy,
                value=teacher_value,
                metadata=record,
            )
        )

    return samples


class TargetGenerator:
    """Protocol for teacher target generation."""

    def generate(
        self,
        board: chess.Board,
        move: chess.Move,
        default_value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


__all__ = [
    "PositionRecord",
    "TrainingSample",
    "iter_position_records",
    "extract_position_records",
    "build_training_samples",
    "TargetGenerator",
]
