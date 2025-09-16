"""Teacher target generation for policy/value supervision."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import chess
import torch

from chessref.moves.encoding import NUM_MOVES, encode_move
from .pgn_extract import TargetGenerator


@dataclass
class SelfPlayTargetGenerator(TargetGenerator):
    """Generate targets directly from recorded self-play games.

    The policy becomes a one-hot distribution on the played move and the value
    is the recorded game outcome from the perspective of the side to move.
    """

    num_moves: int = NUM_MOVES

    def generate(
        self,
        board: chess.Board,
        move: chess.Move,
        default_value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        policy = torch.zeros(self.num_moves, dtype=torch.float32)
        idx = encode_move(move, board)
        policy[idx] = 1.0
        return policy, default_value.clone()


class StockfishTargetGenerator(TargetGenerator):
    """Generate policy/value targets using a UCI engine such as Stockfish."""

    def __init__(
        self,
        engine_path: str,
        *,
        depth: Optional[int] = None,
        nodes: Optional[int] = None,
        multipv: int = 8,
        temperature: float = 1.0,
    ) -> None:
        import chess.engine  # imported lazily to avoid mandatory dependency

        self._engine = chess.engine.SimpleEngine.popen_uci(engine_path)
        self._limit = chess.engine.Limit(depth=depth, nodes=nodes)
        self._multipv = multipv
        self._temperature = temperature

    def __enter__(self) -> "StockfishTargetGenerator":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        if self._engine is not None:
            self._engine.close()
            self._engine = None

    def generate(
        self,
        board: chess.Board,
        move: chess.Move,
        default_value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._engine is None:
            raise RuntimeError("Stockfish engine has been closed")

        import math

        analysis = self._engine.analyse(board, self._limit, multipv=self._multipv)
        policy = torch.zeros(NUM_MOVES, dtype=torch.float32)
        weights = []

        for entry in analysis:
            pv = entry.get("pv")
            if not pv:
                continue
            candidate = pv[0]
            try:
                idx = encode_move(candidate, board)
            except ValueError:
                continue
            score = entry.get("score")
            if score is None:
                continue
            cp = score.pov(board.turn).score(mate_score=10000)
            weight = math.exp(cp / (100.0 * max(self._temperature, 1e-3)))
            weights.append((idx, weight))

        if not weights:
            # Fallback to the played move if the engine returns nothing.
            idx = encode_move(move, board)
            policy[idx] = 1.0
            return policy, default_value.clone()

        total = sum(weight for _, weight in weights)
        for idx, weight in weights:
            policy[idx] = weight / total

        top_score = analysis[0].get("score")
        if top_score is None:
            return policy, default_value.clone()

        pov_score = top_score.pov(board.turn)
        value = torch.tensor(pov_score.score(mate_score=10000) / 1000.0, dtype=torch.float32)
        value = value.clamp(min=-1.0, max=1.0)
        return policy, value


__all__ = [
    "SelfPlayTargetGenerator",
    "StockfishTargetGenerator",
]
