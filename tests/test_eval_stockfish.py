from pathlib import Path

import chess
import torch

from chessref.eval import eval_stockfish
from chessref.eval.eval_stockfish import EvalConfig, InferenceConfig, StockfishEvalResult
from chessref.train.train_supervised import ModelConfig
from chessref.moves.encoding import NUM_MOVES, encode_move
from chessref.data.pgn_extract import PositionRecord


class FakeEngine:
    def __init__(self):
        self.closed = False

    def analyse(self, board: chess.Board, limit, multipv: int = 1):
        moves = list(board.legal_moves)
        pref = chess.Move.from_uci("e2e4")
        best_move = pref if pref in moves else moves[0]
        second_move = moves[1] if len(moves) > 1 else moves[0]
        entries = []
        score_best = chess.engine.PovScore(chess.engine.Cp(50), chess.WHITE if board.turn else chess.BLACK)
        entries.append({"pv": [best_move], "score": score_best})
        score_second = chess.engine.PovScore(chess.engine.Cp(20), chess.WHITE if board.turn else chess.BLACK)
        entries.append({"pv": [second_move], "score": score_second})
        return entries[:multipv]

    def close(self):
        self.closed = True


def test_stockfish_evaluation(monkeypatch, tmp_path: Path) -> None:
    game_pgn = """
[Event "Test"]
[White "Test"]
[Black "Test"]
[Result "*"]

1. e4
"""
    pgn_path = tmp_path / "game.pgn"
    pgn_path.write_text(game_pgn, encoding="utf-8")

    pending_boards = []

    def fake_extract(records_paths):
        board = chess.Board()
        record = PositionRecord(
            fen=board.fen(),
            move_uci="e2e4",
            white_result=1.0,
            ply=0,
            game_index=0,
        )
        pending_boards.clear()
        pending_boards.append(board)
        return [record]

    def fake_refine(model, planes, legal_mask, **kwargs):
        board = pending_boards.pop(0)
        policy = torch.zeros(NUM_MOVES)
        idx = encode_move(chess.Move.from_uci("e2e4"), board)
        policy[idx] = 1.0
        step = eval_stockfish.InferenceStep(policy=policy.unsqueeze(0), value=torch.zeros(1), halt=None)
        return eval_stockfish.InferenceResult(steps=[step])

    def fake_build_model(cfg, device):
        return object()

    monkeypatch.setattr(eval_stockfish, "extract_position_records", fake_extract)
    monkeypatch.setattr(eval_stockfish, "refine_predict", fake_refine)
    monkeypatch.setattr(eval_stockfish, "_build_model", fake_build_model)
    monkeypatch.setattr("chess.engine.SimpleEngine.popen_uci", lambda path: FakeEngine())

    cfg = EvalConfig(
        model=ModelConfig(num_moves=NUM_MOVES, d_model=32, nhead=4, depth=1, dim_feedforward=64, dropout=0.0, use_act=False),
        pgn_paths=[str(pgn_path)],
        stockfish_path="fake",
        inference=InferenceConfig(max_loops=1, min_loops=1, use_act=False),
        limit=eval_stockfish.StockfishLimit(depth=1, nodes=None, movetime=None),
        multipv=2,
        max_positions=1,
        device="cpu",
        checkpoint=None,
    )

    result = eval_stockfish.evaluate(cfg)
    assert isinstance(result, StockfishEvalResult)
    assert result.evaluated_positions == 1
    assert result.top1_match == 1.0
    assert result.mean_cp_loss == 0.0
