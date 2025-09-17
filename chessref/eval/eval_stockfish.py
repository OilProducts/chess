"""Evaluate model move quality against Stockfish."""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import chess
import torch
from omegaconf import OmegaConf

from chessref.data.pgn_extract import extract_position_records
from chessref.data.planes import board_to_planes
from chessref.inference.predict import InferenceResult, InferenceStep, refine_predict
from chessref.model.refiner import IterativeRefiner
from chessref.moves.encoding import NUM_MOVES, decode_index
from chessref.moves.legal_mask import legal_move_mask
from chessref.train.train_supervised import ModelConfig, _select_device


@dataclass
class InferenceConfig:
    max_loops: int = 1
    min_loops: int = 1
    use_act: bool = False
    entropy_threshold: Optional[float] = None
    act_threshold: float = 0.5


@dataclass
class StockfishLimit:
    depth: Optional[int] = 12
    nodes: Optional[int] = None
    movetime: Optional[int] = None  # milliseconds


@dataclass
class EvalConfig:
    model: ModelConfig
    pgn_paths: Sequence[str]
    stockfish_path: str
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    limit: StockfishLimit = field(default_factory=StockfishLimit)
    multipv: int = 5
    max_positions: Optional[int] = None
    device: str = "auto"
    checkpoint: Optional[str] = None


@dataclass
class StockfishEvalResult:
    mean_cp_loss: float
    top1_match: float
    evaluated_positions: int


def _load_config(path: Path) -> EvalConfig:
    cfg = OmegaConf.load(path)
    model_cfg = ModelConfig(**OmegaConf.to_object(cfg.model))
    inference_cfg = InferenceConfig(**OmegaConf.to_object(cfg.get("inference", {})))
    limit_cfg = StockfishLimit(**OmegaConf.to_object(cfg.get("limit", {})))
    pgn_paths = OmegaConf.to_object(cfg.get("pgn_paths", []))
    stockfish_path = cfg.get("stockfish_path")
    multipv = cfg.get("multipv", 5)
    max_positions = cfg.get("max_positions")
    device = cfg.get("device", "auto")
    checkpoint = cfg.get("checkpoint")
    return EvalConfig(
        model=model_cfg,
        pgn_paths=pgn_paths,
        stockfish_path=stockfish_path,
        inference=inference_cfg,
        limit=limit_cfg,
        multipv=multipv,
        max_positions=max_positions,
        device=device,
        checkpoint=checkpoint,
    )


def _build_model(cfg: EvalConfig, device: torch.device) -> IterativeRefiner:
    model = IterativeRefiner(
        num_moves=cfg.model.num_moves,
        d_model=cfg.model.d_model,
        nhead=cfg.model.nhead,
        depth=cfg.model.depth,
        dim_feedforward=cfg.model.dim_feedforward,
        dropout=cfg.model.dropout,
        use_act=cfg.model.use_act,
    ).to(device)
    if cfg.checkpoint:
        state = torch.load(cfg.checkpoint, map_location=device)
        if "model" in state:
            state = state["model"]
        model.load_state_dict(state)
    return model.eval()


def _score_to_cp(score: chess.engine.PovScore, mate_score: int = 10000) -> float:
    cp = score.cp
    if cp is not None:
        return float(cp)
    mate = score.mate
    if mate is None:
        return 0.0
    sign = 1 if mate > 0 else -1
    return sign * mate_score


def _evaluate_move_difference(
    board: chess.Board,
    model_policy: torch.Tensor,
    engine,
    cfg: EvalConfig,
) -> tuple[float, bool]:
    limit_kwargs = {}
    if cfg.limit.depth is not None:
        limit_kwargs["depth"] = cfg.limit.depth
    if cfg.limit.nodes is not None:
        limit_kwargs["nodes"] = cfg.limit.nodes
    if cfg.limit.movetime is not None:
        limit_kwargs["movetime"] = cfg.limit.movetime
    limit = chess.engine.Limit(**limit_kwargs)

    analysis = engine.analyse(board, limit, multipv=cfg.multipv)
    if not analysis:
        return 0.0, False

    best_entry = analysis[0]
    best_move = best_entry.get("pv", [None])[0]
    best_score = _score_to_cp(best_entry["score"].pov(board.turn))

    idx = int(torch.argmax(model_policy).item())
    model_move = decode_index(idx, board)
    if model_move not in board.legal_moves:
        model_move = best_move or next(iter(board.legal_moves))

    match = model_move == best_move

    model_score: Optional[float] = None
    for entry in analysis:
        pv = entry.get("pv")
        if pv and pv[0] == model_move:
            model_score = _score_to_cp(entry["score"].pov(board.turn))
            break

    if model_score is None:
        board.push(model_move)
        if board.is_game_over(claim_draw=True):
            result = board.result(claim_draw=True)
            board.pop()
            if result == "1-0":
                model_score = 10000.0
            elif result == "0-1":
                model_score = -10000.0
            else:
                model_score = 0.0
        else:
            child_analysis = engine.analyse(board, limit)
            entry = child_analysis[0] if isinstance(child_analysis, list) else child_analysis
            score = _score_to_cp(entry["score"].pov(board.turn))
            board.pop()
            model_score = -score

    cp_loss = max(0.0, best_score - model_score)
    return cp_loss, match


@torch.no_grad()
def evaluate(cfg: EvalConfig) -> StockfishEvalResult:
    if not cfg.pgn_paths:
        raise ValueError("pgn_paths must be provided for Stockfish evaluation")
    if not cfg.stockfish_path:
        raise ValueError("stockfish_path must be provided")

    device = _select_device(cfg.device)
    model = _build_model(cfg, device)

    records = []
    for path in cfg.pgn_paths:
        records.extend(extract_position_records([Path(path)]))
    if cfg.max_positions is not None and len(records) > cfg.max_positions:
        records = random.sample(records, cfg.max_positions)

    import chess.engine

    engine = chess.engine.SimpleEngine.popen_uci(cfg.stockfish_path)

    total_loss = 0.0
    total_match = 0
    evaluated = 0

    try:
        for record in records:
            board = record.board()
            legal_mask = legal_move_mask(board).unsqueeze(0).to(device)
            planes = board_to_planes(board).unsqueeze(0).to(device)
            result = refine_predict(
                model,
                planes,
                legal_mask,
                max_loops=cfg.inference.max_loops,
                min_loops=cfg.inference.min_loops,
                use_act=cfg.inference.use_act,
                act_threshold=cfg.inference.act_threshold,
                entropy_threshold=cfg.inference.entropy_threshold,
            )
            policy = result.final_policy[0].cpu()
            loss, match = _evaluate_move_difference(board, policy, engine, cfg)
            total_loss += loss
            total_match += int(match)
            evaluated += 1
    finally:
        engine.close()

    mean_loss = total_loss / evaluated if evaluated else 0.0
    top1 = total_match / evaluated if evaluated else 0.0
    return StockfishEvalResult(mean_cp_loss=mean_loss, top1_match=top1, evaluated_positions=evaluated)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate model vs Stockfish")
    parser.add_argument("--config", type=Path, default=Path("configs/eval_stockfish.yaml"))
    args = parser.parse_args(argv)

    cfg = _load_config(args.config)
    result = evaluate(cfg)
    print(
        f"positions={result.evaluated_positions} mean_cp_loss={result.mean_cp_loss:.2f} top1={result.top1_match:.3f}"
    )


if __name__ == "__main__":
    main()


__all__ = ["EvalConfig", "StockfishEvalResult", "evaluate"]
