"""Lightweight match evaluation for the chess refiner."""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import chess
import torch
from omegaconf import OmegaConf

from chessref.data.planes import board_to_planes
from chessref.inference.predict import refine_predict
from chessref.model.refiner import IterativeRefiner
from chessref.moves.encoding import decode_index
from chessref.moves.legal_mask import legal_move_mask
from chessref.train.train_supervised import ModelConfig, _select_device


@dataclass
class MatchConfig:
    model: ModelConfig
    checkpoint: Optional[str] = None
    num_games: int = 1
    max_moves: int = 200
    opponent: str = "random"
    device: str = "auto"
    inference_max_loops: int = 4


@dataclass
class MatchResult:
    wins: int
    losses: int
    draws: int


def _load_config(path: Path) -> MatchConfig:
    cfg = OmegaConf.load(path)
    model = ModelConfig(**OmegaConf.to_object(cfg.model))
    checkpoint = cfg.get("checkpoint")
    num_games = cfg.get("num_games", 1)
    max_moves = cfg.get("max_moves", 200)
    opponent = cfg.get("opponent", "random")
    device = cfg.get("device", "auto")
    inference_max_loops = cfg.get("inference_max_loops", 4)
    return MatchConfig(
        model=model,
        checkpoint=checkpoint,
        num_games=num_games,
        max_moves=max_moves,
        opponent=opponent,
        device=device,
        inference_max_loops=inference_max_loops,
    )


def _load_model(cfg: MatchConfig, device: torch.device) -> IterativeRefiner:
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
    model.eval()
    return model


def _choose_model_move(model: IterativeRefiner, board: chess.Board, cfg: MatchConfig, device: torch.device) -> chess.Move:
    planes = board_to_planes(board).unsqueeze(0).to(device)
    legal_mask = legal_move_mask(board).unsqueeze(0).to(device)
    result = refine_predict(
        model,
        planes,
        legal_mask,
        max_loops=cfg.inference_max_loops,
        min_loops=1,
        use_act=cfg.model.use_act,
    )
    policy = result.final_policy[0].cpu()
    idx = int(torch.argmax(policy))
    move = decode_index(idx, board)
    if move not in board.legal_moves:
        move = random.choice(list(board.legal_moves))
    return move


def _choose_opponent_move(board: chess.Board, opponent: str) -> chess.Move:
    if opponent == "random":
        return random.choice(list(board.legal_moves))
    raise ValueError(f"Unknown opponent '{opponent}'")


@torch.no_grad()
def evaluate_matches(cfg: MatchConfig) -> MatchResult:
    device = _select_device(cfg.device)
    model = _load_model(cfg, device)

    wins = losses = draws = 0

    for game_idx in range(cfg.num_games):
        board = chess.Board()
        move_count = 0
        while not board.is_game_over(claim_draw=True) and move_count < cfg.max_moves:
            if board.turn == chess.WHITE:
                move = _choose_model_move(model, board, cfg, device)
            else:
                move = _choose_opponent_move(board, cfg.opponent)
            board.push(move)
            move_count += 1

        if board.is_game_over():
            result = board.result()
            if result == "1-0":
                wins += 1
            elif result == "0-1":
                losses += 1
            else:
                draws += 1
        else:
            draws += 1

    return MatchResult(wins=wins, losses=losses, draws=draws)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run quick model vs random matches")
    parser.add_argument("--config", type=Path, default=Path("configs/match.yaml"))
    args = parser.parse_args(argv)

    cfg = _load_config(args.config)
    result = evaluate_matches(cfg)
    print(f"wins={result.wins} losses={result.losses} draws={result.draws}")


if __name__ == "__main__":
    main()


__all__ = ["evaluate_matches", "MatchConfig", "MatchResult"]
