"""Self-play game generation for bootstrapping datasets."""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import chess
import chess.pgn
import torch
from omegaconf import OmegaConf

from chessref.data.planes import board_to_planes
from chessref.inference.predict import refine_predict
from chessref.model.refiner import IterativeRefiner
from chessref.moves.encoding import decode_index
from chessref.moves.legal_mask import legal_move_mask
from chessref.train.train_supervised import ModelConfig, _select_device


@dataclass
class SelfPlayConfig:
    model: ModelConfig
    checkpoint: Optional[str] = None
    num_games: int = 1
    max_moves: int = 200
    output_pgn: str = "selfplay.pgn"
    device: str = "auto"
    inference_max_loops: int = 4
    temperature: float = 1.0


def _load_config(path: Path) -> SelfPlayConfig:
    cfg = OmegaConf.load(path)
    model = ModelConfig(**OmegaConf.to_object(cfg.model))
    checkpoint = cfg.get("checkpoint")
    num_games = cfg.get("num_games", 1)
    max_moves = cfg.get("max_moves", 200)
    output_pgn = cfg.get("output_pgn", "selfplay.pgn")
    device = cfg.get("device", "auto")
    inference_max_loops = cfg.get("inference_max_loops", 4)
    temperature = cfg.get("temperature", 1.0)
    return SelfPlayConfig(
        model=model,
        checkpoint=checkpoint,
        num_games=num_games,
        max_moves=max_moves,
        output_pgn=output_pgn,
        device=device,
        inference_max_loops=inference_max_loops,
        temperature=temperature,
    )


def _load_model(cfg: SelfPlayConfig, device: torch.device) -> IterativeRefiner:
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
        checkpoint = torch.load(cfg.checkpoint, map_location=device)
        state = checkpoint.get("model", checkpoint)
        model.load_state_dict(state)
    model.eval()
    return model


def _sample_policy(policy: torch.Tensor, legal_mask: torch.Tensor, temperature: float) -> int:
    legal_indices = torch.nonzero(legal_mask, as_tuple=False).squeeze(-1)
    if legal_indices.numel() == 0:
        raise RuntimeError("No legal moves available for sampling")

    logits = policy[legal_indices]
    probs = logits.clone()
    if temperature <= 1e-6:
        idx = legal_indices[torch.argmax(probs)].item()
        return idx

    adjusted = torch.pow(probs, 1.0 / max(temperature, 1e-6))
    if torch.allclose(adjusted.sum(), torch.tensor(0.0)):
        adjusted = torch.ones_like(adjusted) / adjusted.numel()
    else:
        adjusted = adjusted / adjusted.sum()
    chosen = torch.multinomial(adjusted, num_samples=1)
    return legal_indices[chosen].item()


def _choose_move(model: IterativeRefiner, board: chess.Board, cfg: SelfPlayConfig, device: torch.device) -> chess.Move:
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
    mask_cpu = legal_mask[0].cpu()
    move_index = _sample_policy(policy, mask_cpu, cfg.temperature)
    move = decode_index(move_index, board)
    if move not in board.legal_moves:
        move = random.choice(list(board.legal_moves))
    return move


def generate_selfplay_games(cfg: SelfPlayConfig) -> Path:
    device = _select_device(cfg.device)
    model = _load_model(cfg, device)

    output_path = Path(cfg.output_pgn)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        for game_idx in range(cfg.num_games):
            board = chess.Board()
            game = chess.pgn.Game()
            node = game
            move_count = 0

            while not board.is_game_over(claim_draw=True) and move_count < cfg.max_moves:
                move = _choose_move(model, board, cfg, device)
                board.push(move)
                node = node.add_variation(move)
                move_count += 1

            game.headers["Result"] = board.result(claim_draw=True)
            handle.write(str(game) + "\n\n")

    return output_path


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Generate self-play games")
    parser.add_argument("--config", type=Path, default=Path("configs/selfplay.yaml"))
    args = parser.parse_args(argv)

    cfg = _load_config(args.config)
    output = generate_selfplay_games(cfg)
    print(f"Self-play games written to {output}")


if __name__ == "__main__":
    main()


__all__ = ["SelfPlayConfig", "generate_selfplay_games"]
