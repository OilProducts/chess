"""Self-play game generation with Monte Carlo Tree Search."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import chess
import chess.pgn
import torch
from omegaconf import OmegaConf

from chessref.data.pgn_extract import PositionRecord, TrainingSample
from chessref.data.planes import board_to_planes
from chessref.inference.mcts import MCTS, MCTSConfig
from chessref.model.refiner import IterativeRefiner
from chessref.moves.encoding import NUM_MOVES, encode_move
from chessref.train.train_supervised import ModelConfig, _select_device


@dataclass
class SelfPlayConfig:
    model: ModelConfig
    checkpoint: Optional[str] = None
    num_games: int = 1
    max_moves: int = 200
    output_pgn: str = "selfplay.pgn"
    output_dataset: Optional[str] = None
    device: str = "auto"
    mcts_num_simulations: int = 128
    mcts_c_puct: float = 1.5
    mcts_dirichlet_alpha: float = 0.3
    mcts_dirichlet_epsilon: float = 0.25
    inference_max_loops: int = 1
    temperature: float = 1.0


def _load_config(path: Path) -> SelfPlayConfig:
    cfg = OmegaConf.load(path)
    model = ModelConfig(**OmegaConf.to_object(cfg.model))
    kwargs = OmegaConf.to_object(cfg)
    kwargs.pop("model")
    return SelfPlayConfig(model=model, **kwargs)


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


def _select_move(visit_counts: torch.Tensor, temperature: float) -> int:
    visits = visit_counts.clone()
    if visits.sum() == 0:
        visits.fill_(1.0)
    if temperature <= 1e-6:
        return int(torch.argmax(visits).item())
    adjusted = torch.pow(visits, 1.0 / max(temperature, 1e-6))
    probs = adjusted / adjusted.sum()
    choice = torch.multinomial(probs, num_samples=1)
    return int(choice.item())


def _result_value(board: chess.Board, for_white: bool) -> float:
    result = board.result(claim_draw=True)
    if result == "1-0":
        return 1.0 if for_white else 0.0
    if result == "0-1":
        return 0.0 if for_white else 1.0
    return 0.5


def generate_selfplay_games(cfg: SelfPlayConfig) -> Path:
    device = _select_device(cfg.device)
    model = _load_model(cfg, device)
    mcts = MCTS(
        model,
        MCTSConfig(
            num_simulations=cfg.mcts_num_simulations,
            c_puct=cfg.mcts_c_puct,
            dirichlet_alpha=cfg.mcts_dirichlet_alpha,
            dirichlet_epsilon=cfg.mcts_dirichlet_epsilon,
            inference_max_loops=cfg.inference_max_loops,
        ),
        device=device,
    )

    output_path = Path(cfg.output_pgn)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset: List[TrainingSample] = []

    with output_path.open("w", encoding="utf-8") as handle:
        for game_index in range(cfg.num_games):
            board = chess.Board()
            game = chess.pgn.Game()
            node = game
            move_count = 0
            episode_records: List[dict] = []

            while not board.is_game_over(claim_draw=True) and move_count < cfg.max_moves:
                visit_counts_dict = mcts.search(board, add_noise=True)
                visit_tensor = torch.zeros(NUM_MOVES, dtype=torch.float32)
                for move, visits in visit_counts_dict.items():
                    idx = encode_move(move, board)
                    visit_tensor[idx] = float(visits)

                move_index = _select_move(visit_tensor, cfg.temperature)
                chosen_move: Optional[chess.Move] = None
                for move in board.legal_moves:
                    if encode_move(move, board) == move_index:
                        chosen_move = move
                        break
                if chosen_move is None:
                    chosen_move = max(visit_counts_dict.items(), key=lambda item: item[1])[0]

                episode_records.append(
                    {
                        "fen": board.fen(),
                        "policy": visit_tensor.clone(),
                        "move": chosen_move.uci(),
                        "game_index": game_index,
                        "ply": move_count,
                        "turn": board.turn,
                    }
                )

                board.push(chosen_move)
                node = node.add_variation(chosen_move)
                move_count += 1

            result = board.result(claim_draw=True)
            game.headers["Result"] = result
            handle.write(str(game) + "\n\n")

            for record in episode_records:
                value = _result_value(board, for_white=(record["turn"]))
                metadata = PositionRecord(
                    fen=record["fen"],
                    move_uci=record["move"],
                    white_result=_result_value(board, True),
                    ply=record["ply"],
                    game_index=record["game_index"],
                )
                planes = board_to_planes(chess.Board(record["fen"]))
                policy = record["policy"]
                total_visits = policy.sum()
                if total_visits > 0:
                    policy = policy / total_visits
                dataset.append(
                    TrainingSample(
                        planes=planes,
                        policy=policy,
                        value=torch.tensor(value, dtype=torch.float32),
                        metadata=metadata,
                    )
                )

    if cfg.output_dataset:
        ds_path = Path(cfg.output_dataset)
        ds_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dataset, ds_path)

    return output_path


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Generate self-play games with MCTS")
    parser.add_argument("--config", type=Path, default=Path("configs/selfplay.yaml"))
    args = parser.parse_args(argv)

    cfg = _load_config(args.config)
    output = generate_selfplay_games(cfg)
    print(f"Self-play games written to {output}")
    if cfg.output_dataset:
        print(f"Training samples saved to {cfg.output_dataset}")


if __name__ == "__main__":
    main()


__all__ = ["SelfPlayConfig", "generate_selfplay_games"]
