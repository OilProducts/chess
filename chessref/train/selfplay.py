"""Self-play game generation with Monte Carlo Tree Search."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import chess
import chess.pgn
import chess.engine
import torch
from omegaconf import OmegaConf

from chessref.data.pgn_extract import PositionRecord, TrainingSample
from chessref.data.planes import board_to_planes
from chessref.inference.mcts import MCTS, MCTSConfig
from chessref.moves.encoding import NUM_MOVES, encode_move
from chessref.model.refiner import IterativeRefiner
from chessref.train.train_supervised import (
    ModelConfig,
    TrainConfig,
    load_train_config,
    train,
    _select_device,
)


def _score_to_cp(score: chess.engine.Score, mate_value: int = 10000) -> float:
    """Return a numeric centipawn approximation for a :class:`chess.engine.Score`."""

    # Prefer python-chess helper methods when available.
    if hasattr(score, "is_mate") and score.is_mate():  # type: ignore[attr-defined]
        mate = score.mate()  # type: ignore[attr-defined]
        if mate is None:
            return 0.0
        return mate_value if mate > 0 else -mate_value

    if hasattr(score, "is_cp") and score.is_cp():  # type: ignore[attr-defined]
        cp = score.cp()  # type: ignore[attr-defined]
        if cp is not None:
            return float(cp)

    # Fallback for PovScore objects (.cp / .mate attributes)
    cp = getattr(score, "cp", None)
    if cp is not None:
        return float(cp)

    mate = getattr(score, "mate", None)
    if mate is not None:
        return mate_value if mate > 0 else -mate_value

    return 0.0


def _evaluate_stockfish_move(
    engine: chess.engine.SimpleEngine,
    limit: chess.engine.Limit,
    board: chess.Board,
    move: chess.Move,
    multipv: int,
) -> tuple[float, chess.Move]:
    analysis = engine.analyse(board, limit, multipv=multipv)
    if not analysis:
        return 0.0, move

    best_entry = analysis[0]
    pv = best_entry.get("pv", [])
    best_move = pv[0] if pv else move
    best_score = _score_to_cp(best_entry["score"].pov(board.turn))

    model_score: Optional[float] = None
    for entry in analysis:
        pv = entry.get("pv")
        if pv and pv[0] == move:
            model_score = _score_to_cp(entry["score"].pov(board.turn))
            break

    if model_score is None:
        board.push(move)
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
    return cp_loss, best_move


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
    train_after: bool = False
    train_config: Optional[str] = None
    eval_engine_path: Optional[str] = None
    eval_depth: Optional[int] = 12
    eval_nodes: Optional[int] = None
    eval_movetime: Optional[int] = None
    eval_log_interval: int = 1
    eval_multipv: int = 2
    max_datasets: Optional[int] = None
    train_every_batches: int = 1
    run_forever: bool = False
    max_batches: Optional[int] = None


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


def _format_batch_path(base: Optional[Path], batch_index: int, run_forever: bool) -> Optional[Path]:
    if base is None:
        return None
    if batch_index == 0 and not run_forever:
        return base
    stem = base.stem
    suffix = base.suffix
    return base.with_name(f"{stem}_b{batch_index}{suffix}")


def generate_selfplay_games(cfg: SelfPlayConfig) -> Path:
    if cfg.train_after and cfg.output_dataset is None:
        raise ValueError("output_dataset must be set when train_after=True")

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

    output_base = Path(cfg.output_pgn)
    output_base.parent.mkdir(parents=True, exist_ok=True)

    dataset_base = Path(cfg.output_dataset) if cfg.output_dataset else None
    if dataset_base is not None:
        dataset_base.parent.mkdir(parents=True, exist_ok=True)

    engine: Optional[chess.engine.SimpleEngine] = None
    limit: Optional[chess.engine.Limit] = None
    multipv = max(1, cfg.eval_multipv)
    if cfg.eval_engine_path:
        engine = chess.engine.SimpleEngine.popen_uci(cfg.eval_engine_path)
        limit_kwargs = {}
        if cfg.eval_depth is not None:
            limit_kwargs["depth"] = cfg.eval_depth
        if cfg.eval_nodes is not None:
            limit_kwargs["nodes"] = cfg.eval_nodes
        if cfg.eval_movetime is not None:
            limit_kwargs["movetime"] = cfg.eval_movetime
        limit = chess.engine.Limit(**limit_kwargs)

    dataset_paths: List[str] = []
    batch_index = 0
    last_pgn = output_base

    try:
        while True:
            pgn_path = _format_batch_path(output_base, batch_index, cfg.run_forever)
            dataset_path = (
                _format_batch_path(dataset_base, batch_index, cfg.run_forever)
                if dataset_base is not None
                else None
            )

            dataset: List[TrainingSample] = []
            with pgn_path.open("w", encoding="utf-8") as handle:
                for game_idx in range(cfg.num_games):
                    global_game = batch_index * cfg.num_games + game_idx + 1
                    print(f"[selfplay] Generating game {global_game} (batch {batch_index + 1})...")
                    board = chess.Board()
                    game = chess.pgn.Game()
                    node = game
                    move_count = 0
                    episode_records: List[dict] = []
                    cp_losses: List[float] = []

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

                        if engine is not None and limit is not None:
                            cp_loss, best_move = _evaluate_stockfish_move(
                                engine,
                                limit,
                                board.copy(stack=False),
                                chosen_move,
                                multipv,
                            )
                            cp_losses.append(cp_loss)
                            if cfg.eval_log_interval > 0 and (move_count + 1) % cfg.eval_log_interval == 0:
                                print(
                                    f"[selfplay] batch={batch_index + 1} game={global_game} ply={move_count + 1} "
                                    f"model={chosen_move.uci()} best={best_move.uci()} cp_loss={cp_loss:.1f}"
                                )

                        episode_records.append(
                            {
                                "fen": board.fen(),
                                "policy": visit_tensor.clone(),
                                "move": chosen_move.uci(),
                                "game_index": game_idx,
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
                    if cp_losses:
                        mean_loss = sum(cp_losses) / len(cp_losses)
                        print(
                            f"[selfplay] Game {global_game} finished result={result} mean_cp_loss={mean_loss:.1f} moves={len(cp_losses)}"
                        )
                    else:
                        print(f"[selfplay] Game {global_game} finished result={result} moves={move_count}")

                    for record in episode_records:
                        value = _result_value(board, for_white=(record["turn"]))
                        metadata = PositionRecord(
                            fen=record["fen"],
                            move_uci=record["move"],
                            white_result=_result_value(board, True),
                            ply=record["ply"],
                            game_index=batch_index,
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

            if dataset_path is not None:
                torch.save(dataset, dataset_path)
                dataset_paths.append(str(dataset_path))
                print(f"[selfplay] Saved {len(dataset)} training samples to {dataset_path}")
                if cfg.max_datasets is not None and len(dataset_paths) > cfg.max_datasets:
                    stale = dataset_paths.pop(0)
                    try:
                        Path(stale).unlink()
                        print(f"[selfplay] Removed stale dataset {stale}")
                    except FileNotFoundError:
                        pass

            if cfg.train_after and ((batch_index + 1) % max(1, cfg.train_every_batches) == 0):
                train_cfg_path = Path(cfg.train_config) if cfg.train_config else Path("configs/train.yaml")
                train_cfg = load_train_config(train_cfg_path)
                combined = list(train_cfg.data.selfplay_datasets)
                for path in dataset_paths:
                    if path not in combined:
                        combined.append(path)
                if cfg.max_datasets is not None:
                    combined = combined[-cfg.max_datasets:]
                train_cfg.data.selfplay_datasets = combined
                print(
                    f"[selfplay] Starting supervised training on {len(combined)} datasets (batch {batch_index + 1})"
                )
                train(train_cfg)

            batch_index += 1
            last_pgn = pgn_path

            if not cfg.run_forever:
                break
            if cfg.max_batches is not None and batch_index >= cfg.max_batches:
                break

        return last_pgn
    finally:
        if engine is not None:
            engine.close()


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
