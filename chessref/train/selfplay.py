"""Self-play game generation with Monte Carlo Tree Search."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import statistics

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
from chessref.utils.checkpoint import load_checkpoint


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


@dataclass
class StockfishEval:
    chosen_move: chess.Move
    best_move: chess.Move
    best_cp: float
    model_cp: float
    cp_loss: float


def _evaluate_stockfish_move(
    engine: chess.engine.SimpleEngine,
    limit: chess.engine.Limit,
    board: chess.Board,
    move: chess.Move,
    multipv: int,
) -> StockfishEval:
    analysis = engine.analyse(board, limit, multipv=multipv)
    if not analysis:
        return StockfishEval(
            chosen_move=move,
            best_move=move,
            best_cp=0.0,
            model_cp=0.0,
            cp_loss=0.0,
        )

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

    model_cp = model_score if model_score is not None else 0.0
    cp_loss = max(0.0, best_score - model_cp)

    return StockfishEval(
        chosen_move=move,
        best_move=best_move,
        best_cp=best_score,
        model_cp=model_cp,
        cp_loss=cp_loss,
    )


def _summarize_stockfish_evals(
    evaluations: Sequence[StockfishEval],
    thresholds: Optional[Dict[str, float]],
) -> str:
    cp_losses = [evaluation.cp_loss for evaluation in evaluations]
    if not cp_losses:
        return "stockfish=none"

    mean_loss = statistics.mean(cp_losses)
    summary_parts = [f"mean_cp_loss={mean_loss:.1f}"]

    if thresholds:
        for label, threshold in sorted(thresholds.items(), key=lambda item: item[1]):
            count = sum(1 for loss in cp_losses if loss >= threshold)
            summary_parts.append(f"{label}={count}")

    return "stockfish=" + " ".join(summary_parts)


def _prune_pgn_history(
    history: List[Tuple[str, int]],
    limit: int,
) -> None:
    total_games = sum(count for _, count in history)
    while total_games > limit and len(history) > 1:
        stale_path, stale_count = history.pop(0)
        total_games -= stale_count
        try:
            Path(stale_path).unlink()
            print(f"[selfplay] Removed stale PGN {stale_path}")
        except FileNotFoundError:
            pass

@dataclass
class SelfPlayConfig:
    model: ModelConfig
    checkpoint: Optional[str] = None
    num_games: int = 1
    max_moves: int = 200
    output_pgn: str = "selfplay.pgn"
    output_dataset: Optional[str] = None
    output_manifest: Optional[str] = None
    device: str = "auto"
    mcts_num_simulations: int = 128
    mcts_c_puct: float = 1.5
    mcts_dirichlet_alpha: float = 0.3
    mcts_dirichlet_epsilon: float = 0.25
    mcts_eval_batch_size: int = 1
    inference_max_loops: int = 1
    temperature: float = 1.0
    opening_moves: int = 12
    opening_temperature: float = 0.6
    opening_dirichlet_epsilon: float = 0.25
    train_after: bool = False
    train_config: Optional[str] = None
    eval_engine_path: Optional[str] = None
    eval_depth: Optional[int] = 12
    eval_nodes: Optional[int] = None
    eval_movetime: Optional[int] = None
    eval_multipv: int = 2
    eval_print_moves: bool = False
    eval_summary_thresholds: Optional[Dict[str, float]] = None
    max_positions: Optional[int] = None
    train_every_batches: int = 1
    run_forever: bool = False
    max_batches: Optional[int] = None


def _load_config(path: Path) -> SelfPlayConfig:
    cfg = OmegaConf.load(path)
    model = ModelConfig(**OmegaConf.to_object(cfg.model))
    kwargs = OmegaConf.to_object(cfg)
    kwargs.pop("model")
    return SelfPlayConfig(model=model, **kwargs)


def _load_manifest(path: Path) -> List[dict[str, object]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        try:
            data = json.load(handle)
        except json.JSONDecodeError:
            return []
    if not isinstance(data, list):
        return []
    return data


def _save_manifest(path: Path, entries: List[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(entries, handle, indent=2)


def _remove_game_files(entry: dict[str, object]) -> None:
    dataset_path = entry.get("dataset")
    pgn_path = entry.get("pgn")
    if isinstance(dataset_path, str):
        try:
            Path(dataset_path).unlink()
            print(f"[selfplay] Removed stale dataset {dataset_path}")
        except FileNotFoundError:
            pass
    if isinstance(pgn_path, str):
        try:
            Path(pgn_path).unlink()
            print(f"[selfplay] Removed stale PGN {pgn_path}")
        except FileNotFoundError:
            pass


def _game_files_exist(entry: dict[str, object]) -> bool:
    dataset_path = entry.get("dataset")
    pgn_path = entry.get("pgn")
    dataset_ok = isinstance(dataset_path, str) and Path(dataset_path).exists()
    pgn_ok = isinstance(pgn_path, str) and Path(pgn_path).exists()
    return dataset_ok and pgn_ok


def _dataset_paths_from_manifest(entries: List[dict[str, object]]) -> List[str]:
    return [str(entry["dataset"]) for entry in entries if isinstance(entry.get("dataset"), str)]


def _prune_manifest(
    entries: List[dict[str, object]],
    *,
    max_positions: Optional[int],
) -> List[dict[str, object]]:
    def total_positions(items: List[dict[str, object]]) -> int:
        return sum(int(entry.get("num_samples", 0)) for entry in items)

    changed = False

    if max_positions is not None:
        while entries and total_positions(entries) > max_positions:
            removed = entries.pop(0)
            _remove_game_files(removed)
            changed = True

    if changed:
        entries[:] = [entry for entry in entries if _game_files_exist(entry)]

    return entries


def _build_model(model_cfg: ModelConfig, device: torch.device) -> IterativeRefiner:
    model = IterativeRefiner(
        num_moves=model_cfg.num_moves,
        d_model=model_cfg.d_model,
        nhead=model_cfg.nhead,
        depth=model_cfg.depth,
        dim_feedforward=model_cfg.dim_feedforward,
        dropout=model_cfg.dropout,
        use_act=model_cfg.use_act,
    ).to(device)
    return model


def _load_model(cfg: SelfPlayConfig, device: torch.device) -> IterativeRefiner:
    model = _build_model(cfg.model, device)
    if cfg.checkpoint:
        checkpoint = torch.load(cfg.checkpoint, map_location=device)
        state = checkpoint.get("model", checkpoint)
        model.load_state_dict(state)
    model.eval()
    return model


def _latest_iteration_checkpoint(directory: Path) -> Optional[Path]:
    if not directory.exists():
        return None
    iteration_paths = sorted(directory.glob("iteration_*.pt"))
    if iteration_paths:
        return iteration_paths[-1]
    generic_paths = sorted(directory.glob("*.pt"))
    if generic_paths:
        return generic_paths[-1]
    return None


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
    if cfg.train_after and cfg.output_dataset is None:
        raise ValueError("output_dataset must be set when train_after=True")

    _print_run_configuration(cfg)

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
            eval_batch_size=cfg.mcts_eval_batch_size,
        ),
        device=device,
    )

    pgn_base_path = Path(cfg.output_pgn)
    if pgn_base_path.suffix:
        pgn_dir = pgn_base_path.parent
        pgn_stem = pgn_base_path.stem
        pgn_suffix = pgn_base_path.suffix or ".pgn"
    else:
        pgn_dir = pgn_base_path
        pgn_stem = pgn_base_path.name
        pgn_suffix = ".pgn"
    pgn_dir.mkdir(parents=True, exist_ok=True)

    dataset_base_path = Path(cfg.output_dataset) if cfg.output_dataset else None
    if dataset_base_path is not None and dataset_base_path.suffix:
        dataset_dir = dataset_base_path.parent
        dataset_stem = dataset_base_path.stem
        dataset_suffix = dataset_base_path.suffix or ".pt"
    elif dataset_base_path is not None:
        dataset_dir = dataset_base_path
        dataset_stem = dataset_base_path.name
        dataset_suffix = ".pt"
    else:
        dataset_dir = None
        dataset_stem = None
        dataset_suffix = None
    if dataset_dir is not None:
        dataset_dir.mkdir(parents=True, exist_ok=True)

    if cfg.output_manifest:
        manifest_path = Path(cfg.output_manifest)
    elif dataset_dir is not None:
        manifest_path = dataset_dir / "manifest.json"
    else:
        manifest_path = pgn_dir / "manifest.json"

    manifest_entries = _load_manifest(manifest_path)
    manifest_entries = [entry for entry in manifest_entries if _game_files_exist(entry)]
    if cfg.max_positions is not None:
        manifest_entries = _prune_manifest(
            manifest_entries,
            max_positions=cfg.max_positions,
        )
    _save_manifest(manifest_path, manifest_entries)

    manifest_signatures = {
        entry.get("line_hash")
        for entry in manifest_entries
        if isinstance(entry.get("line_hash"), str)
    }
    dataset_paths: List[str] = _dataset_paths_from_manifest(manifest_entries)

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

    batch_index = 0
    last_pgn: Optional[Path] = None

    try:
        while True:
            games_in_batch = 0
            for game_idx in range(cfg.num_games):
                global_game = batch_index * cfg.num_games + game_idx + 1
                print(f"[selfplay] Generating game {global_game} (batch {batch_index + 1})...")
                board = chess.Board()
                game = chess.pgn.Game()
                node = game
                move_count = 0
                episode_records: List[dict] = []
                game_evals: List[StockfishEval] = []
                game_samples: List[TrainingSample] = []
                mcts.reset_timing_stats()

                while not board.is_game_over(claim_draw=True) and move_count < cfg.max_moves:
                    is_opening = move_count < cfg.opening_moves
                    search_temperature = cfg.opening_temperature if is_opening else cfg.temperature
                    search_epsilon = cfg.opening_dirichlet_epsilon if is_opening else cfg.mcts_dirichlet_epsilon

                    original_epsilon = mcts.cfg.dirichlet_epsilon
                    mcts.cfg.dirichlet_epsilon = search_epsilon
                    visit_counts_dict = mcts.search(board, add_noise=True)
                    mcts.cfg.dirichlet_epsilon = original_epsilon
                    visit_tensor = torch.zeros(NUM_MOVES, dtype=torch.float32)
                    for move, visits in visit_counts_dict.items():
                        idx = encode_move(move, board)
                        visit_tensor[idx] = float(visits)

                    move_index = _select_move(visit_tensor, search_temperature)
                    chosen_move: Optional[chess.Move] = None
                    for move in board.legal_moves:
                        if encode_move(move, board) == move_index:
                            chosen_move = move
                            break
                if chosen_move is None:
                    print(
                        "[selfplay] Warning: failed to map move_index to legal move; choosing "
                        "highest-visit fallback"
                    )
                    chosen_move = max(visit_counts_dict.items(), key=lambda item: item[1])[0]

                    if engine is not None and limit is not None:
                        eval_result = _evaluate_stockfish_move(
                            engine,
                            limit,
                            board.copy(stack=False),
                            chosen_move,
                            multipv,
                        )
                        game_evals.append(eval_result)
                        if cfg.eval_print_moves:
                            print(
                                f"[selfplay] batch={batch_index + 1} game={global_game} ply={move_count + 1} "
                                f"model={chosen_move.uci()} best={eval_result.best_move.uci()} "
                                f"model_cp={eval_result.model_cp:.1f} best_cp={eval_result.best_cp:.1f} "
                                f"cp_loss={eval_result.cp_loss:.1f}"
                            )

                    episode_records.append(
                        {
                            "fen": board.fen(),
                            "policy": visit_tensor.clone(),
                            "move": chosen_move.uci(),
                            "game_index": global_game,
                            "ply": move_count,
                            "turn": board.turn,
                        }
                    )

                    board.push(chosen_move)
                    node = node.add_variation(chosen_move)
                    move_count += 1

                result = board.result(claim_draw=True)
                game.headers["Result"] = result
                timing_summary = mcts.pop_timing_summary()
                timing_suffix = ""
                if timing_summary is not None:
                    timing_suffix = (
                        f" mcts_batches={timing_summary.batches}"
                        f" mcts_positions={timing_summary.positions}"
                        f" mcts_avg_batch_ms={timing_summary.avg_batch_ms:.2f}"
                        f" mcts_avg_pos_ms={timing_summary.avg_position_ms:.3f}"
                    )

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
                    game_samples.append(
                        TrainingSample(
                            planes=planes,
                            policy=policy,
                            value=torch.tensor(value, dtype=torch.float32),
                            metadata=metadata,
                        )
                    )

                line_signature = " ".join(record["fen"] for record in episode_records)
                line_hash = hashlib.sha1(line_signature.encode("utf-8")).hexdigest()
                if line_hash in manifest_signatures:
                    duplicates = [entry for entry in manifest_entries if entry.get("line_hash") == line_hash]
                    for entry in duplicates:
                        _remove_game_files(entry)
                    manifest_entries = [entry for entry in manifest_entries if entry.get("line_hash") != line_hash]
                    manifest_signatures.discard(line_hash)
                    dataset_paths = _dataset_paths_from_manifest(manifest_entries)
                    manifest_signatures = {
                        entry.get("line_hash")
                        for entry in manifest_entries
                        if isinstance(entry.get("line_hash"), str)
                    }
                    _save_manifest(manifest_path, manifest_entries)

                timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
                unique_id = f"{timestamp}_g{global_game:05d}"
                pgn_file = pgn_dir / f"{pgn_stem}_{unique_id}{pgn_suffix}"
                with pgn_file.open("w", encoding="utf-8") as handle:
                    handle.write(str(game) + "\n\n")

                dataset_file: Optional[Path] = None
                if dataset_dir is not None and dataset_stem is not None and dataset_suffix is not None:
                    dataset_file = dataset_dir / f"{dataset_stem}_{unique_id}{dataset_suffix}"
                    torch.save(game_samples, dataset_file)
                    print(f"[selfplay] Saved {len(game_samples)} training samples to {dataset_file}")

                if game_evals:
                    summary = _summarize_stockfish_evals(game_evals, cfg.eval_summary_thresholds)
                    print(
                        f"[selfplay] Game {global_game} finished result={result} plies={move_count} "
                        f"evaluated={len(game_evals)} {summary}{timing_suffix}"
                    )
                else:
                    print(
                        f"[selfplay] Game {global_game} finished result={result} plies={move_count}"
                        f"{timing_suffix}"
                    )

                games_in_batch += 1
                last_pgn = pgn_file

                if dataset_file is not None:
                    entry = {
                        "dataset": str(dataset_file),
                        "pgn": str(pgn_file),
                        "num_samples": len(game_samples),
                        "num_plies": move_count,
                        "line_hash": line_hash,
                        "result": result,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                    manifest_entries.append(entry)
                    manifest_signatures.add(line_hash)
                    manifest_entries = _prune_manifest(
                        manifest_entries,
                        max_positions=cfg.max_positions,
                    )
                    manifest_signatures = {
                        entry.get("line_hash")
                        for entry in manifest_entries
                        if isinstance(entry.get("line_hash"), str)
                    }
                    _save_manifest(manifest_path, manifest_entries)
                    dataset_paths = _dataset_paths_from_manifest(manifest_entries)

            if cfg.train_after and games_in_batch > 0 and dataset_paths:
                train_cfg_path = Path(cfg.train_config) if cfg.train_config else Path("configs/train.yaml")
                train_cfg = load_train_config(train_cfg_path)
                manifest_datasets = list(dict.fromkeys(dataset_paths))
                train_cfg.data.selfplay_datasets = manifest_datasets
                print(
                    f"[selfplay] Starting supervised training on {len(train_cfg.data.selfplay_datasets)} datasets (batch {batch_index + 1})"
                )
                train(train_cfg)

                checkpoint_dir = Path(train_cfg.checkpoint.directory)
                latest_checkpoint = _latest_iteration_checkpoint(checkpoint_dir)
                if latest_checkpoint is not None:
                    print(f"[selfplay] Reloading model from {latest_checkpoint}")
                    reloaded_model = _build_model(train_cfg.model, device)
                    checkpoint = load_checkpoint(latest_checkpoint, map_location=device)
                    reloaded_model.load_state_dict(checkpoint["model"])
                    reloaded_model.eval()
                    model = reloaded_model
                    cfg.model = train_cfg.model
                    mcts.model = model
                else:
                    print(
                        f"[selfplay] Warning: no checkpoint found in {checkpoint_dir} after training;"
                        " continuing with previous weights."
                    )

            batch_index += 1

            if not cfg.run_forever:
                break
            if cfg.max_batches is not None and batch_index >= cfg.max_batches:
                break

        return last_pgn if last_pgn is not None else pgn_dir
    finally:
        if engine is not None:
            engine.close()

def _print_run_configuration(cfg: SelfPlayConfig) -> None:
    print("[selfplay] Configuration summary:")
    print(
        f"  model: d_model={cfg.model.d_model} depth={cfg.model.depth} nhead={cfg.model.nhead} "
        f"dim_feedforward={cfg.model.dim_feedforward} use_act={cfg.model.use_act}"
    )
    print(
        f"  mcts: sims={cfg.mcts_num_simulations} c_puct={cfg.mcts_c_puct} "
        f"dirichlet_alpha={cfg.mcts_dirichlet_alpha} dirichlet_epsilon={cfg.mcts_dirichlet_epsilon} "
        f"eval_batch={cfg.mcts_eval_batch_size}"
    )
    print(
        f"  refinement: inference_max_loops={cfg.inference_max_loops} temperature={cfg.temperature}"
    )
    if cfg.eval_engine_path:
        print(
            f"  stockfish: path={cfg.eval_engine_path} depth={cfg.eval_depth} nodes={cfg.eval_nodes} "
            f"movetime={cfg.eval_movetime} multipv={cfg.eval_multipv}"
        )
    else:
        print("  stockfish: disabled")
    print(
        f"  storage: dataset_base={cfg.output_dataset} manifest={cfg.output_manifest} "
        f"max_positions={cfg.max_positions}"
    )
    print(
        f"  pgn: base={cfg.output_pgn}"
    )
    if cfg.run_forever:
        print(
            f"  loop: run_forever=true num_games={cfg.num_games} max_batches={cfg.max_batches}"
        )
    else:
        print(
            f"  loop: run_forever=false num_games={cfg.num_games} max_batches={cfg.max_batches}"
        )
    print(
        f"  training: train_after={cfg.train_after} train_every_batches={cfg.train_every_batches}"
    )
    print(f"  checkpoint: checkpoint={cfg.checkpoint}")


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Generate self-play games with MCTS")
    parser.add_argument("--config", type=Path, default=Path("configs/selfplay.yaml"))
    args = parser.parse_args(argv)

    cfg = _load_config(args.config)
    output = generate_selfplay_games(cfg)
    print(f"Self-play games written to {output}")
    if cfg.output_manifest:
        print(f"Manifest updated at {cfg.output_manifest}")
    elif cfg.output_dataset:
        print(f"Training samples saved under {cfg.output_dataset}")


if __name__ == "__main__":
    main()


__all__ = ["SelfPlayConfig", "generate_selfplay_games"]
