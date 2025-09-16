"""Policy evaluation utilities."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import torch
from torch.utils.data import DataLoader

from omegaconf import OmegaConf

from chessref.data.dataset import ChessSampleDataset, collate_samples, load_records
from chessref.data.dataset import build_target_generator
from chessref.inference.predict import refine_predict
from chessref.model.refiner import IterativeRefiner
from chessref.train.train_supervised import ModelConfig, DataConfig, _select_device
from chessref.moves.legal_mask import legal_move_mask


@dataclass
class InferenceConfig:
    max_loops: int = 4
    min_loops: int = 1
    use_act: bool = False
    entropy_threshold: Optional[float] = None
    act_threshold: float = 0.5


@dataclass
class RuntimeConfig:
    batch_size: int = 32
    device: str = "auto"


@dataclass
class EvalConfig:
    model: ModelConfig
    data: DataConfig
    inference: InferenceConfig
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    checkpoint: Optional[str] = None


@dataclass
class PolicyEvalResult:
    cross_entropy: float
    top1: float


def _load_eval_config(path: Path) -> EvalConfig:
    cfg = OmegaConf.load(path)
    model = ModelConfig(**OmegaConf.to_object(cfg.model))
    data = DataConfig(**OmegaConf.to_object(cfg.data))
    inference = InferenceConfig(**OmegaConf.to_object(cfg.inference))
    runtime = RuntimeConfig(**OmegaConf.to_object(cfg.get("runtime", {})))
    checkpoint = cfg.get("checkpoint")
    return EvalConfig(model=model, data=data, inference=inference, runtime=runtime, checkpoint=checkpoint)


def _build_model(cfg: EvalConfig, device: torch.device) -> IterativeRefiner:
    model = IterativeRefiner(
        num_moves=cfg.model.num_moves,
        d_model=cfg.model.d_model,
        nhead=cfg.model.nhead,
        depth=cfg.model.depth,
        dim_feedforward=cfg.model.dim_feedforward,
        dropout=cfg.model.dropout,
        use_act=cfg.model.use_act,
    )
    model.to(device)
    if cfg.checkpoint:
        state = torch.load(cfg.checkpoint, map_location=device)
        if "model" in state:
            state = state["model"]
        model.load_state_dict(state)
    return model


@torch.no_grad()
def evaluate_policy(cfg: EvalConfig) -> PolicyEvalResult:
    device = _select_device(cfg.runtime.device)
    if not cfg.data.pgn_paths:
        raise ValueError("data.pgn_paths must contain evaluation PGN files")

    records = load_records([Path(p) for p in cfg.data.pgn_paths], max_games=cfg.data.max_games)
    target_generator = build_target_generator(cfg.data.target)
    dataset = ChessSampleDataset(records, target_generator, transforms=cfg.data.transforms)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.runtime.batch_size,
        shuffle=False,
        collate_fn=collate_samples,
    )

    model = _build_model(cfg, device).eval()

    total_ce = 0.0
    total_top1 = 0.0
    total_samples = 0

    inference_cfg = cfg.inference

    for batch in dataloader:
        planes = batch.planes.to(device)
        policy_targets = batch.policy.to(device)
        legal_masks = torch.stack([legal_move_mask(record.board()) for record in batch.metadata]).to(device)

        result = refine_predict(
            model,
            planes,
            legal_masks,
            max_loops=inference_cfg.max_loops,
            min_loops=inference_cfg.min_loops,
            use_act=inference_cfg.use_act,
            act_threshold=inference_cfg.act_threshold,
            entropy_threshold=inference_cfg.entropy_threshold,
        )
        policy = result.final_policy

        log_probs = (policy + 1e-12).log()
        ce = -(policy_targets * log_probs).sum(dim=-1)
        preds = policy.argmax(dim=-1)
        labels = policy_targets.argmax(dim=-1)

        total_ce += ce.sum().item()
        total_top1 += (preds == labels).float().sum().item()
        total_samples += policy.size(0)

    return PolicyEvalResult(
        cross_entropy=total_ce / total_samples,
        top1=total_top1 / total_samples,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate policy accuracy")
    parser.add_argument("--config", type=Path, default=Path("configs/eval.yaml"))
    args = parser.parse_args(argv)

    cfg = _load_eval_config(args.config)
    result = evaluate_policy(cfg)
    print(f"cross_entropy={result.cross_entropy:.4f} top1={result.top1:.4f}")


if __name__ == "__main__":
    main()


__all__ = ["EvalConfig", "InferenceConfig", "PolicyEvalResult", "evaluate_policy"]
