"""Supervised training loop for the iterative refiner."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import ConcatDataset, DataLoader

from omegaconf import OmegaConf

from chessref.data.dataset import (
    ChessSampleDataset,
    PrecomputedSampleDataset,
    build_target_generator,
    collate_samples,
    load_records,
)
from chessref.data.pgn_extract import TrainingSample
from chessref.model.loss import LossConfig, compute_refiner_loss
from chessref.model.refiner import IterativeRefiner
from chessref.moves.legal_mask import legal_move_mask
from chessref.utils.checkpoint import save_checkpoint
from chessref.utils.logging import LoggingConfig, TrainLogger


@dataclass
class ModelConfig:
    num_moves: int
    d_model: int = 512
    nhead: int = 8
    depth: int = 8
    dim_feedforward: int = 2048
    dropout: float = 0.1
    use_act: bool = True


@dataclass
class DataConfig:
    pgn_paths: List[str] = field(default_factory=list)
    max_games: Optional[int] = None
    transforms: List[str] = field(default_factory=lambda: ["identity"])
    target: dict[str, Any] = field(default_factory=lambda: {"type": "selfplay"})
    selfplay_datasets: List[str] = field(default_factory=list)
    shuffle: bool = True
    num_workers: int = 0
    pin_memory: bool = False


@dataclass
class OptimConfig:
    lr: float = 3e-4
    weight_decay: float = 0.01
    betas: Sequence[float] = (0.9, 0.999)
    grad_accum_steps: int = 1
    use_amp: bool = False


@dataclass
class TrainingConfig:
    epochs: int = 1
    batch_size: int = 32
    k_train: int = 8
    detach_prev_policy: bool = True
    act_weight: float = 0.0
    value_loss: str = "mse"
    device: str = "auto"
    log_interval: int = 50


@dataclass
class CheckpointConfig:
    directory: str = "checkpoints"
    save_interval: int = 0  # steps; 0 disables


@dataclass
class TrainConfig:
    model: ModelConfig
    data: DataConfig
    optim: OptimConfig
    training: TrainingConfig
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


@dataclass
class TrainSummary:
    epochs_completed: int
    steps: int
    last_loss: float


def _select_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():  # type: ignore[attr-defined]
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def load_train_config(path: Path) -> TrainConfig:
    cfg = OmegaConf.load(path)
    model = ModelConfig(**OmegaConf.to_object(cfg.model))
    data = DataConfig(**OmegaConf.to_object(cfg.data))
    optim = OptimConfig(**OmegaConf.to_object(cfg.optim))
    training = TrainingConfig(**OmegaConf.to_object(cfg.training))
    checkpoint = CheckpointConfig(**OmegaConf.to_object(cfg.get("checkpoint", {})))
    logging = LoggingConfig(**OmegaConf.to_object(cfg.get("logging", {})))
    return TrainConfig(
        model=model,
        data=data,
        optim=optim,
        training=training,
        checkpoint=checkpoint,
        logging=logging,
    )


def _prepare_dataloader(cfg: TrainConfig) -> DataLoader:
    datasets = []

    if cfg.data.pgn_paths:
        paths = [Path(p) for p in cfg.data.pgn_paths]
        records = load_records(paths, max_games=cfg.data.max_games)
        if records:
            target_generator = build_target_generator(cfg.data.target)
            datasets.append(
                ChessSampleDataset(records, target_generator, transforms=cfg.data.transforms)
            )

    if cfg.data.selfplay_datasets:
        samples: List[TrainingSample] = []
        for dataset_path in cfg.data.selfplay_datasets:
            loaded = torch.load(dataset_path, map_location="cpu", weights_only=False)
            samples.extend(loaded)
        if samples:
            datasets.append(PrecomputedSampleDataset(samples))

    if not datasets:
        raise ValueError("No training data provided (PGNs or selfplay datasets).")

    if len(datasets) == 1:
        dataset = datasets[0]
    else:
        dataset = ConcatDataset(datasets)

    loader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=cfg.data.shuffle,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        collate_fn=collate_samples,
        drop_last=False,
    )
    return loader


def _build_model(cfg: TrainConfig, device: torch.device) -> IterativeRefiner:
    model = IterativeRefiner(
        num_moves=cfg.model.num_moves,
        d_model=cfg.model.d_model,
        nhead=cfg.model.nhead,
        depth=cfg.model.depth,
        dim_feedforward=cfg.model.dim_feedforward,
        dropout=cfg.model.dropout,
        use_act=cfg.model.use_act,
    )
    return model.to(device)


def _prepare_optimizer(model: IterativeRefiner, cfg: TrainConfig) -> torch.optim.Optimizer:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
        betas=tuple(cfg.optim.betas),
    )
    return optimizer


def train(cfg: TrainConfig) -> TrainSummary:
    device = _select_device(cfg.training.device)
    dataloader = _prepare_dataloader(cfg)
    model = _build_model(cfg, device)
    optimizer = _prepare_optimizer(model, cfg)
    use_amp = cfg.optim.use_amp and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)
    logger = TrainLogger(cfg.logging)

    loss_cfg = LossConfig(
        policy_weight=1.0,
        value_weight=1.0,
        act_weight=cfg.training.act_weight,
        value_loss=cfg.training.value_loss,
    )

    global_step = 0
    accum_counter = 0
    last_loss_value = float("nan")
    model.train()
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(cfg.training.epochs):
        for batch_idx, batch in enumerate(dataloader, start=1):
            planes = batch.planes.to(device)
            policy_targets = batch.policy.to(device)
            value_targets = batch.value.to(device)
            legal_mask = torch.stack(
                [legal_move_mask(record.board()).to(device) for record in batch.metadata]
            )
            if not torch.all(legal_mask.any(dim=-1)):
                raise RuntimeError("Encountered position without legal moves")

            target_mass_on_legal = []
            for idx in range(policy_targets.size(0)):
                mask = legal_mask[idx]
                target_mass_on_legal.append(policy_targets[idx][mask].sum())
            legal_mass = torch.stack(target_mass_on_legal)
            if not torch.allclose(legal_mass, legal_mass.new_ones(legal_mass.shape), atol=1e-4):
                raise RuntimeError("Policy targets place probability outside legal moves")

            with autocast(enabled=use_amp):
                outputs = model(
                    planes,
                    num_steps=cfg.training.k_train,
                    detach_prev_policy=cfg.training.detach_prev_policy,
                )
                policy_logits = [step.policy_logits for step in outputs.steps]
                for tensor in policy_logits:
                    if torch.isnan(tensor).any():
                        raise RuntimeError("Model produced NaN policy logits")
                values = [step.value for step in outputs.steps]
                halt_logits = (
                    [step.halt_logits for step in outputs.steps]
                    if cfg.model.use_act and cfg.training.act_weight > 0
                    else None
                )
                halt_targets = (
                    torch.zeros(policy_targets.size(0), device=device)
                    if halt_logits is not None
                    else None
                )

                loss_outputs = compute_refiner_loss(
                    policy_logits,
                    values,
                    policy_targets=policy_targets,
                    value_targets=value_targets,
                    legal_mask=legal_mask,
                    halt_logits=halt_logits,
                    halt_targets=halt_targets,
                    config=loss_cfg,
                )
                if torch.isnan(loss_outputs.total):
                    raise RuntimeError(
                        f"Loss produced NaN (policy={loss_outputs.policy}, value={loss_outputs.value})"
                    )
                loss = loss_outputs.total / cfg.optim.grad_accum_steps

            scaler.scale(loss).backward()
            accum_counter += 1
            last_loss_value = float(loss_outputs.total.detach().cpu().item())

            if accum_counter == cfg.optim.grad_accum_steps:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                accum_counter = 0

            global_step += 1

            if cfg.training.log_interval and global_step % cfg.training.log_interval == 0:
                metrics = {
                    "train/loss": loss_outputs.total.item(),
                    "train/policy_loss": loss_outputs.policy.item(),
                    "train/value_loss": loss_outputs.value.item(),
                }
                print(
                    f"epoch={epoch+1} step={global_step} loss={metrics['train/loss']:.4f} "
                    f"policy={metrics['train/policy_loss']:.4f} value={metrics['train/value_loss']:.4f}"
                )
                if logger.enabled:
                    logger.log_scalars(metrics, global_step)
                    logger.flush()

            if (
                cfg.checkpoint.save_interval
                and global_step % cfg.checkpoint.save_interval == 0
            ):
                save_checkpoint(
                    Path(cfg.checkpoint.directory) / f"step_{global_step}.pt",
                    model_state=model.state_dict(),
                    optimizer_state=optimizer.state_dict(),
                    step=global_step,
                )

        # Flush remaining gradients at epoch end.
        if accum_counter > 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            accum_counter = 0

    logger.close()

    return TrainSummary(
        epochs_completed=cfg.training.epochs,
        steps=global_step,
        last_loss=last_loss_value,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Supervised training for the chess refiner")
    parser.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    args = parser.parse_args(argv)

    cfg = load_train_config(args.config)
    summary = train(cfg)
    print(
        f"Training complete: epochs={summary.epochs_completed} steps={summary.steps} "
        f"last_loss={summary.last_loss:.4f}"
    )


if __name__ == "__main__":
    main()


__all__ = [
    "ModelConfig",
    "DataConfig",
    "OptimConfig",
    "TrainingConfig",
    "CheckpointConfig",
    "LoggingConfig",
    "TrainConfig",
    "TrainSummary",
    "train",
    "load_train_config",
]
