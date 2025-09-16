"""Loss utilities for refinement training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class LossConfig:
    policy_weight: float = 1.0
    value_weight: float = 1.0
    act_weight: float = 0.1
    value_loss: str = "mse"  # or "smooth_l1"


@dataclass
class LossOutputs:
    total: torch.Tensor
    policy: torch.Tensor
    value: torch.Tensor
    act: torch.Tensor
    per_step_policy: List[torch.Tensor]


def _value_loss(pred: torch.Tensor, target: torch.Tensor, *, mode: str) -> torch.Tensor:
    if mode == "mse":
        return F.mse_loss(pred, target)
    if mode == "smooth_l1":
        return F.smooth_l1_loss(pred, target)
    raise ValueError(f"Unknown value loss mode '{mode}'")


def policy_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    legal_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if legal_mask is not None:
        logits = logits.masked_fill(~legal_mask, float("-inf"))
    log_probs = F.log_softmax(logits, dim=-1)
    loss = -(target * log_probs).sum(dim=-1)
    return loss.mean()


def compute_refiner_loss(
    policy_logits: List[torch.Tensor],
    values: List[torch.Tensor],
    *,
    policy_targets: torch.Tensor,
    value_targets: torch.Tensor,
    legal_mask: Optional[torch.Tensor] = None,
    halt_logits: Optional[List[torch.Tensor]] = None,
    halt_targets: Optional[torch.Tensor] = None,
    config: Optional[LossConfig] = None,
) -> LossOutputs:
    cfg = config or LossConfig()

    per_step_policy_losses: List[torch.Tensor] = []
    for logits in policy_logits:
        loss = policy_cross_entropy(logits, policy_targets, legal_mask=legal_mask)
        per_step_policy_losses.append(loss)

    policy_loss = torch.stack(per_step_policy_losses).mean()

    stacked_values = torch.stack(values)
    value_pred = stacked_values.mean(dim=0)
    value_loss = _value_loss(value_pred, value_targets, mode=cfg.value_loss)

    if cfg.act_weight > 0 and halt_logits is not None and halt_targets is not None:
        stacked_halt = torch.stack(halt_logits)
        if halt_targets.dim() == 1:
            halt_targets_expanded = halt_targets.unsqueeze(0).expand_as(stacked_halt)
        else:
            halt_targets_expanded = halt_targets
        act_loss = F.binary_cross_entropy_with_logits(stacked_halt, halt_targets_expanded)
    else:
        act_loss = value_targets.new_zeros(())

    total = cfg.policy_weight * policy_loss + cfg.value_weight * value_loss + cfg.act_weight * act_loss

    return LossOutputs(
        total=total,
        policy=policy_loss,
        value=value_loss,
        act=act_loss,
        per_step_policy=per_step_policy_losses,
    )


__all__ = ["LossConfig", "LossOutputs", "compute_refiner_loss", "policy_cross_entropy"]
