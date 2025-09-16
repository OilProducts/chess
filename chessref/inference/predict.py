"""Inference helpers for iterative refinement predictions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch

from chessref.model.refiner import IterativeRefiner


@dataclass
class InferenceStep:
    policy: torch.Tensor
    value: torch.Tensor
    halt: Optional[torch.Tensor]


@dataclass
class InferenceResult:
    steps: List[InferenceStep]

    @property
    def final_policy(self) -> torch.Tensor:
        return self.steps[-1].policy

    @property
    def final_value(self) -> torch.Tensor:
        return self.steps[-1].value

    @property
    def final_halt(self) -> Optional[torch.Tensor]:
        return self.steps[-1].halt


def _entropy(probs: torch.Tensor) -> torch.Tensor:
    eps = 1e-8
    return -(probs * (probs + eps).log()).sum(dim=-1)


@torch.no_grad()
def refine_predict(
    model: IterativeRefiner,
    board_planes: torch.Tensor,
    legal_mask: torch.Tensor,
    *,
    max_loops: int = 4,
    min_loops: int = 1,
    use_act: bool = False,
    act_threshold: float = 0.5,
    entropy_threshold: Optional[float] = None,
    prev_policy: Optional[torch.Tensor] = None,
) -> InferenceResult:
    """Run iterative refinement and return per-step predictions.

    Parameters
    ----------
    model:
        Trained :class:`IterativeRefiner` in evaluation mode.
    board_planes:
        Tensor of shape ``[batch, NUM_PLANES, 8, 8]``.
    legal_mask:
        Boolean mask of shape ``[batch, num_moves]`` indicating legal moves.
    max_loops / min_loops:
        Minimum and maximum refinement iterations.
    use_act:
        Whether to respect the model's ACT halting head.
    act_threshold:
        Halt probability above which a batch element stops refining.
    entropy_threshold:
        Optional entropy cutoff that stops when all batch elements drop below it.
    prev_policy:
        Optional initial policy distribution. If omitted, a uniform distribution
        over legal moves is used.
    """

    device = board_planes.device
    batch, num_moves = legal_mask.shape

    current_policy = prev_policy
    if current_policy is None:
        legal_cast = legal_mask.to(dtype=board_planes.dtype)
        uniform = legal_cast / legal_cast.sum(dim=-1, keepdim=True)
        current_policy = torch.where(legal_mask, uniform, torch.zeros_like(uniform))

    steps: List[InferenceStep] = []
    model = model.eval()

    active = torch.ones(batch, dtype=torch.bool, device=device)

    for step_idx in range(max_loops):
        step_output = model.forward_step(board_planes, current_policy)
        policy = torch.softmax(step_output.policy_logits, dim=-1)
        policy = torch.where(legal_mask, policy, torch.zeros_like(policy))
        policy_sum = policy.sum(dim=-1, keepdim=True)
        policy = torch.where(policy_sum > 0, policy / policy_sum, current_policy)

        steps.append(
            InferenceStep(
                policy=policy,
                value=step_output.value,
                halt=torch.sigmoid(step_output.halt_logits)
                if step_output.halt_logits is not None
                else None,
            )
        )

        current_policy = policy

        if step_idx + 1 < min_loops:
            continue

        stop_mask = torch.zeros_like(active)
        if use_act and step_output.halt_logits is not None:
            halt_probs = torch.sigmoid(step_output.halt_logits)
            stop_mask |= halt_probs >= act_threshold

        if entropy_threshold is not None:
            ent = _entropy(policy)
            stop_mask |= ent <= entropy_threshold

        active &= ~stop_mask
        if not active.any():
            break

    return InferenceResult(steps=steps)


__all__ = ["refine_predict", "InferenceResult", "InferenceStep"]
