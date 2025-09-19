"""Iterative refinement model for chess policy/value prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
from torch import nn
from torch.nn import functional as F

from chessref.data.planes import NUM_PLANES


@dataclass
class RefinementStepOutput:
    policy_logits: torch.Tensor
    value: torch.Tensor
    halt_logits: Optional[torch.Tensor]


@dataclass
class RefinementOutputs:
    steps: List[RefinementStepOutput]

    def stack_policy_logits(self) -> torch.Tensor:
        return torch.stack([step.policy_logits for step in self.steps], dim=0)

    def stack_values(self) -> torch.Tensor:
        return torch.stack([step.value for step in self.steps], dim=0)

    def stack_halt_logits(self) -> Optional[torch.Tensor]:
        if any(step.halt_logits is None for step in self.steps):
            return None
        return torch.stack([step.halt_logits for step in self.steps], dim=0)


class IterativeRefiner(nn.Module):
    """Transformer backbone with iterative refinement outer loop."""

    def __init__(
        self,
        num_moves: int,
        *,
        d_model: int = 512,
        nhead: int = 8,
        depth: int = 8,
        dim_feedforward: Optional[int] = None,
        dropout: float = 0.1,
        use_act: bool = True,
    ) -> None:
        super().__init__()
        self.num_moves = num_moves
        self.use_act = use_act

        self.square_embed = nn.Linear(NUM_PLANES, d_model)
        self.policy_embed = nn.Linear(num_moves, d_model)
        self.position_embed = nn.Parameter(torch.randn(1, 65, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward or 4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.policy_head = nn.Linear(d_model, num_moves)
        self.value_head = nn.Linear(d_model, 1)
        self.halt_head = nn.Linear(d_model, 1) if use_act else None

    def forward_step(self, board_planes: torch.Tensor, prev_policy: torch.Tensor) -> RefinementStepOutput:
        batch_size = board_planes.size(0)

        planes = board_planes.reshape(batch_size, NUM_PLANES, 64).permute(0, 2, 1)
        token_embed = self.square_embed(planes)

        policy_token = self.policy_embed(prev_policy).unsqueeze(1)
        tokens = torch.cat([policy_token, token_embed], dim=1)

        pos = self.position_embed[:, : tokens.size(1), :]
        tokens = tokens + pos
        encoded = self.encoder(tokens)
        global_token = encoded[:, 0]

        policy_logits = self.policy_head(global_token)
        value = self.value_head(global_token).squeeze(-1)
        halt_logits = self.halt_head(global_token).squeeze(-1) if self.use_act else None

        return RefinementStepOutput(policy_logits=policy_logits, value=value, halt_logits=halt_logits)

    def forward(
        self,
        board_planes: torch.Tensor,
        *,
        num_steps: int,
        prev_policy: Optional[torch.Tensor] = None,
        detach_prev_policy: bool = True,
        legal_mask: Optional[torch.Tensor] = None,
    ) -> RefinementOutputs:
        if prev_policy is None:
            if legal_mask is not None:
                legal_cast = legal_mask.to(dtype=board_planes.dtype)
                mass = legal_cast.sum(dim=-1, keepdim=True).clamp_min(1.0)
                prev_policy = legal_cast / mass
            else:
                prev_policy = torch.full(
                    (board_planes.size(0), self.num_moves),
                    1.0 / self.num_moves,
                    dtype=board_planes.dtype,
                    device=board_planes.device,
                )
        outputs: List[RefinementStepOutput] = []

        current_policy = prev_policy
        for _ in range(num_steps):
            step = self.forward_step(board_planes, current_policy)
            outputs.append(step)
            next_policy = F.softmax(step.policy_logits, dim=-1)
            if legal_mask is not None:
                masked_policy = torch.where(legal_mask, next_policy, torch.zeros_like(next_policy))
                mass = masked_policy.sum(dim=-1, keepdim=True)
                fallback = current_policy
                next_policy = torch.where(mass > 0, masked_policy / mass, fallback)
            if detach_prev_policy:
                next_policy = next_policy.detach()
            current_policy = next_policy

        return RefinementOutputs(steps=outputs)


__all__ = ["IterativeRefiner", "RefinementOutputs", "RefinementStepOutput"]
