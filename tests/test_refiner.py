import torch

from chessref.model.refiner import IterativeRefiner
from chessref.model.loss import LossConfig, compute_refiner_loss
from chessref.moves.encoding import NUM_MOVES
from chessref.data.planes import NUM_PLANES


def test_refiner_forward_shapes() -> None:
    model = IterativeRefiner(num_moves=NUM_MOVES, d_model=64, nhead=8, depth=2, use_act=True)
    batch = 2
    steps = 3
    board_planes = torch.randn(batch, NUM_PLANES, 8, 8)

    outputs = model(board_planes, num_steps=steps)
    assert len(outputs.steps) == steps

    for step in outputs.steps:
        assert step.policy_logits.shape == (batch, NUM_MOVES)
        assert step.value.shape == (batch,)
        assert step.halt_logits is not None
        assert step.halt_logits.shape == (batch,)


def test_refiner_loss_backward() -> None:
    batch = 2
    steps = 2
    logits = [torch.randn(batch, NUM_MOVES, requires_grad=True) for _ in range(steps)]
    values = [torch.randn(batch, requires_grad=True) for _ in range(steps)]
    policy_targets = torch.zeros(batch, NUM_MOVES)
    policy_targets[:, 0] = 1.0
    value_targets = torch.zeros(batch)
    legal_mask = torch.zeros(batch, NUM_MOVES, dtype=torch.bool)
    legal_mask[:, 0] = True
    halt_logits = [torch.randn(batch, requires_grad=True) for _ in range(steps)]
    halt_targets = torch.ones(batch)

    loss = compute_refiner_loss(
        logits,
        values,
        policy_targets=policy_targets,
        value_targets=value_targets,
        legal_mask=legal_mask,
        halt_logits=halt_logits,
        halt_targets=halt_targets,
        config=LossConfig(act_weight=0.5),
    )
    loss.total.backward()

    assert loss.policy.requires_grad
    assert loss.value.requires_grad
    assert loss.act.requires_grad
