from pathlib import Path

import torch
from torch import nn

from chessref.utils.checkpoint import load_checkpoint, save_checkpoint


def test_save_and_load_checkpoint(tmp_path: Path) -> None:
    model = nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    for _ in range(3):
        optimizer.zero_grad()
        out = model(torch.randn(1, 4))
        out.sum().backward()
        optimizer.step()

    path = save_checkpoint(
        tmp_path / "ckpt.pt",
        model_state=model.state_dict(),
        optimizer_state=optimizer.state_dict(),
        step=42,
        extra={"foo": "bar"},
    )

    assert path.exists()

    # Load into new instances
    new_model = nn.Linear(4, 2)
    new_optimizer = torch.optim.SGD(new_model.parameters(), lr=0.1)
    checkpoint = load_checkpoint(path, model=new_model, optimizer=new_optimizer)

    for p_old, p_new in zip(model.parameters(), new_model.parameters()):
        assert torch.allclose(p_old, p_new)
    assert checkpoint["step"] == 42
    assert checkpoint["foo"] == "bar"
