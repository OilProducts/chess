import torch

from chessref.utils.metrics import AverageMeter, topk_accuracy


def test_topk_accuracy() -> None:
    logits = torch.tensor([[0.1, 0.9, 0.0], [0.2, 0.1, 0.7]])
    targets = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    acc1 = topk_accuracy(logits, targets, k=1)
    assert torch.isclose(acc1, torch.tensor(1.0))

    acc2 = topk_accuracy(logits, targets, k=2)
    assert torch.isclose(acc2, torch.tensor(1.0))


def test_average_meter() -> None:
    meter = AverageMeter()
    meter.update(2.0, n=2)
    meter.update(3.0, n=1)
    assert meter.count == 3
    assert meter.average == 7.0 / 3.0
