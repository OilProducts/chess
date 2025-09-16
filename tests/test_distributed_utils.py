from chessref.utils import distributed


def test_distributed_defaults_when_uninitialized() -> None:
    assert distributed.is_distributed_available() is False
    assert distributed.get_world_size() == 1
    assert distributed.get_rank() == 0
    # barrier should be a no-op even if not initialised
    distributed.barrier()
