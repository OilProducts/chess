from pathlib import Path

import chess
import pytest
import torch

from chessref.data.dataset import (
    ChessSampleDataset,
    build_target_generator,
    collate_samples,
    load_records,
)
from chessref.data.engine_targets import SelfPlayTargetGenerator
from chessref.data.pgn_extract import extract_position_records
from chessref.data.planes import NUM_PLANES
from chessref.moves.encoding import NUM_MOVES, encode_move

PGN_SAMPLE = """
[Event "Test"]
[White "WhitePlayer"]
[Black "BlackPlayer"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 O-O 8. c3 d5 9. exd5 Nxd5 10. Nxe5 Nxe5 11. Rxe5 c6 12. d4 Bd6 13. Re1 Qh4 14. g3 Qh3 15. Re4 g5 16. Bxg5 Qf5 17. Bc2 Qxg5 18. Rh4 Bf5 19. Rh5 Qxh5 20. Qxh5 Bxc2 21. Nd2 Rae8 22. Nf3 Re6 23. Ne5 Bxe5 24. dxe5 Rfe8 25. f4 f5 26. Kf2 R8e7 27. Rg1 Rg6 28. Qh4 Rd7 29. h3 Be4 30. g4 Nxf4 31. gxf5 Rxg1 32. Kxg1 Rd1+ 33. Kf2 Nd3+ 34. Ke3 Re1+ 35. Kd2 Rh1 36. Qxe4 Rh2+ 37. Ke3 Rxh3+ 38. Kd4 c5+ 39. Kd5 Nf2 40. Qg2+ Kf7 41. Qxf2 Rh6 42. e6+ Ke8 43. Kd6 Rf6 44. Qxc5 Kf8 45. Ke5+ 1-0
"""


def _write_sample_pgn(tmp_path: Path) -> Path:
    pgn_path = tmp_path / "sample.pgn"
    pgn_path.write_text(PGN_SAMPLE, encoding="utf-8")
    return pgn_path


def test_dataset_basic_shapes(tmp_path: Path) -> None:
    pgn_path = _write_sample_pgn(tmp_path)
    records = extract_position_records([pgn_path])[:4]
    generator = SelfPlayTargetGenerator()
    dataset = ChessSampleDataset(records, generator)

    sample = dataset[0]
    assert sample.planes.shape == (NUM_PLANES, 8, 8)
    assert torch.isclose(sample.policy.sum(), sample.policy.new_tensor(1.0))

    batch = collate_samples([dataset[0], dataset[1]])
    assert batch.planes.shape == (2, NUM_PLANES, 8, 8)
    assert batch.policy.shape == (2, NUM_MOVES)


def test_dataset_with_transforms(tmp_path: Path) -> None:
    pgn_path = _write_sample_pgn(tmp_path)
    records = extract_position_records([pgn_path])[:2]
    generator = SelfPlayTargetGenerator()
    dataset = ChessSampleDataset(records, generator, transforms=["identity", "flip_horizontal"])

    assert len(dataset) == len(records) * 2

    base_sample = dataset[0]
    mirrored_sample = dataset[len(records)]

    board = chess.Board(mirrored_sample.metadata.fen)
    move = chess.Move.from_uci(mirrored_sample.metadata.move_uci)
    assert board.is_legal(move)
    idx = torch.argmax(mirrored_sample.policy).item()
    assert idx == encode_move(move, board)

    # Ensure value tensor shape is scalar.
    assert mirrored_sample.value.shape == ()


def test_build_target_generator_requires_stockfish_path() -> None:
    cfg = {"type": "stockfish"}
    with pytest.raises(ValueError):
        build_target_generator(cfg)
