from pathlib import Path

import chess
import pytest
import torch

from chessref.data.engine_targets import SelfPlayTargetGenerator
from chessref.data.pgn_extract import (
    PositionRecord,
    build_training_samples,
    extract_position_records,
    iter_position_records,
)
from chessref.moves.encoding import encode_move


PGN_SAMPLE = """
[Event "Test"]
[White "WhitePlayer"]
[Black "BlackPlayer"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 O-O 8. c3 d5 9. exd5 Nxd5 10. Nxe5 Nxe5 11. Rxe5 c6 12. d4 Bd6 13. Re1 Qh4 14. g3 Qh3 15. Re4 g5 16. Bxg5 Qf5 17. Bc2 Qxg5 18. Rh4 Bf5 19. Rh5 Qxh5 20. Qxh5 Bxc2 21. Nd2 Rae8 22. Nf3 Re6 23. Ne5 Bxe5 24. dxe5 Rfe8 25. f4 f5 26. Kf2 R8e7 27. Rg1 Rg6 28. Qh4 Rd7 29. h3 Be4 30. g4 Nxf4 31. gxf5 Rxg1 32. Kxg1 Rd1+ 33. Kf2 Nd3+ 34. Ke3 Re1+ 35. Kd2 Rh1 36. Qxe4 Rh2+ 37. Ke3 Rxh3+ 38. Kd4 c5+ 39. Kd5 Nf2 40. Qg2+ Kf7 41. Qxf2 Rh6 42. e6+ Ke8 43. Kd6 Rf6 44. Qxc5 Kf8 45. Ke5+ 1-0
"""


def test_iter_position_records(tmp_path: Path) -> None:
    pgn_path = tmp_path / "sample.pgn"
    pgn_path.write_text(PGN_SAMPLE, encoding="utf-8")

    records = list(iter_position_records(pgn_path))
    assert len(records) > 0
    sample = records[0]
    assert isinstance(sample, PositionRecord)
    assert sample.fen.startswith("rnbqkbnr")
    assert sample.move_uci == "e2e4"
    assert sample.white_result == 1.0


def test_build_training_samples_selfplay(tmp_path: Path) -> None:
    pgn_path = tmp_path / "sample.pgn"
    pgn_path.write_text(PGN_SAMPLE, encoding="utf-8")

    records = extract_position_records([pgn_path])[:5]
    generator = SelfPlayTargetGenerator()
    samples = build_training_samples(records, generator)

    assert len(samples) == len(records)
    for record, sample in zip(records, samples):
        board = chess.Board(record.fen)
        move = chess.Move.from_uci(record.move_uci)
        assert pytest.approx(1.0, abs=1e-6) == sample.policy.sum().item()
        idx = torch.argmax(sample.policy).item()
        assert idx == encode_move(move, board)
        assert sample.value.shape == ()
