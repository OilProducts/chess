"""Dataset and DataLoader utilities for chess refinement training."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from chessref.data.augment import TRANSFORMS, BoardTransform, transform_move
from chessref.data.engine_targets import SelfPlayTargetGenerator, StockfishTargetGenerator
from chessref.data.pgn_extract import (
    PositionRecord,
    TrainingSample,
    build_training_samples,
    extract_position_records,
    TargetGenerator,
)


@dataclass
class SampleBatch:
    planes: torch.Tensor
    policy: torch.Tensor
    value: torch.Tensor
    metadata: List[PositionRecord]


def _select_transforms(transform_names: Sequence[str]) -> List[Optional[BoardTransform]]:
    mapping = {tf.name: tf for tf in TRANSFORMS}
    selected: List[Optional[BoardTransform]] = []
    for name in transform_names:
        if name == "identity":
            selected.append(None)
            continue
        tf = mapping.get(name)
        if tf is None:
            raise ValueError(f"Unknown transform '{name}'. Available: {sorted(mapping)}")
        selected.append(tf)
    if not selected:
        selected.append(None)
    return selected


class ChessSampleDataset(Dataset[TrainingSample]):
    """Dataset that materialises training samples on demand."""

    def __init__(
        self,
        records: Sequence[PositionRecord],
        target_generator: TargetGenerator,
        *,
        transforms: Sequence[str] | None = None,
    ) -> None:
        self._records = list(records)
        self._target_generator = target_generator
        self._transforms = _select_transforms(transforms or ["identity"])
        # Expand indices to include transforms.
        self._index: List[Tuple[int, Optional[BoardTransform]]] = []
        for i, record in enumerate(self._records):
            for transform in self._transforms:
                self._index.append((i, transform))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> TrainingSample:
        record_idx, transform = self._index[idx]
        base_record = self._records[record_idx]
        board = base_record.board()
        move = base_record.move()

        if transform is not None:
            board = transform.apply(board)
            move = transform_move(move, transform)
            record = PositionRecord(
                fen=board.fen(),
                move_uci=move.uci(),
                white_result=base_record.white_result,
                ply=base_record.ply,
                game_index=base_record.game_index,
            )
        else:
            record = base_record

        sample = build_training_samples([record], self._target_generator)[0]
        return sample


class PrecomputedSampleDataset(Dataset[TrainingSample]):
    """Dataset backed by precomputed :class:`TrainingSample` objects."""

    def __init__(self, samples: Sequence[TrainingSample]) -> None:
        self._samples = list(samples)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> TrainingSample:
        return self._samples[idx]


def collate_samples(samples: Sequence[TrainingSample]) -> SampleBatch:
    planes = torch.stack([sample.planes for sample in samples])
    policy = torch.stack([sample.policy for sample in samples])
    value = torch.stack([sample.value for sample in samples])
    metadata = [sample.metadata for sample in samples]
    return SampleBatch(planes=planes, policy=policy, value=value, metadata=metadata)


def build_target_generator(cfg: dict) -> TargetGenerator:
    target_type = cfg.get("type", "selfplay")
    if target_type == "selfplay":
        return SelfPlayTargetGenerator()
    if target_type == "stockfish":
        path = cfg.get("stockfish_path")
        if not path:
            raise ValueError("stockfish_path must be provided for stockfish target generation")
        return StockfishTargetGenerator(
            engine_path=path,
            depth=cfg.get("depth"),
            nodes=cfg.get("nodes"),
            multipv=cfg.get("multipv", 8),
            temperature=cfg.get("temperature", 1.0),
        )
    raise ValueError(f"Unknown target type '{target_type}'")


def load_records(paths: Sequence[Path], *, max_games: Optional[int] = None) -> List[PositionRecord]:
    return extract_position_records(paths, max_games=max_games)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Dataset smoke test")
    parser.add_argument("pgn", type=Path, help="Path to a PGN file")
    parser.add_argument("--max-games", type=int, default=1)
    parser.add_argument("--target", type=str, default="selfplay", choices=["selfplay", "stockfish"])
    parser.add_argument("--stockfish-path", type=Path, default=None)
    parser.add_argument("--transforms", type=str, nargs="*", default=["identity"])
    args = parser.parse_args(argv)

    records = load_records([args.pgn], max_games=args.max_games)
    if args.target == "stockfish":
        if args.stockfish_path is None:
            parser.error("--stockfish-path is required when --target=stockfish")
        generator = StockfishTargetGenerator(str(args.stockfish_path))
    else:
        generator = SelfPlayTargetGenerator()
    dataset = ChessSampleDataset(records, generator, transforms=args.transforms)

    sample = dataset[0]
    print(json.dumps(
        {
            "planes_shape": list(sample.planes.shape),
            "policy_sum": float(sample.policy.sum().item()),
            "value": float(sample.value.item()),
            "metadata": sample.metadata.move_uci,
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()


__all__ = [
    "ChessSampleDataset",
    "PrecomputedSampleDataset",
    "SampleBatch",
    "collate_samples",
    "build_target_generator",
    "load_records",
]
