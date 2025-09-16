#!/usr/bin/env bash
set -euo pipefail

python -m chessref.train.train_supervised --config configs/train.yaml
