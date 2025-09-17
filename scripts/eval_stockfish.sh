#!/usr/bin/env bash
set -euo pipefail

python -m chessref.eval.eval_stockfish --config configs/eval_stockfish.yaml
