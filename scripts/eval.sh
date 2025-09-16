#!/usr/bin/env bash
set -euo pipefail

python -m chessref.eval.eval_policy --config configs/eval.yaml
