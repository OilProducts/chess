#!/usr/bin/env bash
set -euo pipefail

python -m chessref.train.selfplay --config configs/selfplay.yaml
