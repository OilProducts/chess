# Chess Refiner

An experimental chess engine project focused on iterative refinement of policy and value predictions. The implementation follows the `plan.md` roadmap and begins by building a clean repository skeleton with modern Python tooling.

## Quick Start

1. Create and activate a Python 3.10+ virtual environment.
2. Install dependencies with `pip install -r requirements.txt` (PyTorch builds will depend on your platform and accelerator).
3. Use `make fmt`, `make lint`, and `make test` to run formatters, static checks, and tests as they become available.
4. See `plan.md` for the detailed step-by-step build plan.

## Repository Layout

The project will grow into the following structure:

```
configs/          # Hydra configurations for model, data, training
scripts/          # Entry-point shell scripts for training/evaluation
chessref/         # Python package containing implementation modules
└─ ...            # (utils, moves, data, model, train, inference, eval)
tests/            # Automated test suite
```

Stay tuned as the modules are filled out in later steps of the plan.
