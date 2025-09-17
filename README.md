# Chess Refiner

An experimental chess engine centred on iterative refinement (HRM-inspired) for policy/value prediction. The codebase follows the canonical roadmap in `plan.md` and now includes data ingestion, training, evaluation, and utility infrastructure.

## Environment Setup

1. Create and activate a Python 3.10+ virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Optional: add `~/Library/Python/3.11/bin` (or your platform equivalent) to `PATH` so the `pytest`, `tensorboard`, and `wandb` scripts installed in step 2 are available on the CLI.
4. Run the sanity checks:
   ```bash
   make fmt
   make lint
   make test
   ```

## Training Workflow

Supervised training consumes PGNs (human games, engine matches, or self-play logs) and writes checkpoints/metrics.

```bash
python -m chessref.train.train_supervised --config configs/train.yaml \
  data.pgn_paths='["/path/to/train.pgn"]' \
  training.epochs=4 training.batch_size=128 \
  checkpoint.save_interval=500 \
  logging.enabled=true logging.log_dir=runs/exp1
```

Key knobs exposed via `configs/train.yaml`:
- `training.k_train`: number of refinement loops to supervise (baseline `4–8`).
- `detatch_prev_policy`: toggle for deep-supervision detach behaviour.
- `logging.enabled`: if `true`, TensorBoard logs land under `logging.log_dir`.
- `checkpoint.save_interval`: controls checkpoint cadence.

Grad accumulation, AMP, and legality-aware masking are already wired in; you only need to override config values on the command line when experimenting.

## Self-Play Generation

Stage 10 introduces a self-play generator so you can bootstrap data without Stockfish. Configure it in `configs/selfplay.yaml` (see also `scripts/selfplay.sh`).

```bash
python -m chessref.train.selfplay --config configs/selfplay.yaml \
  checkpoint=checkpoints/step_5000.pt \
  num_games=50 max_moves=160 \
  output_pgn=data/selfplay/selfplay_50games.pgn \
  output_dataset=data/selfplay/selfplay_50games.pt \
  temperature=0.8
```

The command writes a PGN containing the requested number of self-play games **and** a tensor dataset with visit-count policy targets. Feed these back into supervised training by pointing `data.selfplay_datasets` (and optionally `data.pgn_paths`) at the generated files:

```bash
python -m chessref.train.train_supervised --config configs/train.yaml \
  data.selfplay_datasets='["data/selfplay/selfplay_50games.pt"]' \
  data.pgn_paths=[]
```

## Evaluation & Matches

Policy accuracy and lightweight match play are available out-of-the-box:

```bash
# Policy CE / top-1 accuracy
python -m chessref.eval.eval_policy --config configs/eval.yaml \
  data.pgn_paths='["/path/to/val.pgn"]' \
  checkpoint=checkpoints/step_5000.pt \
  inference.max_loops=2

# Rapid matches vs. random opponent (baseline smoke test)
python -m chessref.eval.eval_match --config configs/match.yaml \
  checkpoint=checkpoints/step_5000.pt \
  num_games=10 max_moves=80

# Stockfish centipawn evaluation (set `stockfish_path` first)
python -m chessref.eval.eval_stockfish --config configs/eval_stockfish.yaml \
  pgn_paths='["/path/to/val.pgn"]' \
  checkpoint=checkpoints/step_5000.pt \
  stockfish_path=/path/to/stockfish
```

Both scripts reuse the inference helper (`refine_predict`) so ACT halting and entropy thresholds behave exactly as they do in training.

## Stage 10 Checklist (Ablations, Profiling, Export)

The plan calls for several experiment tracks. Use the following commands/templates:

1. **Refinement/ACT ablations** – override `training.k_train`, `inference.max_loops`, and `model.use_act` while logging CE/top‑1 via `eval_policy`.
   ```bash
   python -m chessref.train.train_supervised --config configs/train.yaml training.k_train=1
   python -m chessref.train.train_supervised --config configs/train.yaml training.k_train=8
   python -m chessref.eval.eval_policy --config configs/eval.yaml checkpoint=... inference.use_act=true
   ```
2. **Augmentation toggles** – adjust `data.transforms` in both train/eval configs to measure the impact of mirroring vs. no augmentation.
3. **Profiling** – enable AMP (`optim.use_amp=true`), try `torch.compile` inside `train_supervised.py` (wrap model after instantiation), and measure wall-clock with `torch.profiler` or `nvprof` as appropriate.
4. **Export / Deployment** – once a model is trained, export with:
   ```python
   torch.save(model.state_dict(), "exports/refiner.pt")
   torch.jit.script(model).save("exports/refiner.ts")
   ```
   Wrap the scripted module in a simple UCI adapter to integrate with chess GUIs.

Document experiment results and profiling notes alongside configuration changes so they remain reproducible.

## Additional Resources

- `plan.md`: canonical roadmap, completed stages, and future work.
- `scripts/`: convenience wrappers (`train.sh`, `eval.sh`, `selfplay.sh`).
- `tests/`: extensive unit coverage; run `make test` before committing.

With data ingestion, model, training, evaluation, and utility scaffolding in place, the remaining work focuses on experimentation, tuning, and integrating the self-play loop into your regular training cadence.
