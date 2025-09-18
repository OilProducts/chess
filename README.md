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

## Config Files

- `configs/train.yaml` controls the supervised trainer (`chessref.train.train_supervised`). It covers data sources, optimizer settings, logging, and device targeting for gradient updates.
- `configs/selfplay.yaml` feeds the self-play generator (`chessref.train.selfplay`). It defines the rollout model, MCTS parameters, dataset/PGN outputs, and optional handoff into training or Stockfish evals.
- `configs/eval.yaml` configures policy evaluation (`chessref.eval.eval_policy`). It specifies held-out PGNs, inference loop settings, and checkpoints for offline accuracy reports.
- `configs/match.yaml` drives lightweight match play (`chessref.eval.eval_match`). It sets the opponent type, match length, inference budget, and device used for head-to-head smoke tests.

```bash
python -m chessref.train.train_supervised --config configs/train.yaml \
  data.pgn_paths='["/path/to/train.pgn"]' \
  training.epochs=4 training.batch_size=128 \
  checkpoint.save_interval=500 \
  logging.enabled=true logging.experiment=chessref logging.run_name=exp1
```

Key knobs exposed via `configs/train.yaml`:
- `training.k_train`: number of refinement loops to supervise (baseline `4–8`).
- `detatch_prev_policy`: toggle for deep-supervision detach behaviour.
- `logging.enabled`: enable MLflow metric logging (see `logging.experiment` / `logging.run_name`).
- `checkpoint.save_interval`: controls checkpoint cadence.

When MLflow logging is enabled, runs are recorded in the configured tracking URI (default `./mlruns`). Launch `mlflow ui --backend-store-uri mlruns` to inspect metrics.
Training automatically resumes from the latest `iteration_*.pt` inside `checkpoint.directory` and retains the most recent 10 snapshots.
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

To see Stockfish feedback during generation, set `eval_engine_path` (and optionally `eval_depth`, `eval_print_moves`) in `configs/selfplay.yaml`. Every move is evaluated and summarised at the end of each game. Example output:

```
[selfplay] Game 3 finished result=1-0 plies=76 evaluated=76 stockfish=mean_cp_loss=18.4 blunder=2
```

Enable per-move logging by setting `eval_print_moves=true` to see lines such as `model=e2e4 best=e2e4 model_cp=12.0 best_cp=12.0 cp_loss=0.0`.

The command writes a PGN containing the requested number of self-play games **and** a tensor dataset with visit-count policy targets. Feed these back into supervised training by pointing `data.selfplay_datasets` (and optionally `data.pgn_paths`) at the generated files:

```bash
python -m chessref.train.train_supervised --config configs/train.yaml \
  data.selfplay_datasets='["data/selfplay/selfplay_50games.pt"]' \
  data.pgn_paths=[]
```

The `.pt` tensor datasets are pickled lists of `TrainingSample` objects. Each sample stores the encoded board planes, the normalised policy target (visit-count distribution from MCTS), the eventual game result for the side to move, and lightweight metadata (FEN, ply, game index). When you add these files to `data.selfplay_datasets`, the supervised trainer loads them directly into PyTorch without re-parsing PGNs, so new self-play batches can be recycled into training almost immediately.

Additional knobs (see `configs/selfplay.yaml`):
- `max_datasets`: cap how many recent self-play batches to retain (older `.pt` files are deleted).
- `train_every_batches`: run supervised training every N batches when `train_after=true`.
- `run_forever`/`max_batches`: continuously generate batches until interrupted (useful for automation).
- `eval_summary_thresholds`: configure cp-loss cutoffs that will be counted in the per-game summary (e.g., mistakes/blunders).
- `max_pgn_games`: keep at most N recent self-play games on disk; older PGN files are pruned automatically.

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
