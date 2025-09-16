# Chess Refiner: Canonical Build Plan

This plan reconciles the earlier notes and terminal-friendly checklist into one canonical roadmap. Each stage lists the key objectives, files, and definition of done so we can track progress and keep tooling decisions consistent. Completed work is noted where applicable.

---

## Stage 0 · Repository Bootstrap *(completed)*
- **Files:** `requirements.txt`, `pyproject.toml`, `README.md`, `Makefile`.
- **Dependencies:**
  - Core: `torch`, `torchvision`, `torchaudio`, `python-chess`, `tqdm`.
  - Config & logging: `hydra-core`, `omegaconf`, `tensorboard`, `wandb`.
  - Data utilities: `numpy`, `pandas`, `pyarrow`.
  - Tooling & tests: `pytest`, plus formatters/linters declared in `pyproject.toml` (`black`, `ruff`, `mypy`).
- **Actions:** create the files, record dependencies, add quick-start instructions, and wire basic make targets (`fmt`, `lint`, `typecheck`, `test`, `train`). Optionally initialise git once the scaffolding stabilises.
- **DoD:** running `make fmt` and `make lint` on the empty project succeeds.
- ✅ Files already exist; adjust dependency lists/tooling as needed when requirements change.

---

## Stage 1 · Package Skeleton *(completed)*
- **Dirs:** `chessref/{moves,data,model,train,inference,eval,utils}`, `configs/`, `scripts/`, `tests/`.
- **Files:** `__init__.py` placeholders in every package/subpackage.
- **Actions:** create the directory tree and placeholder modules so imports resolve from day one.
- **DoD:** importing `chessref` and its subpackages succeeds without side effects.
- ✅ Directory structure and placeholders are in place.

---

## Stage 2 · Move Encoding & Legal Masks *(completed)*
- **Files:** `chessref/moves/encoding.py`, `chessref/moves/legal_mask.py`, `tests/test_encoding.py`, `tests/test_masks.py`.
- **Actions:** implement the AlphaZero 8×8×73 indexing scheme, encode/decode helpers, and legal move masking backed by python-chess. Cover round-trip, underpromotion, and mask correctness in tests.
- **DoD:** `python3 -m pytest tests/test_encoding.py tests/test_masks.py` passes once dependencies are installed.
- ✅ Encoding utilities and tests exist; rerun tests after installing dev requirements.

---

## Stage 3 · Board Encoding & Augmentations
- **Files:** `chessref/data/planes.py`, `chessref/data/augment.py`, `tests/test_augment.py`.
- **Actions:**
  - Implement board-to-plane tensor conversion consistent with the model input shape.
  - Add symmetry-safe augmentations (rotations, reflections) preserving castling rights and en-passant targets.
  - Unit-test invertibility of augmentations and promotion legality after transforms.
- **DoD:** conversion and augmentation utilities round-trip sample positions; corresponding tests pass.

---

## Stage 4 · Data Extraction & Teacher Targets
- **Files:** `chessref/data/pgn_extract.py`, `chessref/data/engine_targets.py`, `configs/data.yaml`.
- **Actions:**
  - Parse PGNs (or other sources) into `(planes, move, result)` records.
  - Generate teacher policy/value targets via Stockfish (MultiPV) or cached evaluations.
  - Expose data paths, augmentation toggles, and caching options through Hydra configs.
- **DoD:** running the extraction pipeline over a small PGN sample yields tensors and teacher distributions that sum to 1 over legal moves.

---

## Stage 5 · Dataset & DataLoader Assembly
- **Files:** `chessref/data/dataset.py`, optional helpers in `chessref/utils/`.
- **Actions:**
  - Create PyTorch `Dataset`/`DataLoader` logic with augmentation hooks, cached legal masks, and distributed sharding support.
  - Provide a CLI or module entry point (e.g. `python -m chessref.data.dataset --check`) to validate shapes and batching.
- **DoD:** a smoke test loads a batch without shape mismatches; integration with Hydra config works.

---

## Stage 6 · Refinement Model & Losses
- **Files:** `chessref/model/refiner.py`, `chessref/model/loss.py`, `configs/model.yaml`.
- **Actions:**
  - Implement the transformer backbone with iterative refinement steps, optional ACT halting head, and deep supervision (detach between steps).
  - Define loss composition for policy/value/ACT terms with configurable weights.
  - Expose model hyperparameters via config.
- **DoD:** a forward pass on mock data returns per-step policies and values without runtime errors.

---

## Stage 7 · Training & Evaluation Loops
- **Files:** `chessref/train/train_supervised.py`, `chessref/train/selfplay.py` (stub), `configs/train.yaml`, `scripts/train.sh`.
- **Actions:**
  - Build the supervised training loop with gradient accumulation, mixed precision, checkpointing, and logging hooks.
  - Stub out self-play RL for future work.
  - Provide Hydra configs for optimiser, schedules, and refinement loop counts; add shell wrappers for convenience.
- **DoD:** `bash scripts/train.sh` runs a short training job, logging per-step cross-entropy/value losses.

---

## Stage 8 · Inference & Evaluation Utilities
- **Files:** `chessref/inference/predict.py`, `chessref/eval/eval_policy.py`, `chessref/eval/eval_match.py`, optional `chessref/eval/ablations.py`, `scripts/eval.sh`, `scripts/run_match.sh`.
- **Actions:**
  - Implement batched inference with configurable refinement loops or ACT halting.
  - Provide evaluation scripts for policy accuracy and engine matches.
  - Add ablation tooling to toggle refinement depth, ACT, and augmentations.
- **DoD:** evaluation scripts run end-to-end on a held-out set and (optionally) play matches against a baseline engine.

---

## Stage 9 · Testing, Monitoring & Tooling
- **Files:** Additional tests in `tests/`, logging/checkpoint helpers in `chessref/utils/`.
- **Actions:**
  - Expand unit/integration tests (dataset collation, model forward, augmentation consistency).
  - Implement logging (`utils/logging.py`), checkpointing (`utils/checkpoint.py`), distributed helpers, and Elo metrics.
  - Ensure `make test` covers the suite; integrate TensorBoard or W&B monitoring dashboards.
- **DoD:** automated tests pass; dashboards show refinement metrics decreasing across steps.

---

## Stage 10 · Tuning, Ablations & Production Prep
- **Actions:**
  - Run comparative experiments (e.g. `k_train=1` vs `k_train=8`, ACT on/off, augmentation toggles) and document results.
  - Profile training/inference (AMP, fused optimisers, `torch.compile` where stable).
  - Prepare deployment artifacts: TorchScript/ONNX export, optional UCI engine wrapper, quantisation experiments.
- **DoD:** ablation scripts demonstrate expected trends; exported models run in the target environment.

---

## Ongoing Checklist
- Keep Hydra configs (`configs/`) as the single source of truth; document deviations per experiment.
- Continuously monitor refinement-loop behaviour and ACT usage to ensure the outer loop delivers its intended gains.
- When introducing new architectural ideas (hierarchies, alternative move encodings), gate them behind targeted ablations before adoption.
- Revisit dependencies and tooling periodically to prune unused packages and keep the environment reproducible.

This canonical plan supersedes earlier documents. Update it as scope changes, and mark stages as completed once their DoD conditions are verifiably met.
