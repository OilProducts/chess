# Chess Refiner: Canonical Build Plan

This plan reconciles the earlier notes and terminal-friendly checklist into one canonical roadmap. Each stage lists the key objectives, files, and definition of done so we can track progress and keep tooling decisions consistent. Completed work is noted where applicable.


Below is a drop‑in section you can paste **at the very top** of your planning document. It explains what “HRM‑inspired” means here and how it’s realized in this repo.

---

# What “HRM‑Inspired” Means in This Repo

**Purpose.** The original Hierarchical Reasoning Model (HRM) shows that *iterative latent computation*—short bursts of internal reasoning with an option to “halt” early—can substitute for brittle, token‑level step‑by‑step chains. ARC Prize’s ablation study further found that **the outer refinement loop and training with multiple refinement steps** were the main contributors to HRM’s gains, while the explicit two‑timescale H/L hierarchy added little over a comparable transformer.

**Our adaptation for chess.** We keep the parts that clearly help and fit the domain:

* ✅ **Iterative refinement (outer loop).** The model predicts a policy/value, then **feeds its own policy back in** as context and refines for a few steps.
* ✅ **Deep supervision with detach.** We compute a loss **at each refinement step** and **detach** the previous step’s outputs to avoid full BPTT—mirroring HRM’s 1‑step/segment training effect.
* ✅ **Adaptive halting (ACT).** A small “halt/continue” head decides whether more refinement is worth it; at inference we can early‑stop or enforce a small minimum.
* ✅ **Inference‑time scaling.** Harder positions can get more refinement steps; easy ones can stop quickly.
* ✅ **Augmentations.** Like HRM’s task augs, we use chess‑safe symmetries (mirror, rotate180+color‑swap) and label tricks (e.g., temperature jitter) to strengthen generalization.

We **do not** include a puzzle/position‑ID embedding (which ARC found to drive memorization) and we start with a **single transformer stack** rather than separate H/L modules. A two‑timescale variant is left as an optional extension once the baseline is stable.

---

## Minimal Mental Model

Each **refinement step** $k$ takes the board features and the **previous step’s policy** and returns a refined (policy, value) and a **halt** score:

```
(board_planes, prev_policy_k-1) ──> Transformer ──> policy_k, value_k, halt_k
                                          ↑
                                   (used next step)
```

Training runs **K\_train** steps (e.g., 8–16), with a loss at each step and **detach** between steps:

1. Step k: compute losses (policy CE, value MSE, small ACT BCE)
2. Detach the softmax(policy\_k) → becomes prev\_policy\_k for the next step
3. Average losses over steps and update

Inference runs 1–4 steps with ACT/entropy to early‑stop.

---

## Where Each Concept Lives (File Map)

* **Outer refinement step**
  `chessref/model/refiner.py`

  * `IterativeRefiner.forward_step(...)` embeds the board, fuses `prev_policy`, runs a **Transformer encoder stack**, and returns `(policy_logits, value, halt_logits, state)`.

* **Deep supervision (detach between steps)**
  `chessref/train/train_supervised.py`

  * The training loop calls `forward_step` **K\_train** times; after each step it **detaches** the softmaxed policy before feeding it into the next step.

* **Policy/value losses + masking**
  `chessref/model/loss.py`

  * `policy_ce` applies the **legal‑move mask** (−∞ on illegal indices) before softmax.

* **ACT halting**
  `chessref/model/refiner.py` (head) and `chessref/model/loss.py` (targets)

  * Small BCE term encourages correct halting; can be disabled or replaced by entropy thresholds.

* **Inference‑time scaling / early stop**
  `chessref/inference/predict.py`

  * `refine_predict(..., max_loops, min_loops, entropy_stop, use_act)`.

* **Chess‑safe augmentations**
  `chessref/data/augment.py` with tests in `tests/test_augment.py`.

* **Legal move masking & indexing (AlphaZero 8×8×73)**
  `chessref/moves/encoding.py`, `chessref/moves/legal_mask.py` with tests in `tests/test_encoding.py`, `tests/test_masks.py`.

* **Teacher targets (MultiPV/MCTS) and dataset**
  `chessref/data/engine_targets.py`, `chessref/data/dataset.py`.

* **Evaluation & Ablations**
  `chessref/eval/eval_policy.py`, `chessref/eval/eval_match.py`, optional `chessref/eval/ablations.py`.

---

## Why This Design (vs. “full” HRM)

* **Keeps the winners.** ARC’s evaluation showed the **outer loop** and **training with multiple refinement steps** drove most of the gains. We center the codebase on those.
* **Avoids non‑generalizing shortcuts.** No position IDs or task‑specific embeddings that would leak identity.
* **Simple, extensible core.** Start with a single encoder stack. If later ablations justify it, you can introduce a **two‑timescale variant** by adding a slower “planner” block that updates every N steps while a faster block updates each step (see “Optional extensions” below).

---

## Defaults & Knobs (config‑friendly)

* `K_train` (training refinement steps): **8** (good starting point; 16 if you have VRAM/time).
* `max_loops` (inference): **1–4** (allocate more on “hard” positions).
* `act_weight`: **0.1** (start at 0, warm in after a few epochs).
* `d_model/depth/heads`: baseline \~30–35M params (e.g., 512/8/8).
* Augmentations: **mirror** + **rotate180+color‑swap** always on; add **label smoothing** and **temperature jitter** for teacher policies.

---

## Success Criteria (what to watch)

* **Refinement helps at train‑time:** `K_train=8` beats `K_train=1` on **held‑out CE/top‑1** even when inference uses a **single** step.
* **Early‑stop saves compute:** ACT or entropy reduces average steps with negligible strength loss.
* **Entropy drops across steps:** policy entropy curves decrease from step 1 → step K for hard positions.

---

This section is the conceptual anchor for the repo: **an iterative chess policy/value model that “thinks a bit in latent space,” learns when to stop, and plays stronger without relying on textual chains or identity hacks.**



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

## Stage 3 · Board Encoding & Augmentations *(completed)*
- **Files:** `chessref/data/planes.py`, `chessref/data/augment.py`, `tests/test_planes.py`, `tests/test_augment.py`.
- **Actions:**
  - Implement board-to-plane tensor conversion consistent with the model input shape.
  - Add symmetry-safe augmentations (currently horizontal reflection; extend with additional symmetries once legality mapping is proven robust) preserving castling rights and en-passant targets.
  - Unit-test tensor features, promotion, and en-passant handling after transforms.
- **DoD:** `python3 -m pytest tests/test_planes.py tests/test_augment.py` passes.
- ✅ Implemented board plane encoder and horizontal mirror augmentation with coverage for promotions and en-passant remapping.

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
