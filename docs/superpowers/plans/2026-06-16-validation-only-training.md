# Validation-Only Two-Stage Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train the existing two-stage DGCNN without using the official test split for checkpoint selection, while retaining periodic and top-performing checkpoints.

**Architecture:** Add deterministic stratified splitting for the teacher training set, build separate train/validation caches, and make training evaluate only the validation cache. Keep official-test evaluation in a separate command that runs after checkpoint selection is frozen. A checkpoint retention helper will preserve named best checkpoints, every tenth epoch, and the top five validation-balanced epochs.

**Tech Stack:** Python 3.11, PyTorch, NumPy, built-in `unittest`, PowerShell.

---

### Task 1: Deterministic stratified split

**Files:**
- Modify: `pointnet_final/data.py`
- Create: `tests/test_data_split.py`

- [ ] Write tests proving reproducibility, disjointness, full coverage, and per-class representation.
- [ ] Run `conda run -n pointnet python -m unittest tests.test_data_split -v` and confirm failure because the split function does not exist.
- [ ] Implement `stratified_train_val_split(samples, val_fraction, seed)`.
- [ ] Rerun the test and confirm all cases pass.

### Task 2: Validation-only training caches

**Files:**
- Modify: `pointnet_final/train.py`
- Modify: `pointnet_final/prepare_cache.py`
- Create: `tests/test_training_protocol.py`

- [ ] Write tests asserting the default selection split is `teacher_val_split` and official test is absent from training cache construction.
- [ ] Confirm the tests fail against the old implementation.
- [ ] Add `--val-fraction`, `--split-seed`, and explicit train/validation cache construction.
- [ ] Persist `train_ids.txt` and `val_ids.txt` in each run directory.
- [ ] Ensure resumed second-stage training verifies and reuses the same split.
- [ ] Rerun all tests.

### Task 3: Checkpoint retention

**Files:**
- Create: `pointnet_final/checkpoints.py`
- Modify: `pointnet_final/train.py`
- Create: `tests/test_checkpoints.py`

- [ ] Write tests for top-five ranking and replacement by validation balanced score.
- [ ] Confirm failure before implementation.
- [ ] Implement named best checkpoints, `epoch_010.pt` cadence, and top-five `top_balanced_*.pt` retention.
- [ ] Store checkpoint manifest JSON with epoch and validation metrics.
- [ ] Run tests and verify only the configured number of top checkpoints remains.

### Task 4: Independent official-test evaluation

**Files:**
- Create: `pointnet_final/evaluate_official.py`
- Create: `tests/test_official_evaluation.py`

- [ ] Write a test proving the evaluator explicitly builds `official_test` and requires a checkpoint.
- [ ] Implement the command using the existing evaluation pipeline without training or checkpoint selection.
- [ ] Save one JSON report per evaluated checkpoint.
- [ ] Run all unit tests and Python compilation.

### Task 5: Clean old artifacts and smoke test

**Files:**
- Modify: `.gitignore`
- Modify: `README.md`
- Modify: `docs/run_instructions.md`

- [ ] Remove old tracked weights, prediction CSV files, and old evaluation reports from the new clone only.
- [ ] Point commands at `F:\Python Project\pointnet\dataset\train` and `F:\Python Project\pointnet\modelnet40_normal_resampled`.
- [ ] Run a short one-epoch smoke training in a dedicated run directory.
- [ ] Verify train/validation IDs are disjoint and checkpoints are generated.

### Task 6: Full two-stage training

**Files:**
- Generate: `runs/clean_stage1_seed2026/`
- Generate: `runs/clean_stage2_balanced_seed2026/`

- [ ] Train stage one from random initialization using only the training split.
- [ ] Monitor validation Instance Accuracy and Class Accuracy.
- [ ] Start stage two from stage-one validation-selected `best.pt`.
- [ ] Preserve the identical train/validation split and apply the existing balanced fine-tuning strategy.
- [ ] Freeze the final checkpoint choice using validation results only.

### Task 7: One-time official test and inference update

**Files:**
- Modify: `run_inference.ps1`
- Modify: `README.md`
- Modify: `docs/rehearsal_results.md`

- [ ] Run official-test evaluation once on the frozen selected checkpoints.
- [ ] Record results without retraining or changing checkpoint selection.
- [ ] Update inference modes to use only the new clean checkpoints.
- [ ] Run a final inference smoke test and CSV format validation.
- [ ] Commit the verified code, new weights, metrics, and documentation.
