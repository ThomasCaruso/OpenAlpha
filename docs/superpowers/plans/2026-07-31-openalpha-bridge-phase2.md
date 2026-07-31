# OpenAlpha Bridge Phase 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reach one evidence-backed terminal Bridge-2K feasibility conclusion without changing the frozen architecture, opening test data early, or accessing the Sentinel holdout.

**Architecture:** Treat the original experiment file as immutable and apply at most one content-hashed Phase 2 amendment before external access. The pre-data gate validates that every chronological partition can form the locked 512-candle examples; a failure emits compact terminal artifacts and stops before provider, checkpoint, feature-cache, training, or evaluation work.

**Tech Stack:** Git, YAML/JSON, SHA-256, pytest, Ruff, Pyright, existing OpenAlpha preservation checks.

---

### Task 1: Verify the immutable inputs and pre-data feasibility

**Files:**
- Read: `research/bridge-v0/experiment.yaml`
- Read: `research/bridge-v0/preregistration.md`
- Read: `docs/KRONOS_COMPATIBILITY_BOUNDARY.md`
- Read: `docs/BRIDGE_MATHEMATICAL_REPRESENTATION.md`
- Read: `docs/BRIDGE_NUMERICAL_CONTRACT.md`

- [x] **Step 1: Verify branch and ancestry**

Run `git status --short --branch`, `git rev-parse HEAD`, and `git merge-base --is-ancestor` for Phase 0 and Phase 1.

- [x] **Step 2: Verify the original experiment bytes**

Run `Get-FileHash -Algorithm SHA256 research/bridge-v0/experiment.yaml` and require `d52a9be733f4ec331e081346e64ab7415ac0f9510e494733746f975f114f47b2`.

- [x] **Step 3: Audit every required lock field**

Record provider, symbols, frequency, periods, splits, windows, loss, optimizer, learning-rate behavior, early stopping, seed, metrics, bootstrap, success gates, failure gates, and operational ambiguities.

- [x] **Step 4: Prove partition feasibility without provider access**

Count calendar days and weekdays in each half-open locked period. Require at least 512 possible complete daily candles in validation and test before any network or checkpoint operation.

Expected result: validation has at most 260 weekdays and reconstruction test at most 390; both fail.

### Task 2: Seal the single pre-data amendment

**Files:**
- Create: `research/bridge-v0/phase2-preregistration-amendment.yaml`
- Create: `research/bridge-v0/phase2-preregistration-amendment.sha256`
- Create: `research/bridge-v0/phase2-preregistration-amendment.md`

- [ ] **Step 1: Write the immutable amendment overlay**

Preserve the original experiment file and hash. Lock literal non-crossing semantics, constant learning rate, deterministic tie-breaking, staged subsets, coverage and CPU gates, and the provider-free infeasibility proof.

- [ ] **Step 2: Compute and store the amendment SHA-256**

Run `Get-FileHash -Algorithm SHA256 research/bridge-v0/phase2-preregistration-amendment.yaml` and store the lowercase digest as the new Phase 2 experiment hash.

- [ ] **Step 3: Commit before external access**

Commit only the amendment, hash, rationale, and this plan. Do not access a market provider or Kronos checkpoint.

### Task 3: Specify terminal-artifact integrity first

**Files:**
- Create: `tests/smoke/test_bridge_phase2_terminal_artifacts.py`
- Create: `research/bridge-v0/phase2/*.json`
- Create: `research/bridge-v0/phase2/training_history.jsonl`
- Create: `research/bridge-v0/phase2/report.md`

- [ ] **Step 1: Write the failing terminal-artifact test**

Require all requested Phase 2 artifacts, the `OPERATIONALLY_BLOCKED` conclusion, zero provider/checkpoint/training/test access, original and amendment hashes, null scientific metrics, and preserved tree identities.

- [ ] **Step 2: Run the test and observe the missing-artifact failure**

Run `uv run pytest tests/smoke/test_bridge_phase2_terminal_artifacts.py -q`.

Expected result: FAIL because `research/bridge-v0/phase2/data_manifest.json` does not yet exist.

- [ ] **Step 3: Add minimal canonical terminal artifacts**

Create the ten requested artifacts with explicit `NOT_RUN` states rather than fabricated data, training, checkpoint, or evaluation values.

- [ ] **Step 4: Run the test and observe it pass**

Run `uv run pytest tests/smoke/test_bridge_phase2_terminal_artifacts.py -q`.

Expected result: PASS.

### Task 4: Update factual status documentation

**Files:**
- Modify: `docs/OPENALPHA_KRONOS_BRIDGE.md`
- Modify: `docs/KRONOS_COMPATIBILITY_BOUNDARY.md`
- Modify: `docs/BRIDGE_NUMERICAL_CONTRACT.md`
- Modify: `docs/STATUS.md`
- Modify: `README.md`

- [ ] **Step 1: Add only the terminal Phase 2 status**

State that no reconstruction evidence was produced, no Bridge quality conclusion is justified, and the blocker is the locked partition/window contract.

- [ ] **Step 2: Preserve Phase 0/1 and Sentinel claims**

Do not rewrite prior empirical results, hashes, or research conclusions.

### Task 5: Verify and commit the terminal result

**Files:**
- Verify: entire repository

- [ ] **Step 1: Run targeted and complete tests**

Run Bridge tests, the new smoke test, and the complete pytest suite.

- [ ] **Step 2: Run static and preservation gates**

Run Ruff, Pyright, original/amendment lock checks, Sentinel/research tree preservation checks, secret/policy scans, and `git diff --check`.

- [ ] **Step 3: Review the complete diff**

Require no provider data, model weights, checkpoints, caches, or changes to original lock and Sentinel artifacts.

- [ ] **Step 4: Commit one terminal Phase 2 result**

Commit the tested artifacts and factual documentation. Stop without forecasting integration.
