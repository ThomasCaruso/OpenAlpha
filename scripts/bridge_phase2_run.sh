#!/usr/bin/env bash
# Execute the real OpenAlpha Bridge-2K Phase 2 experiment on one GPU host.
#
# Runs the locked state machine in order: preflight, retrieval, validation,
# windowing, coverage, asset resolution, Stage A, Stage B, Stage C, checkpoint
# freeze, one-time test opening, test evaluation, external evaluation, and the
# final gate decision.
#
# The reconstruction-test partition opens only because --open-test-partition is
# passed explicitly here, and only once per run identity.
#
# Usage:
#   scripts/bridge_phase2_run.sh [RUN_DIR] [CACHE_DIR] [OPERATOR]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${1:-${OPENALPHA_RUN_DIR:-/var/openalpha/runs/phase2}}"
CACHE_DIR="${2:-${OPENALPHA_CACHE_DIR:-/var/openalpha/cache/phase2}}"
OPERATOR="${3:-${OPENALPHA_OPERATOR:-unspecified-operator}}"

"${REPO_ROOT}/scripts/bridge_phase2_gpu_preflight.sh" "${RUN_DIR}" "${CACHE_DIR}"

cd "${REPO_ROOT}"
exec uv run python -m openalpha_bridge.phase2 run \
  --experiment research/bridge-v0 \
  --run-dir "${RUN_DIR}" \
  --cache-dir "${CACHE_DIR}" \
  --repository-root "${REPO_ROOT}" \
  --device cuda \
  --provider-mode real \
  --kronos-mode pinned_official \
  --evidence-class real_phase2 \
  --open-test-partition \
  --operator "${OPERATOR}" \
  --json
