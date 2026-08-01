#!/usr/bin/env bash
# GPU-host preflight for OpenAlpha Bridge-2K Phase 2.
#
# Verifies the host BEFORE any provider access or asset download. Exits non-zero
# and prints a typed blocker when a hard requirement fails.
#
# Usage:
#   scripts/bridge_phase2_gpu_preflight.sh [RUN_DIR] [CACHE_DIR]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${1:-${OPENALPHA_RUN_DIR:-/var/openalpha/runs/phase2}}"
CACHE_DIR="${2:-${OPENALPHA_CACHE_DIR:-/var/openalpha/cache/phase2}}"

echo "== OpenAlpha Bridge-2K Phase 2 GPU preflight =="
echo "repository : ${REPO_ROOT}"
echo "run dir    : ${RUN_DIR}"
echo "cache dir  : ${CACHE_DIR}"

case "${CACHE_DIR}" in
  "${REPO_ROOT}"*)
    echo "BLOCKER: CACHE_INSIDE_GIT - cache directory must live outside the worktree" >&2
    exit 2
    ;;
esac

mkdir -p "${RUN_DIR}" "${CACHE_DIR}"

echo "-- git worktree state --"
git -C "${REPO_ROOT}" rev-parse HEAD
if ! git -C "${REPO_ROOT}" diff --quiet || ! git -C "${REPO_ROOT}" diff --cached --quiet; then
  echo "WARNING: worktree is dirty; the run will record it explicitly"
fi

echo "-- forbidden tracked files --"
if git -C "${REPO_ROOT}" ls-files \
    | grep -Ei '\.(pt|pth|bin|safetensors|ckpt|onnx|npy|npz|parquet|csv|pkl|env)$'; then
  echo "BLOCKER: FORBIDDEN_TRACKED_ARTIFACT" >&2
  exit 2
fi
echo "none"

echo "-- structured preflight --"
cd "${REPO_ROOT}"
uv run python -m openalpha_bridge.phase2 preflight \
  --experiment research/bridge-v0 \
  --run-dir "${RUN_DIR}" \
  --cache-dir "${CACHE_DIR}" \
  --repository-root "${REPO_ROOT}" \
  --json
