#!/usr/bin/env bash
# Resume an interrupted Phase 2 run from its last verified artifact.
#
# Resume is safe: the run identity is derived from the locked hashes plus the
# resolved configuration, so a changed configuration refuses to reuse the run
# directory rather than silently continuing under different settings.
#
# A run that already opened the reconstruction-test partition cannot reopen it.
#
# Usage:
#   scripts/bridge_phase2_resume.sh [RUN_DIR] [CACHE_DIR] [OPERATOR]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${1:-${OPENALPHA_RUN_DIR:-/var/openalpha/runs/phase2}}"
CACHE_DIR="${2:-${OPENALPHA_CACHE_DIR:-/var/openalpha/cache/phase2}}"
OPERATOR="${3:-${OPENALPHA_OPERATOR:-unspecified-operator}}"

if [[ ! -f "${RUN_DIR}/journal.json" ]]; then
  echo "BLOCKER: MISSING_JOURNAL - no resumable run in ${RUN_DIR}" >&2
  exit 2
fi

cd "${REPO_ROOT}"

echo "-- current run state --"
uv run python -m openalpha_bridge.phase2 verify \
  --experiment research/bridge-v0 \
  --run-dir "${RUN_DIR}" \
  --cache-dir "${CACHE_DIR}" \
  --repository-root "${REPO_ROOT}" \
  --json

echo "-- resuming --"
exec uv run python -m openalpha_bridge.phase2 run \
  --experiment research/bridge-v0 \
  --run-dir "${RUN_DIR}" \
  --cache-dir "${CACHE_DIR}" \
  --repository-root "${REPO_ROOT}" \
  --device cuda \
  --provider-mode real \
  --kronos-mode pinned_official \
  --evidence-class real_phase2 \
  --resume \
  --open-test-partition \
  --operator "${OPERATOR}" \
  --json
