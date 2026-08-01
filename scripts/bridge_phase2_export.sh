#!/usr/bin/env bash
# Export Phase 2 run artifacts for review.
#
# Exports only compact manifests, hashes, aggregates, and reports. Raw provider
# data, feature-cache shards, and model weights are never exported, matching the
# locked governance rules.
#
# Usage:
#   scripts/bridge_phase2_export.sh [RUN_DIR] [OUT_DIR]
set -euo pipefail

RUN_DIR="${1:-${OPENALPHA_RUN_DIR:-/var/openalpha/runs/phase2}}"
OUT_DIR="${2:-./phase2-export}"

if [[ ! -d "${RUN_DIR}" ]]; then
  echo "BLOCKER: MISSING_RUN_DIR - ${RUN_DIR}" >&2
  exit 2
fi

mkdir -p "${OUT_DIR}"

for name in journal.json gate_table.json test_opening_record.json; do
  if [[ -f "${RUN_DIR}/${name}" ]]; then
    cp "${RUN_DIR}/${name}" "${OUT_DIR}/"
    echo "exported ${name}"
  fi
done

# Manifests and reports only; never shards, weights, or raw candles.
find "${RUN_DIR}" -maxdepth 1 -type f \( -name '*_manifest.json' -o -name '*_metrics.json' \
  -o -name 'report.md' -o -name '*.sha256' \) -exec cp {} "${OUT_DIR}/" \;

if find "${OUT_DIR}" -type f \
    \( -name '*.pt' -o -name '*.pth' -o -name '*.safetensors' -o -name '*.npz' \
       -o -name '*.csv' -o -name '*.parquet' \) | grep -q .; then
  echo "BLOCKER: FORBIDDEN_ARTIFACT_IN_EXPORT" >&2
  exit 2
fi

( cd "${OUT_DIR}" && find . -type f -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS )
echo "export complete: ${OUT_DIR}"
