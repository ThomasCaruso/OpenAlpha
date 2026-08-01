# OpenAlpha Bridge-2K Phase 2 execution image.
#
# One GPU, at most 24 GiB VRAM, CUDA 12.4. Builds a fully locked uv environment
# so the real Stage A/B/C run is reproducible on a single accelerator host.
#
# Build:
#   docker build -f docker/bridge-phase2.Dockerfile -t openalpha-bridge-phase2:latest .
#
# Run (cache and run directories are mounted OUTSIDE the repository):
#   docker run --rm --gpus '"device=0"' \
#     -v /var/openalpha/cache:/var/openalpha/cache \
#     -v /var/openalpha/runs:/var/openalpha/runs \
#     -e OPENALPHA_OPERATOR="your-name" \
#     openalpha-bridge-phase2:latest \
#     scripts/bridge_phase2_run.sh /var/openalpha/runs/phase2 /var/openalpha/cache/phase2
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON=3.13

RUN apt-get update && apt-get install --no-install-recommends -y \
        ca-certificates \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

# Pinned uv for a reproducible resolver.
COPY --from=ghcr.io/astral-sh/uv:0.11.19 /uv /usr/local/bin/uv

WORKDIR /opt/openalpha

# Dependency manifests first so the layer caches independently of source edits.
COPY pyproject.toml uv.lock ./
COPY packages/bridge/pyproject.toml packages/bridge/
COPY packages/sentinel/pyproject.toml packages/sentinel/
COPY packages/research-core/pyproject.toml packages/research-core/
COPY packages/experiment-spec/pyproject.toml packages/experiment-spec/
COPY apps/api/pyproject.toml apps/api/
COPY apps/worker/pyproject.toml apps/worker/

RUN uv sync --locked --no-install-workspace

COPY . .

# Torch and the Kronos asset clients are extras, never base dependencies.
RUN uv sync --locked --extra bridge-gpu

# The cache and run directories must live outside the Git worktree.
ENV OPENALPHA_CACHE_DIR=/var/openalpha/cache/phase2 \
    OPENALPHA_RUN_DIR=/var/openalpha/runs/phase2
RUN mkdir -p "${OPENALPHA_CACHE_DIR}" "${OPENALPHA_RUN_DIR}"

RUN chmod +x scripts/bridge_phase2_*.sh

# Fails closed with a typed blocker when no accelerator is present.
HEALTHCHECK --interval=1m --timeout=30s --retries=1 \
    CMD uv run python -m openalpha_bridge.phase2 preflight \
        --run-dir "${OPENALPHA_RUN_DIR}" \
        --cache-dir "${OPENALPHA_CACHE_DIR}" \
        --repository-root /opt/openalpha --json || exit 1

ENTRYPOINT ["/bin/bash", "-lc"]
CMD ["scripts/bridge_phase2_gpu_preflight.sh"]
