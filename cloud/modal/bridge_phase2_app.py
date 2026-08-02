# pyright: reportMissingImports=false, reportAttributeAccessIssue=false
# pyright: reportFunctionMemberAccess=false
# This module targets the deployed Modal image, where `modal` is installed. The
# base development environment deliberately has no cloud SDK, so static analysis
# cannot resolve these symbols locally.
"""OpenAlpha Bridge-2K Phase 2 Modal application.

One managed application: lightweight CPU web endpoints for the control plane,
one background GPU worker that runs the complete empirical pipeline, and a
separate CPU worker for synthetic validation.

Deploy:
    modal deploy cloud/modal/bridge_phase2_app.py

Nothing here runs on a user workstation. There is no GPU host to administer, no
CUDA to install, and no artifact to copy by hand.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import modal

APP_NAME = "openalpha-bridge-phase2"
APP_VERSION = "1.0.0"

# --------------------------------------------------------------------- image

# Pinned, code-defined, built by Modal. The user never builds Docker locally.
PYTHON_VERSION = "3.13"
TORCH_VERSION = "2.5.1"
CUDA_INDEX = "https://download.pytorch.org/whl/cu124"

# The pinned official Kronos source. Mirrors experiment.yaml official_source;
# openalpha_bridge.phase2.kronos.SOURCE_SPEC is the authority and a deployment
# test asserts the two agree.
KRONOS_SOURCE_REPOSITORY = "https://github.com/shiyu-coder/Kronos"
KRONOS_SOURCE_REVISION = "67b630e67f6a18c9e9be918d9b4337c960db1e9a"
KRONOS_SOURCE_ROOT = "/opt/kronos"
KRONOS_SOURCE_FILES: dict[str, str] = {
    "model/kronos.py": "638a56e035856c600c9848b368be087cb706a61603a0790124968c95b8c69f3a",
    "model/module.py": "a07edbadc0e96804c8158c021bbc6063bb7cc43b34d7fc470d5c8ff2005a409f",
}

def _repo_root() -> Path | None:
    """The repository root when deploying, or None inside the container.

    Modal re-imports this module in the container to resolve a function, where
    it is flattened to /root/<name>.py. Unconditionally taking parents[2] there
    raises IndexError and every container crash-loops on startup.

    The client imports it from cloud/modal/, so the root is two levels up. The
    layout check keeps this honest rather than trusting the depth alone.
    """
    here = Path(__file__).resolve()
    if len(here.parents) < 3:
        return None
    candidate = here.parents[2]
    return candidate if (candidate / "packages" / "bridge").is_dir() else None

#: Workspace packages copied into the image, and their import locations.
#:
#: `add_local_python_source` is deliberately not used: it resolves a package by
#: importing it in the deploying process. `modal deploy` runs from an isolated
#: uvx environment where the workspace packages are not installed, so it fails
#: with "openalpha_bridge has no spec - might not be installed?". Explicit
#: directory copies have no such dependency.
#:
#: openalpha_bridge.validation imports openalpha_sentinel.structural_validity,
#: so sentinel is a runtime requirement, not merely a declared one.
LOCAL_PACKAGES: tuple[tuple[str, str], ...] = (
    ("packages/bridge/src/openalpha_bridge", "/root/openalpha_bridge"),
    ("packages/sentinel/src/openalpha_sentinel", "/root/openalpha_sentinel"),
    ("packages/research-core/src/openalpha_research", "/root/openalpha_research"),
)

#: Bytecode from the host Python must never shadow the image's own.
_IGNORE = ["__pycache__", "**/__pycache__", "*.pyc", "*.pyo"]


def _build_image() -> Any:
    built = (
        modal.Image.debian_slim(python_version=PYTHON_VERSION)
        .apt_install("git")
        .pip_install(
            f"torch=={TORCH_VERSION}",
            extra_index_url=CUDA_INDEX,
        )
        .pip_install(
            "numpy>=2.5,<3",
            "pydantic>=2.11,<3",
            "yfinance==1.5.2",
            "huggingface-hub>=0.34,<1",
            "safetensors>=0.4,<1",
            "boto3>=1.35,<2",
            "fastapi>=0.115,<1",
            # Runtime dependencies of the pinned official Kronos source. Pinned
            # explicitly rather than relied on transitively.
            "pandas>=2.2,<3",
            "tqdm>=4.66,<5",
            "einops>=0.8,<1",
        )
        # Clone the pinned official source, check out the exact detached
        # revision, and verify both the revision and every locked file hash.
        # Any mismatch fails the image build rather than surfacing at runtime.
        .run_commands(
            f"git clone --no-checkout {KRONOS_SOURCE_REPOSITORY} {KRONOS_SOURCE_ROOT}",
            f"cd {KRONOS_SOURCE_ROOT} && git checkout --detach {KRONOS_SOURCE_REVISION}",
            (
                f"cd {KRONOS_SOURCE_ROOT} && "
                f'test "$(git rev-parse HEAD)" = "{KRONOS_SOURCE_REVISION}" '
                f'|| (echo "KRONOS_SOURCE_REVISION_MISMATCH" && exit 1)'
            ),
            (
                f"cd {KRONOS_SOURCE_ROOT} && git status --porcelain | tee /tmp/kronos_dirty && "
                f'test ! -s /tmp/kronos_dirty || (echo "KRONOS_WORKTREE_DIRTY" && exit 1)'
            ),
            *(
                f'cd {KRONOS_SOURCE_ROOT} && echo "{digest}  {relative}" | sha256sum -c - '
                f'|| (echo "KRONOS_SOURCE_HASH_MISMATCH {relative}" && exit 1)'
                for relative, digest in KRONOS_SOURCE_FILES.items()
            ),
        )
        .env({"OPENALPHA_KRONOS_SOURCE_PATH": KRONOS_SOURCE_ROOT})
        # /root is the container working directory; making it explicit keeps the
        # copied packages importable regardless of how a function is invoked.
        .env({"PYTHONPATH": "/root"})
    )

    root = _repo_root()
    if root is None:
        # Container: the image is already built and no repository is present, so
        # there is nothing to add. Returning the base keeps module import safe.
        return built

    # Source is added last so edits do not invalidate the dependency layers.
    for local_path, remote_path in LOCAL_PACKAGES:
        source = root / local_path
        if not source.is_dir():
            raise FileNotFoundError(f"declared local package is missing: {source}")
        built = built.add_local_dir(str(source), remote_path=remote_path, ignore=_IGNORE)

    research = root / "research" / "bridge-v0"
    if not research.is_dir():
        raise FileNotFoundError(f"locked research directory is missing: {research}")
    return built.add_local_dir(
        str(research), remote_path="/root/research/bridge-v0", ignore=_IGNORE
    )


image = _build_image()

app = modal.App(APP_NAME)

# ------------------------------------------------------------------- storage

# Reusable cache: Hugging Face assets, derived features, resumable checkpoints.
cache_volume = modal.Volume.from_name("openalpha-bridge-phase2-cache", create_if_missing=True)
CACHE_ROOT = "/cache"

# ------------------------------------------------------------------- secrets

# Names only. Values live in Modal's managed secret store and never appear in
# code, requests, journals, logs, or artifacts.
SECRET_NAMES = ("openalpha-api", "openalpha-storage", "openalpha-huggingface")
secrets = [modal.Secret.from_name(name) for name in SECRET_NAMES]

# ------------------------------------------------------------------- limits

GPU_CONFIG = "T4"  # 16 GiB, comfortably inside the 24 GiB lock
MAX_GPU_SECONDS = 24 * 60 * 60
CONTROL_TIMEOUT = 60


def _now() -> datetime:
    return datetime.now(UTC)


def _build_store() -> Any:
    """S3-compatible artifact store, defaulting to Cloudflare R2."""
    from openalpha_bridge.cloud.objectstore import S3CompatibleObjectStore

    return S3CompatibleObjectStore(
        bucket=os.environ["OPENALPHA_ARTIFACT_BUCKET"],
        endpoint_url=os.environ.get("OPENALPHA_S3_ENDPOINT_URL"),
        region_name=os.environ.get("OPENALPHA_S3_REGION", "auto"),
        stage="cloud-storage",
    )


def _register_secrets() -> None:
    from openalpha_bridge.cloud.redaction import default_redactor

    default_redactor().register_environment(dict(os.environ))


def _service() -> Any:
    """Control service bound to the Modal compute backend."""
    from openalpha_bridge.cloud.service import Phase2ControlService, ServiceConfig
    from openalpha_bridge.phase2.kronos import TOKENIZER_SPEC

    _register_secrets()
    config = ServiceConfig(
        base_url=os.environ.get("OPENALPHA_API_BASE_URL", "https://example.modal.run"),
        object_store_namespace=os.environ.get("OPENALPHA_ARTIFACT_BUCKET", "openalpha"),
        dependency_lock_sha256=os.environ.get("OPENALPHA_DEPENDENCY_LOCK_SHA256", "0" * 64),
        kronos_revision=TOKENIZER_SPEC.revision,
        provider_identity="yahoo_finance/yfinance==1.5.2",
    )
    return Phase2ControlService(store=_build_store(), backend=ModalComputeBackend(), config=config)


class ModalComputeBackend:
    """Modal implementation of the narrow ComputeBackend protocol."""

    name = "modal"

    @property
    def image_digest(self) -> str:
        return os.environ.get("OPENALPHA_IMAGE_DIGEST", f"{APP_NAME}:{APP_VERSION}")

    @property
    def app_version(self) -> str:
        return APP_VERSION

    def spawn_real_run(self, run_id: str, payload: dict[str, Any]) -> str:
        handle = phase2_gpu_worker.spawn(run_id=run_id, payload=payload)
        return handle.object_id

    def spawn_synthetic_run(self, run_id: str, payload: dict[str, Any]) -> str:
        handle = phase2_synthetic_worker.spawn(run_id=run_id, payload=payload)
        return handle.object_id

    def execution_status(self, cloud_execution_id: str) -> str:
        try:
            handle = modal.FunctionCall.from_id(cloud_execution_id)
            handle.get(timeout=0)
        except TimeoutError:
            return "running"
        except Exception:  # noqa: BLE001 - status is advisory only
            return "unknown"
        return "completed"

    def cancel(self, cloud_execution_id: str) -> None:
        modal.FunctionCall.from_id(cloud_execution_id).cancel()


# ------------------------------------------------------- control API (CPU)


@app.function(image=image, secrets=secrets, timeout=CONTROL_TIMEOUT)
@modal.asgi_app(label="openalpha-phase2-api")
def control_api():
    """The documented REST control plane. See docs/BRIDGE_CLOUD_API.md.

    One ASGI application rather than per-function endpoints: Modal's
    fastapi_endpoint binds every declared parameter as a body or query field, so
    a `headers` parameter never receives HTTP headers and a path parameter such
    as `{run_id}` cannot be expressed at all.
    """
    from openalpha_bridge.cloud.http import build_control_api

    _register_secrets()
    return build_control_api(
        service_factory=_service,
        token_provider=lambda: os.environ.get("OPENALPHA_API_TOKEN"),
        version=APP_VERSION,
    )


# ------------------------------------------------------- Phase 2 GPU worker


@app.function(
    image=image,
    gpu=GPU_CONFIG,
    volumes={CACHE_ROOT: cache_volume},
    secrets=secrets,
    timeout=MAX_GPU_SECONDS,
    retries=0,  # a cloud retry must never duplicate a scientific run
)
def phase2_gpu_worker(run_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Execute the complete empirical Phase 2 pipeline on exactly one GPU."""
    from pathlib import Path

    from openalpha_bridge.cloud.identity import CloudRunIdentity
    from openalpha_bridge.cloud.objectstore import get_model
    from openalpha_bridge.cloud.runner import CloudRunner, ResourceGuards
    from openalpha_bridge.phase2.kronos import OfficialKronosBackend
    from openalpha_bridge.phase2.pipeline import Phase2Config
    from openalpha_bridge.phase2.provider import ProviderMode, YahooDailyProvider
    from openalpha_bridge.phase2.states import EvidenceClass
    from openalpha_bridge.phase2.training import TorchTrainingBackend

    _register_secrets()
    store = _build_store()

    identity_key = (
        f"openalpha/bridge-phase2/runs/{run_id}/identity/run_identity.json"
    )
    identity = get_model(store, identity_key, CloudRunIdentity)

    cache_root = Path(CACHE_ROOT)
    os.environ.setdefault("HF_HOME", str(cache_root / "huggingface"))

    config = Phase2Config(
        run_directory=cache_root / "runs" / run_id,
        cache_directory=cache_root / "features" / run_id,
        research_root=Path("/root/research/bridge-v0"),
        repository_root=Path("/root"),
        evidence_class=EvidenceClass.REAL_PHASE2,
        provider_mode=ProviderMode.REAL,
        kronos_mode=OfficialKronosBackend(stage="stage-c").mode,
        device="cuda",
        require_accelerator=True,
    )

    runner = CloudRunner(
        store=store,
        identity=identity,
        pipeline_config=config,
        provider=YahooDailyProvider(stage="retrieve"),
        kronos=OfficialKronosBackend(stage="stage-c", cache_dir=str(cache_root / "huggingface")),
        training_backend=TorchTrainingBackend(stage="stage-c", device="cuda"),
        guards=ResourceGuards(),
    )
    result = runner.execute(
        holder=f"modal:{os.environ.get('MODAL_TASK_ID', run_id)}",
        confirm_open_test_partition=bool(payload.get("confirm_open_test_partition")),
    )
    cache_volume.commit()
    return {
        "run_id": result.run_id,
        "final_state": result.final_state.value,
        "conclusion": result.conclusion.value if result.conclusion else None,
        "blocker_code": result.blocker_code,
    }


# ------------------------------------------------- synthetic worker (CPU)


@app.function(image=image, secrets=secrets, timeout=1800)
def phase2_synthetic_worker(run_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    """SYNTHETIC CLOUD PIPELINE VALIDATION - NOT EMPIRICAL EVIDENCE.

    Runs the real state machine and adapter against fake components on CPU. It
    writes to a separate object-store namespace and can never create the real
    test-opening record.
    """
    import tempfile
    from pathlib import Path

    from openalpha_bridge.cloud.identity import CloudRunIdentity
    from openalpha_bridge.cloud.objectstore import get_model
    from openalpha_bridge.cloud.runner import CloudRunner
    from openalpha_bridge.phase2.kronos import DeterministicFakeKronosBackend, KronosMode
    from openalpha_bridge.phase2.pipeline import Phase2Config
    from openalpha_bridge.phase2.provider import DeterministicFakeProvider, ProviderMode
    from openalpha_bridge.phase2.states import EvidenceClass
    from openalpha_bridge.phase2.training import NumpyTrainingBackend

    _register_secrets()
    store = _build_store()

    identity_key = (
        f"openalpha-synthetic/bridge-phase2/runs/{run_id}/identity/run_identity.json"
    )
    identity = get_model(store, identity_key, CloudRunIdentity)

    with tempfile.TemporaryDirectory() as scratch:
        root = Path(scratch)
        config = Phase2Config(
            run_directory=root / "run",
            cache_directory=root / "cache",
            research_root=Path("/root/research/bridge-v0"),
            repository_root=Path("/root"),
            evidence_class=EvidenceClass.SYNTHETIC_PIPELINE_VALIDATION,
            provider_mode=ProviderMode.FAKE,
            kronos_mode=KronosMode.FAKE,
            dry_run=True,
            require_accelerator=False,
            training_symbols=("SPY", "QQQ"),
            unseen_symbols=("IWM",),
        )
        runner = CloudRunner(
            store=store,
            identity=identity,
            pipeline_config=config,
            provider=DeterministicFakeProvider(),
            kronos=DeterministicFakeKronosBackend(),
            training_backend=NumpyTrainingBackend(),
        )
        result = runner.execute(
            holder=f"modal-synthetic:{os.environ.get('MODAL_TASK_ID', run_id)}",
            confirm_open_test_partition=True,
            measurements=payload.get("measurements") or {},
        )

    return {
        "run_id": result.run_id,
        "final_state": result.final_state.value,
        "conclusion": result.conclusion.value if result.conclusion else None,
        "evidence_class": "synthetic_pipeline_validation",
        "label": "SYNTHETIC CLOUD PIPELINE VALIDATION - NOT EMPIRICAL EVIDENCE",
    }


# -------------------------------------------------------- deployment check


@app.function(image=image, secrets=secrets, timeout=CONTROL_TIMEOUT)
def verify_deployment() -> dict[str, Any]:
    """Fail deployment when the image and the committed locks disagree."""
    from pathlib import Path

    from openalpha_bridge.phase2.identity import verify_locked_hashes
    from openalpha_bridge.windowing import score_mask_sha256

    observed = verify_locked_hashes(Path("/root/research/bridge-v0"))
    return {
        "app": APP_NAME,
        "version": APP_VERSION,
        "gpu": GPU_CONFIG,
        "python": PYTHON_VERSION,
        "torch": TORCH_VERSION,
        "score_mask_sha256": score_mask_sha256(),
        "locked_hashes": observed,
        "verified_at": _now().isoformat(),
    }


# ------------------------------------------- Stage A official canary (GPU)


@app.function(
    image=image,
    gpu=GPU_CONFIG,
    volumes={CACHE_ROOT: cache_volume},
    secrets=secrets,
    timeout=30 * 60,
    retries=0,
)
def stage_a_official_canary(source_commit: str, run_id: str) -> dict[str, Any]:
    """One isolated Stage A compatibility check on the amended SPY window.

    Deliberately NOT routed through CloudRunner: that runner continues into the
    default-driven Stage B and Stage C. This function has no code path into
    training, checkpoint creation, checkpoint selection, metric or gate
    evaluation, or test opening. It retrieves only the single amended SPY
    training window and stops after extraction.

    Evidence lands under a development/compatibility prefix that cannot be
    confused with a complete real_phase2 run.
    """
    from pathlib import Path

    from openalpha_bridge.cloud.objectstore import put_json
    from openalpha_bridge.phase2.cache import FeatureCache
    from openalpha_bridge.phase2.canary import (
        CANARY_EVIDENCE_CLASS,
        run_stage_a_canary,
    )
    from openalpha_bridge.phase2.kronos import OfficialKronosBackend
    from openalpha_bridge.phase2.provider import YahooDailyProvider
    from openalpha_bridge.phase2.states import EvidenceClass

    _register_secrets()
    store = _build_store()
    cache_root = Path(CACHE_ROOT)
    os.environ.setdefault("HF_HOME", str(cache_root / "huggingface"))

    report = run_stage_a_canary(
        provider=YahooDailyProvider(stage="stage-a-canary"),
        backend=OfficialKronosBackend(
            stage="stage-a-canary",
            device="cuda",
            cache_dir=str(cache_root / "huggingface"),
            source_path=os.environ.get("OPENALPHA_KRONOS_SOURCE_PATH", KRONOS_SOURCE_ROOT),
        ),
        cache=FeatureCache(cache_root / "canary" / run_id),
        research_root=Path("/root/research/bridge-v0"),
        source_commit=source_commit,
        run_id=run_id,
    )

    # A prefix distinct from both the real and the synthetic run namespaces.
    key = f"openalpha-compatibility/stage-a-canary/{run_id}/canary_report.json"
    put_json(
        store,
        key,
        report.model_dump(mode="json"),
        schema_version=report.schema_version,
        run_id=run_id,
        experiment_hash=report.experiment_sha256,
        evidence_class=EvidenceClass.SYNTHETIC_PIPELINE_VALIDATION,
        immutable=True,
    )
    cache_volume.commit()

    payload = report.model_dump(mode="json")
    payload["artifact_key"] = key
    payload["evidence_class"] = CANARY_EVIDENCE_CLASS
    return payload
