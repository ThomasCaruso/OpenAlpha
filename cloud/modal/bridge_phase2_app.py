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

# Exact runtime pins live in openalpha_bridge.phase2.runtime_pins, resolved from
# uv.lock. They are duplicated here as literals only because `modal deploy`
# imports this file before the workspace packages are importable; a packaging
# test asserts the two never drift apart.
CANARY_RUNTIME_PINS: dict[str, str] = {
    "torch": "2.13.0",
    "numpy": "2.5.1",
    "pandas": "3.0.5",
    "tqdm": "4.70.0",
    "einops": "0.8.2",
    "huggingface-hub": "0.36.2",
    "safetensors": "0.8.0",
    "yfinance": "1.5.2",
    "pydantic": "2.13.4",
}
TORCH_VERSION = CANARY_RUNTIME_PINS["torch"]

# Torch is installed as an exact artifact, not as a requirement resolved
# against an index. `torch==2.13.0` with extra_index_url=cu124 was wrong: the
# cu124 index stops at 2.6.0, so the requirement fell through to the default
# index and installed a different build of 2.13.0 than the one named. A direct
# URL cannot be satisfied by any other artifact, and pip verifies the
# `#sha256=` fragment before installing, so both the fallback and the
# unverified-download problems disappear together.
#
# cp313 = CPython 3.13. manylinux_2_28_x86_64 = Linux x86_64, glibc >= 2.28;
# Debian bookworm ships 2.36. cu126 bundles the CUDA 12.6 runtime, which runs
# under minor version compatibility on any CUDA 12.x driver.
# Resolved from https://download.pytorch.org/whl/cu126/torch/
TORCH_WHEEL_URL = (
    "https://download.pytorch.org/whl/cu126/"
    "torch-2.13.0%2Bcu126-cp313-cp313-manylinux_2_28_x86_64.whl"
)
TORCH_WHEEL_SHA256 = "4198c8d7478ab47ad2569309387d88b21fb553a1cf8ab06260fbd5a6ab9b9712"
TORCH_WHEEL_SPECIFIER = f"{TORCH_WHEEL_URL}#sha256={TORCH_WHEEL_SHA256}"


def _pin(name: str) -> str:
    return f"{name}=={CANARY_RUNTIME_PINS[name]}"


# The pinned official Kronos source. Mirrors experiment.yaml official_source;
# openalpha_bridge.phase2.kronos.SOURCE_SPEC is the authority and a deployment
# test asserts the two agree.
KRONOS_SOURCE_REPOSITORY = "https://github.com/shiyu-coder/Kronos"
KRONOS_SOURCE_REVISION = "67b630e67f6a18c9e9be918d9b4337c960db1e9a"
KRONOS_SOURCE_ROOT = "/opt/kronos"
# The pinned Kronos-mini forecasting model, for the frozen inference
# diagnostic. openalpha_bridge.diagnostic.spec.KRONOS_MINI_SPEC is the
# authority; a packaging test asserts these agree.
KRONOS_MINI_REPOSITORY = "NeoQuasar/Kronos-mini"
KRONOS_MINI_REVISION = "f4e68697d9d5aed55cef5c96aabc3376bcad9f81"
# The pinned Kronos-base pair, for the separate base replication study.
# openalpha_bridge.base_study.spec is the authority; a packaging test asserts
# these agree. Kronos-base is released paired with Kronos-Tokenizer-base, and a
# crossed pair is refused at runtime rather than silently measured.
KRONOS_BASE_REPOSITORY = "NeoQuasar/Kronos-base"
KRONOS_BASE_REVISION = "2b554741eca47781b64468546e77fef3e85130e6"
KRONOS_BASE_TOKENIZER_REPOSITORY = "NeoQuasar/Kronos-Tokenizer-base"
KRONOS_BASE_TOKENIZER_REVISION = "0e0117387f39004a9016484a186a908917e22426"

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


def _load_sibling(name: str) -> Any:
    """Import a module sitting next to this file, by path.

    Only ever called on the deploying workstation, where this file lives in
    cloud/modal/. In the container the module is flattened to /root and has no
    siblings, but nothing there needs one.
    """
    import importlib.util

    path = Path(__file__).resolve().parent / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"openalpha_modal_{name}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load deployment helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_image() -> Any:
    built = (
        modal.Image.debian_slim(python_version=PYTHON_VERSION)
        .apt_install("git")
        # Exact artifact, hash-verified by pip. No index resolution, so no
        # fallback to a different build is possible.
        .pip_install(TORCH_WHEEL_SPECIFIER)
        .pip_install(
            # Every canary-relevant package is pinned exactly, from uv.lock.
            _pin("numpy"),
            _pin("pydantic"),
            _pin("yfinance"),
            _pin("huggingface-hub"),
            _pin("safetensors"),
            # Runtime imports of the pinned official Kronos source.
            _pin("pandas"),
            _pin("tqdm"),
            _pin("einops"),
            # Control plane; not part of the official numerical path.
            "boto3>=1.35,<2",
            "fastapi>=0.115,<1",
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
            # The sealed digests are over CRLF-normalized content, so the file
            # is normalized before hashing. Hashing the checked-out bytes
            # directly fails for model/kronos.py, which is committed with LF:
            # the sealed value is the digest of its CRLF form. Normalizing to
            # LF first keeps this correct for model/module.py, which is
            # committed with CRLF and must not be doubled.
            *(
                f"cd {KRONOS_SOURCE_ROOT} && "
                f"sed -e 's/\\r$//' -e 's/$/\\r/' {relative} | sha256sum | cut -d' ' -f1 "
                f"> /tmp/observed && "
                f'test "$(cat /tmp/observed)" = "{digest}" '
                f'|| (echo "KRONOS_SOURCE_HASH_MISMATCH {relative}" && '
                f'echo "observed $(cat /tmp/observed) expected {digest}" && exit 1)'
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
        # OPENALPHA_DEPLOYED_COMMIT was baked in at deploy time and is already
        # in the environment; it is never recomputed here.
        return built

    # Bind the deploying repository state into the image. Raises, and so fails
    # the deployment, when HEAD cannot be resolved or the worktree is dirty.
    # Loaded by path rather than by name: `modal deploy` does not guarantee this
    # file's directory is on sys.path.
    binding = _load_sibling("deployed_commit")
    built = built.env(
        {binding.COMMIT_ENVIRONMENT_VARIABLE: binding.bind_for_image(root, environment=os.environ)}
    )

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

# The base study gets its own volume rather than a subdirectory of the mini
# one. Two reasons: the 409 MB base weights should not grow a volume whose
# retention is governed by the completed mini study, and a later maintenance
# session is expected to delete the mini snapshots -- which is a far safer
# operation when the two studies cannot share a directory tree by accident.
# It holds Hugging Face assets and regenerable execution caches only; terminal
# scientific evidence is written to immutable object storage, never here.
base_cache_volume = modal.Volume.from_name("openalpha-kronos-base-cache", create_if_missing=True)
BASE_CACHE_ROOT = "/base-cache"

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

    identity_key = f"openalpha/bridge-phase2/runs/{run_id}/identity/run_identity.json"
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

    identity_key = f"openalpha-synthetic/bridge-phase2/runs/{run_id}/identity/run_identity.json"
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
    """Fail deployment when the image and the committed locks disagree.

    Two independent families of document are checked, and both are returned
    separately. ``locked_hashes`` covers the sealed experiment and its three
    amendments. ``diagnostic_specification_hashes`` covers the two frozen
    inference diagnostic specifications, which are separately hashed and are
    not part of the sealed chain.

    The diagnostic specifications used to be absent here. That made this check
    look like it covered the image's research directory when it covered only
    part of it, so a drifted diagnostic specification would have reached the
    worker with the deployment check reporting success.
    """
    from pathlib import Path

    from openalpha_bridge.diagnostic.spec import verify_diagnostic_specifications
    from openalpha_bridge.phase2.identity import verify_locked_hashes
    from openalpha_bridge.windowing import score_mask_sha256

    research_root = Path("/root/research/bridge-v0")
    # Both verifiers raise on any mismatch or missing file, so reaching the
    # return statement is itself the evidence. Nothing here is hard-coded.
    observed = verify_locked_hashes(research_root)
    diagnostic_hashes = verify_diagnostic_specifications(research_root)
    return {
        "app": APP_NAME,
        "version": APP_VERSION,
        "gpu": GPU_CONFIG,
        "python": PYTHON_VERSION,
        "torch": TORCH_VERSION,
        "score_mask_sha256": score_mask_sha256(),
        "locked_hashes": observed,
        "diagnostic_specification_hashes": diagnostic_hashes,
        # Raises here if the binding is missing or malformed, so a broken
        # deployment is caught by the check rather than by the canary.
        "deployed_commit": _require_deployed_commit(),
        "verified_at": _now().isoformat(),
    }


# -------------------------------------------------- deployed commit binding

#: Duplicated from cloud/modal/deployed_commit.py, which the container does not
#: have. A test asserts the two agree.
DEPLOYED_COMMIT_VARIABLE = "OPENALPHA_DEPLOYED_COMMIT"


def _require_deployed_commit() -> str:
    """The commit baked into this image, or refuse to run.

    There is no default, no empty-string fallback, and no caller-supplied
    override. Every one of those would let a worker produce evidence it cannot
    attribute to specific code.
    """
    import re

    value = os.environ.get(DEPLOYED_COMMIT_VARIABLE, "")
    if not value:
        raise RuntimeError(
            f"DEPLOYED_COMMIT_MISSING: {DEPLOYED_COMMIT_VARIABLE} is not set in this "
            "image. Redeploy from a clean worktree so the commit is bound at build "
            "time; it must never be supplied at call time."
        )
    if not re.fullmatch(r"[0-9a-f]{40}", value):
        raise RuntimeError(
            f"DEPLOYED_COMMIT_MALFORMED: {DEPLOYED_COMMIT_VARIABLE} must be exactly "
            f"forty lowercase hex characters, got {value!r}"
        )
    return value


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

    from openalpha_bridge.phase2.canary_worker import run_canary_worker
    from openalpha_bridge.phase2.kronos import OfficialKronosBackend
    from openalpha_bridge.phase2.provider import YahooDailyProvider

    _register_secrets()
    cache_root = Path(CACHE_ROOT)
    os.environ.setdefault("HF_HOME", str(cache_root / "huggingface"))

    # This function is a shell. Everything it does lives in
    # openalpha_bridge.phase2.canary_worker, which is exercised directly by
    # tests with fakes; a worker whose only tests assert on its source text is
    # a worker nobody has run.
    result = run_canary_worker(
        store=_build_store(),
        provider=YahooDailyProvider(stage="stage-a-canary"),
        backend=OfficialKronosBackend(
            stage="stage-a-canary",
            device="cuda",
            cache_dir=str(cache_root / "huggingface"),
            source_path=os.environ.get("OPENALPHA_KRONOS_SOURCE_PATH", KRONOS_SOURCE_ROOT),
        ),
        feature_cache_root=cache_root / "canary",
        asset_cache_root=cache_root / "huggingface",
        research_root=Path("/root/research/bridge-v0"),
        source_commit=source_commit,
        # The image says what code it was built from. Not negotiable, not
        # defaulted, and never supplied by the caller.
        deployed_commit=_require_deployed_commit(),
        run_id=run_id,
    )
    cache_volume.commit()
    return result.model_dump(mode="json")


# ------------------------------- frozen inference diagnostic (GPU, isolated)


@app.function(
    image=image,
    gpu=GPU_CONFIG,
    volumes={CACHE_ROOT: cache_volume},
    secrets=secrets,
    timeout=60 * 60,
    retries=0,
)
def frozen_inference_diagnostic(source_commit: str, run_id: str) -> dict[str, Any]:
    """The frozen inference diagnostic, and nothing else.

    Its own function rather than a branch inside the canary worker, so the two
    cannot share a code path, an artifact key, or a failure mode. There is no
    import of and no call into CloudRunner, training, an optimizer, checkpoint
    code, Stage B, Stage C, test opening, or held-out evaluation. Execution
    stops after the artifact is written.

    This is a shell. Everything it does lives in
    openalpha_bridge.diagnostic.worker, which tests exercise with doubles.
    """
    from pathlib import Path

    from openalpha_bridge.diagnostic.official_backend import official_runtime
    from openalpha_bridge.diagnostic.spec import OFFICIAL_SNAPSHOT_ALLOW_PATTERNS
    from openalpha_bridge.diagnostic.worker import run_diagnostic_worker
    from openalpha_bridge.phase2.kronos import TOKENIZER_SPEC
    from openalpha_bridge.phase2.provider import YahooDailyProvider

    _register_secrets()
    cache_root = Path(CACHE_ROOT)
    os.environ.setdefault("HF_HOME", str(cache_root / "huggingface"))
    source_root = Path(os.environ.get("OPENALPHA_KRONOS_SOURCE_PATH", KRONOS_SOURCE_ROOT))

    def resolve_runtime():
        """The one official runtime context, entered only when there is work.

        Returns the context manager rather than an entered runtime, so the
        worker owns the lifetime and the tokenizer and model stay imported for
        as long as they are used.
        """
        from huggingface_hub import snapshot_download

        tokenizer_dir = snapshot_download(
            repo_id=TOKENIZER_SPEC.repository,
            revision=TOKENIZER_SPEC.revision,
            cache_dir=str(cache_root / "huggingface"),
            allow_patterns=list(OFFICIAL_SNAPSHOT_ALLOW_PATTERNS),
        )
        model_dir = snapshot_download(
            repo_id=KRONOS_MINI_REPOSITORY,
            revision=KRONOS_MINI_REVISION,
            cache_dir=str(cache_root / "huggingface"),
            allow_patterns=list(OFFICIAL_SNAPSHOT_ALLOW_PATTERNS),
        )
        return official_runtime(
            source_root=source_root,
            tokenizer_directory=tokenizer_dir,
            model_directory=model_dir,
            tokenizer_spec=TOKENIZER_SPEC,
            device="cuda",
        )

    result = run_diagnostic_worker(
        store=_build_store(),
        resolve_runtime=resolve_runtime,
        provider_factory=lambda: YahooDailyProvider(stage="frozen-inference-diagnostic"),
        research_root=Path("/root/research/bridge-v0"),
        source_commit=source_commit,
        deployed_commit=_require_deployed_commit(),
        run_id=run_id,
    )
    cache_volume.commit()
    return result.model_dump(mode="json")


# --------------------------------- frozen inference runtime probe (GPU only)


@app.function(
    image=image,
    gpu=GPU_CONFIG,
    volumes={CACHE_ROOT: cache_volume},
    secrets=secrets,
    timeout=30 * 60,
    retries=0,
)
def verify_frozen_inference_runtime() -> dict[str, Any]:
    """Does the deployed image actually execute the official frozen path?

    A compatibility probe, not a diagnostic and not empirical evidence. It
    retrieves no market data, computes no forecast metric, writes no scientific
    conclusion, and authorizes nothing, including the diagnostic itself.

    Its result carries its own schema and its own outcome code, and nothing
    here can write under the frozen diagnostic's key.
    """
    from pathlib import Path

    from openalpha_bridge.diagnostic.official_backend import official_runtime
    from openalpha_bridge.diagnostic.runtime_probe import (
        RuntimeEnvironment,
        run_frozen_inference_runtime_probe,
    )
    from openalpha_bridge.diagnostic.spec import OFFICIAL_SNAPSHOT_ALLOW_PATTERNS
    from openalpha_bridge.phase2.kronos import TOKENIZER_SPEC

    _register_secrets()
    cache_root = Path(CACHE_ROOT)
    os.environ.setdefault("HF_HOME", str(cache_root / "huggingface"))
    source_root = Path(os.environ.get("OPENALPHA_KRONOS_SOURCE_PATH", KRONOS_SOURCE_ROOT))

    import numpy
    import torch
    from huggingface_hub import snapshot_download

    if not torch.cuda.is_available():
        raise RuntimeError(
            "RUNTIME_PROBE_NO_CUDA: the probe exists to exercise the deployed GPU path"
        )

    environment = RuntimeEnvironment(
        torch_version=torch.__version__,
        torch_cuda_version=torch.version.cuda,
        cuda_available=True,
        device_name=torch.cuda.get_device_name(0),
        device_capability=".".join(str(part) for part in torch.cuda.get_device_capability(0)),
        numpy_version=numpy.__version__,
    )

    # Exactly the files the diagnostic fetches, and every one hash-verified
    # before a weight is read.
    tokenizer_dir = snapshot_download(
        repo_id=TOKENIZER_SPEC.repository,
        revision=TOKENIZER_SPEC.revision,
        cache_dir=str(cache_root / "huggingface"),
        allow_patterns=list(OFFICIAL_SNAPSHOT_ALLOW_PATTERNS),
    )
    model_dir = snapshot_download(
        repo_id=KRONOS_MINI_REPOSITORY,
        revision=KRONOS_MINI_REVISION,
        cache_dir=str(cache_root / "huggingface"),
        allow_patterns=list(OFFICIAL_SNAPSHOT_ALLOW_PATTERNS),
    )

    # The same shared context the diagnostic uses, held open while the probe
    # runs its inference, then exited.
    with official_runtime(
        source_root=source_root,
        tokenizer_directory=tokenizer_dir,
        model_directory=model_dir,
        tokenizer_spec=TOKENIZER_SPEC,
        device="cuda",
    ) as runtime:
        result = run_frozen_inference_runtime_probe(
            codec=runtime.codec,
            model=runtime.model,
            assets=runtime.assets,
            parameter_digest=runtime.parameter_digest,
            environment=environment,
            run_id="runtime_probe",
            deployed_commit=_require_deployed_commit(),
        )

    cache_volume.commit()
    return result.model_dump(mode="json")


# ======================================================================
# Kronos-base replication study
#
# A separate study with its own preregistration, its own artifact namespace,
# its own run-identifier space and its own remote volume. Nothing below reads
# or writes anything belonging to the completed Kronos-mini study, and no
# function here is named as though it were a mini function.
# ======================================================================


@app.function(image=image, secrets=secrets, timeout=CONTROL_TIMEOUT)
def verify_base_deployment() -> dict[str, Any]:
    """Fail deployment when the image and the base preregistration disagree.

    Deliberately separate from ``verify_deployment``. That function is the mini
    study's gate and its meaning is fixed by what has already been run under
    it; conflating the two would let a base drift be reported as a mini failure
    or the reverse. Both are returned here so one call can show that the mini
    chain is still intact alongside the base document, without either check
    depending on the other.
    """
    from pathlib import Path

    from openalpha_bridge.base_study.spec import (
        BASE_ARTIFACT_ROOT,
        BASE_EXPERIMENT_ID,
        BASE_FAILURE_SCHEMA_VERSION,
        BASE_PROBE_SCHEMA_VERSION,
        BASE_SUCCESS_SCHEMA_VERSION,
        KRONOS_BASE_SPEC,
        KRONOS_BASE_TOKENIZER_SPEC,
        MAXIMUM_CONTEXT,
        prove_context_budget,
        verify_base_specification,
    )
    from openalpha_bridge.diagnostic.spec import (
        CONTEXT_CANDLES,
        TARGET_CANDLES,
        verify_diagnostic_specifications,
    )
    from openalpha_bridge.phase2.identity import verify_locked_hashes

    research_root = Path("/root/research/bridge-v0")
    base_hashes = verify_base_specification(research_root)
    # The mini chain must still verify: this study replicates it and a drifted
    # mini document would make the comparison meaningless.
    mini_sealed = verify_locked_hashes(research_root)
    mini_diagnostic = verify_diagnostic_specifications(research_root)

    if KRONOS_BASE_SPEC.repository != KRONOS_BASE_REPOSITORY:
        raise RuntimeError("BASE_MODEL_REPOSITORY_DRIFT")
    if KRONOS_BASE_SPEC.revision != KRONOS_BASE_REVISION:
        raise RuntimeError("BASE_MODEL_REVISION_DRIFT")
    if KRONOS_BASE_TOKENIZER_SPEC.repository != KRONOS_BASE_TOKENIZER_REPOSITORY:
        raise RuntimeError("BASE_TOKENIZER_REPOSITORY_DRIFT")
    if KRONOS_BASE_TOKENIZER_SPEC.revision != KRONOS_BASE_TOKENIZER_REVISION:
        raise RuntimeError("BASE_TOKENIZER_REVISION_DRIFT")

    return {
        "app": APP_NAME,
        "version": APP_VERSION,
        "gpu": GPU_CONFIG,
        "python": PYTHON_VERSION,
        "torch": TORCH_VERSION,
        "experiment_id": BASE_EXPERIMENT_ID,
        "base_specification_hashes": base_hashes,
        "mini_locked_hashes": mini_sealed,
        "mini_diagnostic_specification_hashes": mini_diagnostic,
        "artifact_namespace": BASE_ARTIFACT_ROOT,
        "run_id_pattern": "^base_[0-9a-f]{8,32}$",
        "success_schema": BASE_SUCCESS_SCHEMA_VERSION,
        "failure_schema": BASE_FAILURE_SCHEMA_VERSION,
        "runtime_probe_schema": BASE_PROBE_SCHEMA_VERSION,
        "model": {
            "repository": KRONOS_BASE_SPEC.repository,
            "revision": KRONOS_BASE_SPEC.revision,
            "config_sha256": KRONOS_BASE_SPEC.config_sha256,
            "weights_sha256": KRONOS_BASE_SPEC.weights_sha256,
            "weights_size_bytes": KRONOS_BASE_SPEC.weights_size_bytes,
            "model_dimension": KRONOS_BASE_SPEC.model_dimension,
            "layers": KRONOS_BASE_SPEC.layers,
            "attention_heads": KRONOS_BASE_SPEC.attention_heads,
        },
        "tokenizer": {
            "repository": KRONOS_BASE_TOKENIZER_SPEC.repository,
            "revision": KRONOS_BASE_TOKENIZER_SPEC.revision,
            "config_sha256": KRONOS_BASE_TOKENIZER_SPEC.config_sha256,
            "weights_sha256": KRONOS_BASE_TOKENIZER_SPEC.weights_sha256,
        },
        "maximum_context": MAXIMUM_CONTEXT,
        "context_budget": prove_context_budget(
            context_candles=CONTEXT_CANDLES, target_candles=TARGET_CANDLES
        ),
        "remote_cache_volume": "openalpha-kronos-base-cache",
        "deployed_commit": _require_deployed_commit(),
        "verified_at": _now().isoformat(),
    }


def _download_base_pair(cache_root: Path) -> tuple[str, str]:
    """Fetch only config.json and model.safetensors, into the base volume.

    The 409 MB base weights exist in exactly one place: this remote volume.
    They are never copied into the repository, the image, Git LFS, a local
    temporary directory, a test fixture, or a terminal artifact.
    """
    from huggingface_hub import snapshot_download
    from openalpha_bridge.diagnostic.spec import OFFICIAL_SNAPSHOT_ALLOW_PATTERNS

    tokenizer_dir = snapshot_download(
        repo_id=KRONOS_BASE_TOKENIZER_REPOSITORY,
        revision=KRONOS_BASE_TOKENIZER_REVISION,
        cache_dir=str(cache_root / "huggingface"),
        allow_patterns=list(OFFICIAL_SNAPSHOT_ALLOW_PATTERNS),
    )
    model_dir = snapshot_download(
        repo_id=KRONOS_BASE_REPOSITORY,
        revision=KRONOS_BASE_REVISION,
        cache_dir=str(cache_root / "huggingface"),
        allow_patterns=list(OFFICIAL_SNAPSHOT_ALLOW_PATTERNS),
    )
    return tokenizer_dir, model_dir


@app.function(
    image=image,
    gpu=GPU_CONFIG,
    volumes={BASE_CACHE_ROOT: base_cache_volume},
    secrets=secrets,
    timeout=30 * 60,
    retries=0,
)
def verify_base_frozen_inference_runtime() -> dict[str, Any]:
    """Does the deployed image execute the official Kronos-base path?

    A compatibility probe, not a replication and not empirical evidence. It
    retrieves no market data, computes no forecast metric, writes no scientific
    conclusion and authorizes nothing, including the base diagnostic itself.
    """
    from pathlib import Path

    from openalpha_bridge.base_study.runtime_probe import run_base_runtime_probe
    from openalpha_bridge.base_study.spec import KRONOS_BASE_SPEC, KRONOS_BASE_TOKENIZER_SPEC
    from openalpha_bridge.diagnostic.official_backend import official_runtime
    from openalpha_bridge.diagnostic.runtime_probe import RuntimeEnvironment

    _register_secrets()
    cache_root = Path(BASE_CACHE_ROOT)
    os.environ.setdefault("HF_HOME", str(cache_root / "huggingface"))
    source_root = Path(os.environ.get("OPENALPHA_KRONOS_SOURCE_PATH", KRONOS_SOURCE_ROOT))

    import numpy
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "BASE_RUNTIME_PROBE_NO_CUDA: the probe exists to exercise the deployed GPU path"
        )

    environment = RuntimeEnvironment(
        torch_version=torch.__version__,
        torch_cuda_version=torch.version.cuda,
        cuda_available=True,
        device_name=torch.cuda.get_device_name(0),
        device_capability=".".join(str(part) for part in torch.cuda.get_device_capability(0)),
        numpy_version=numpy.__version__,
    )

    tokenizer_dir, model_dir = _download_base_pair(cache_root)

    with official_runtime(
        source_root=source_root,
        tokenizer_directory=tokenizer_dir,
        model_directory=model_dir,
        tokenizer_spec=KRONOS_BASE_TOKENIZER_SPEC,
        model_spec=KRONOS_BASE_SPEC,
        device="cuda",
    ) as runtime:
        result = run_base_runtime_probe(
            codec=runtime.codec,
            model=runtime.model,
            assets=runtime.assets,
            parameter_digest=runtime.parameter_digest,
            environment=environment,
            run_id="base_runtime_probe",
            deployed_commit=_require_deployed_commit(),
        )

    base_cache_volume.commit()
    return result.model_dump(mode="json")


@app.function(
    image=image,
    gpu=GPU_CONFIG,
    volumes={BASE_CACHE_ROOT: base_cache_volume},
    secrets=secrets,
    timeout=60 * 60,
    retries=0,
)
def kronos_base_frozen_inference_diagnostic(source_commit: str, run_id: str) -> dict[str, Any]:
    """The Kronos-base replication, and nothing else.

    Its own function rather than a flag on the mini diagnostic, so the two can
    never share a code path, an artifact key, a run identifier or a failure
    mode. There is no import of and no call into the mini worker, CloudRunner,
    training, an optimizer, checkpoint code, Stage B, Stage C, test opening, or
    held-out evaluation. Execution stops after the artifact is written.

    This is a shell. Everything it does lives in
    openalpha_bridge.base_study.worker, which tests exercise with doubles.
    """
    from pathlib import Path

    from openalpha_bridge.base_study.spec import KRONOS_BASE_SPEC, KRONOS_BASE_TOKENIZER_SPEC
    from openalpha_bridge.base_study.worker import run_base_study_worker
    from openalpha_bridge.diagnostic.official_backend import official_runtime
    from openalpha_bridge.phase2.provider import YahooDailyProvider

    _register_secrets()
    cache_root = Path(BASE_CACHE_ROOT)
    os.environ.setdefault("HF_HOME", str(cache_root / "huggingface"))
    source_root = Path(os.environ.get("OPENALPHA_KRONOS_SOURCE_PATH", KRONOS_SOURCE_ROOT))

    def resolve_runtime():
        """The one official runtime context, entered only when there is work."""
        tokenizer_dir, model_dir = _download_base_pair(cache_root)
        return official_runtime(
            source_root=source_root,
            tokenizer_directory=tokenizer_dir,
            model_directory=model_dir,
            tokenizer_spec=KRONOS_BASE_TOKENIZER_SPEC,
            model_spec=KRONOS_BASE_SPEC,
            device="cuda",
        )

    result = run_base_study_worker(
        store=_build_store(),
        resolve_runtime=resolve_runtime,
        provider_factory=lambda: YahooDailyProvider(stage="kronos-base-diagnostic"),
        research_root=Path("/root/research/bridge-v0"),
        source_commit=source_commit,
        deployed_commit=_require_deployed_commit(),
        run_id=run_id,
    )
    base_cache_volume.commit()
    return result.model_dump(mode="json")


@app.function(
    image=image,
    volumes={BASE_CACHE_ROOT: base_cache_volume},
    secrets=secrets,
    timeout=CONTROL_TIMEOUT,
)
def inventory_base_remote_cache() -> dict[str, Any]:
    """Report what the base cache volume holds. Read-only, always.

    Deliberately has no delete mode and is not reachable through the control
    API. Removing a cached snapshot is a maintenance action that has to be
    taken deliberately, with the artifacts and revisions it depends on proved
    first -- not something an ordinary endpoint can be talked into.

    Invoke with:
        modal run cloud/modal/bridge_phase2_app.py::inventory_base_remote_cache
    """
    from pathlib import Path

    root = Path(BASE_CACHE_ROOT) / "huggingface"
    repositories: list[dict[str, Any]] = []
    total = 0

    if root.is_dir():
        for repo_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            if not repo_dir.name.startswith("models--"):
                continue
            snapshots_root = repo_dir / "snapshots"
            snapshots: list[dict[str, Any]] = []
            repo_bytes = 0
            for item in repo_dir.rglob("*"):
                if item.is_file() and not item.is_symlink():
                    repo_bytes += item.stat().st_size
            if snapshots_root.is_dir():
                for snapshot in sorted(p for p in snapshots_root.iterdir() if p.is_dir()):
                    resolved_bytes = 0
                    files: list[str] = []
                    for item in sorted(snapshot.rglob("*")):
                        if item.is_file() or item.is_symlink():
                            files.append(item.relative_to(snapshot).as_posix())
                            try:
                                resolved_bytes += item.resolve().stat().st_size
                            except OSError:
                                pass
                    snapshots.append(
                        {
                            "revision": snapshot.name,
                            "path": str(snapshot),
                            "files": files,
                            "resolved_bytes": resolved_bytes,
                        }
                    )
            total += repo_bytes
            repositories.append(
                {
                    "cache_directory": repo_dir.name,
                    "repository": repo_dir.name.removeprefix("models--").replace("--", "/", 1),
                    "path": str(repo_dir),
                    "bytes_on_volume": repo_bytes,
                    "snapshots": snapshots,
                }
            )

    return {
        "volume": "openalpha-kronos-base-cache",
        "mount": BASE_CACHE_ROOT,
        "root": str(root),
        "exists": root.is_dir(),
        "repositories": repositories,
        "total_bytes": total,
        "read_only": True,
        "deletion_supported": False,
        "deployed_commit": _require_deployed_commit(),
        "inspected_at": _now().isoformat(),
    }
