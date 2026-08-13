# pyright: reportMissingImports=false, reportAttributeAccessIssue=false
# pyright: reportFunctionMemberAccess=false
"""Research-only Modal shell for completed Kronos studies."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import modal

APP_NAME = "openalpha-kronos-research"
PYTHON_VERSION = "3.13"
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
TORCH_WHEEL_URL = "https://download.pytorch.org/whl/cu126/torch-2.13.0%2Bcu126-cp313-cp313-manylinux_2_28_x86_64.whl"
TORCH_WHEEL_SHA256 = "4198c8d7478ab47ad2569309387d88b21fb553a1cf8ab06260fbd5a6ab9b9712"
TORCH_WHEEL_SPECIFIER = f"{TORCH_WHEEL_URL}#sha256={TORCH_WHEEL_SHA256}"


def _pin(name: str) -> str:
    return f"{name}=={CANARY_RUNTIME_PINS[name]}"


KRONOS_SOURCE_REPOSITORY = "https://github.com/shiyu-coder/Kronos"
KRONOS_SOURCE_REVISION = "67b630e67f6a18c9e9be918d9b4337c960db1e9a"
KRONOS_SOURCE_ROOT = "/opt/kronos"
KRONOS_MINI_REPOSITORY = "NeoQuasar/Kronos-mini"
KRONOS_MINI_REVISION = "f4e68697d9d5aed55cef5c96aabc3376bcad9f81"
KRONOS_BASE_REPOSITORY = "NeoQuasar/Kronos-base"
KRONOS_BASE_REVISION = "2b554741eca47781b64468546e77fef3e85130e6"
KRONOS_BASE_TOKENIZER_REPOSITORY = "NeoQuasar/Kronos-Tokenizer-base"
KRONOS_BASE_TOKENIZER_REVISION = "0e0117387f39004a9016484a186a908917e22426"
KRONOS_SOURCE_FILES: dict[str, str] = {
    "model/kronos.py": "638a56e035856c600c9848b368be087cb706a61603a0790124968c95b8c69f3a",
    "model/module.py": "a07edbadc0e96804c8158c021bbc6063bb7cc43b34d7fc470d5c8ff2005a409f",
}


def _repo_root() -> Path | None:
    here = Path(__file__).resolve()
    if len(here.parents) < 3:
        return None
    candidate = here.parents[2]
    return candidate if (candidate / "packages" / "kronos-research").is_dir() else None


LOCAL_PACKAGES: tuple[tuple[str, str], ...] = (
    ("packages/research-core/src/openalpha_research", "/root/openalpha_research"),
    ("packages/kronos-research/src/openalpha_kronos", "/root/openalpha_kronos"),
)
_IGNORE = ["__pycache__", "**/__pycache__", "*.pyc", "*.pyo"]


def _load_sibling(name: str) -> Any:
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
        .pip_install(TORCH_WHEEL_SPECIFIER)
        .pip_install(
            _pin("numpy"),
            _pin("pydantic"),
            _pin("yfinance"),
            _pin("huggingface-hub"),
            _pin("safetensors"),
            _pin("pandas"),
            _pin("tqdm"),
            _pin("einops"),
            "boto3>=1.35,<2",
        )
        .run_commands(
            f"git clone --no-checkout {KRONOS_SOURCE_REPOSITORY} {KRONOS_SOURCE_ROOT}",
            f"cd {KRONOS_SOURCE_ROOT} && git checkout --detach {KRONOS_SOURCE_REVISION}",
            f'cd {KRONOS_SOURCE_ROOT} && test "$(git rev-parse HEAD)" = "{KRONOS_SOURCE_REVISION}" || (echo "KRONOS_SOURCE_REVISION_MISMATCH" && exit 1)',
            f'cd {KRONOS_SOURCE_ROOT} && git status --porcelain | tee /tmp/kronos_dirty && test ! -s /tmp/kronos_dirty || (echo "KRONOS_WORKTREE_DIRTY" && exit 1)',
            *(
                f'''cd {KRONOS_SOURCE_ROOT} && sed -e 's/\\r$//' -e 's/$/\\r/' {relative} | sha256sum | cut -d' ' -f1 > /tmp/observed && test "$(cat /tmp/observed)" = "{digest}" || (echo "KRONOS_SOURCE_HASH_MISMATCH {relative}" && echo "observed $(cat /tmp/observed) expected {digest}" && exit 1)'''
                for relative, digest in KRONOS_SOURCE_FILES.items()
            ),
        )
        .env({"OPENALPHA_KRONOS_SOURCE_PATH": KRONOS_SOURCE_ROOT})
        .env({"PYTHONPATH": "/root"})
    )
    root = _repo_root()
    if root is None:
        return built
    binding = _load_sibling("deployed_commit")
    built = built.env(
        {binding.COMMIT_ENVIRONMENT_VARIABLE: binding.bind_for_image(root, environment=os.environ)}
    )
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
cache_volume = modal.Volume.from_name("openalpha-bridge-phase2-cache", create_if_missing=True)
CACHE_ROOT = "/cache"
base_cache_volume = modal.Volume.from_name("openalpha-kronos-base-cache", create_if_missing=True)
BASE_CACHE_ROOT = "/base-cache"
SECRET_NAMES = ("openalpha-storage", "openalpha-huggingface")
secrets = [modal.Secret.from_name(name) for name in SECRET_NAMES]
GPU_CONFIG = "T4"
MAX_GPU_SECONDS = 24 * 60 * 60
CONTROL_TIMEOUT = 60


def _now() -> datetime:
    return datetime.now(UTC)


def _build_research_store() -> Any:
    from openalpha_research.objectstore import S3CompatibleObjectStore

    return S3CompatibleObjectStore(
        bucket=os.environ["OPENALPHA_ARTIFACT_BUCKET"],
        endpoint_url=os.environ.get("OPENALPHA_S3_ENDPOINT_URL"),
        region_name=os.environ.get("OPENALPHA_S3_REGION", "auto"),
    )


def _register_secrets() -> None:
    from openalpha_research.redaction import default_redactor

    default_redactor().register_environment(dict(os.environ))


DEPLOYED_COMMIT_VARIABLE = "OPENALPHA_DEPLOYED_COMMIT"


def _require_deployed_commit() -> str:
    import re

    value = os.environ.get(DEPLOYED_COMMIT_VARIABLE, "")
    if not value:
        raise RuntimeError(
            f"DEPLOYED_COMMIT_MISSING: {DEPLOYED_COMMIT_VARIABLE} is not set in this image. Redeploy from a clean worktree so the commit is bound at build time; it must never be supplied at call time."
        )
    if not re.fullmatch("[0-9a-f]{40}", value):
        raise RuntimeError(
            f"DEPLOYED_COMMIT_MALFORMED: {DEPLOYED_COMMIT_VARIABLE} must be exactly forty lowercase hex characters, got {value!r}"
        )
    return value


@app.function(
    image=image,
    gpu=GPU_CONFIG,
    volumes={CACHE_ROOT: cache_volume},
    secrets=secrets,
    timeout=60 * 60,
    retries=0,
)
def run_mini_structural_validity(source_commit: str, run_id: str) -> dict[str, Any]:
    from pathlib import Path

    from openalpha_kronos.model.assets import TOKENIZER_SPEC
    from openalpha_kronos.model.official import official_runtime
    from openalpha_kronos.studies.structural_validity.mini.spec import (
        OFFICIAL_SNAPSHOT_ALLOW_PATTERNS,
    )
    from openalpha_kronos.studies.structural_validity.mini.worker import run_diagnostic_worker
    from openalpha_research.providers import YahooDailyProvider

    _register_secrets()
    cache_root = Path(CACHE_ROOT)
    os.environ.setdefault("HF_HOME", str(cache_root / "huggingface"))
    source_root = Path(os.environ.get("OPENALPHA_KRONOS_SOURCE_PATH", KRONOS_SOURCE_ROOT))

    def resolve_runtime():
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
        store=_build_research_store(),
        resolve_runtime=resolve_runtime,
        provider_factory=YahooDailyProvider,
        research_root=Path("/root/research/bridge-v0"),
        source_commit=source_commit,
        deployed_commit=_require_deployed_commit(),
        run_id=run_id,
    )
    cache_volume.commit()
    return result.model_dump(mode="json")


@app.function(
    image=image,
    gpu=GPU_CONFIG,
    volumes={CACHE_ROOT: cache_volume},
    secrets=secrets,
    timeout=30 * 60,
    retries=0,
)
def verify_mini_runtime() -> dict[str, Any]:
    from pathlib import Path

    from openalpha_kronos.model.assets import TOKENIZER_SPEC
    from openalpha_kronos.model.official import official_runtime
    from openalpha_kronos.studies.structural_validity.mini.runtime_probe import (
        RuntimeEnvironment,
        run_frozen_inference_runtime_probe,
    )
    from openalpha_kronos.studies.structural_validity.mini.spec import (
        OFFICIAL_SNAPSHOT_ALLOW_PATTERNS,
    )

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


def _download_base_pair(cache_root: Path) -> tuple[str, str]:
    from huggingface_hub import snapshot_download
    from openalpha_kronos.studies.structural_validity.mini.spec import (
        OFFICIAL_SNAPSHOT_ALLOW_PATTERNS,
    )

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
    return (tokenizer_dir, model_dir)


@app.function(
    image=image,
    gpu=GPU_CONFIG,
    volumes={BASE_CACHE_ROOT: base_cache_volume},
    secrets=secrets,
    timeout=30 * 60,
    retries=0,
)
def verify_base_runtime() -> dict[str, Any]:
    from pathlib import Path

    from openalpha_kronos.model.official import official_runtime
    from openalpha_kronos.studies.structural_validity.base.runtime_probe import (
        run_base_runtime_probe,
    )
    from openalpha_kronos.studies.structural_validity.base.spec import (
        KRONOS_BASE_SPEC,
        KRONOS_BASE_TOKENIZER_SPEC,
    )
    from openalpha_kronos.studies.structural_validity.mini.runtime_probe import RuntimeEnvironment

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
def run_base_structural_validity(source_commit: str, run_id: str) -> dict[str, Any]:
    from pathlib import Path

    from openalpha_kronos.model.official import official_runtime
    from openalpha_kronos.studies.structural_validity.base.spec import (
        KRONOS_BASE_SPEC,
        KRONOS_BASE_TOKENIZER_SPEC,
    )
    from openalpha_kronos.studies.structural_validity.base.worker import run_base_study_worker
    from openalpha_research.providers import YahooDailyProvider

    _register_secrets()
    cache_root = Path(BASE_CACHE_ROOT)
    os.environ.setdefault("HF_HOME", str(cache_root / "huggingface"))
    source_root = Path(os.environ.get("OPENALPHA_KRONOS_SOURCE_PATH", KRONOS_SOURCE_ROOT))

    def resolve_runtime():
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
        store=_build_research_store(),
        resolve_runtime=resolve_runtime,
        provider_factory=YahooDailyProvider,
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
    retries=0,
)
def inventory_base_artifacts() -> dict[str, Any]:
    from openalpha_kronos.studies.structural_validity.base.cache_inventory import (
        build_base_cache_inventory,
    )

    return build_base_cache_inventory(
        reload=base_cache_volume.reload,
        mount=BASE_CACHE_ROOT,
        volume_name="openalpha-kronos-base-cache",
        deployed_commit=_require_deployed_commit(),
        inspected_at=_now(),
    )


@app.function(
    image=image,
    gpu=GPU_CONFIG,
    volumes={BASE_CACHE_ROOT: base_cache_volume},
    secrets=secrets,
    timeout=30 * 60,
    retries=0,
)
def verify_zero_shot_runtime() -> dict[str, Any]:
    from pathlib import Path

    from openalpha_kronos.model.official import official_runtime
    from openalpha_kronos.studies.structural_validity.mini.runtime_probe import RuntimeEnvironment
    from openalpha_kronos.studies.zero_shot.runtime_probe import run_zero_shot_runtime_probe
    from openalpha_kronos.studies.zero_shot.spec import (
        ZERO_SHOT_MODEL_SPEC,
        ZERO_SHOT_TOKENIZER_SPEC,
    )

    _register_secrets()
    cache_root = Path(BASE_CACHE_ROOT)
    os.environ.setdefault("HF_HOME", str(cache_root / "huggingface"))
    source_root = Path(os.environ.get("OPENALPHA_KRONOS_SOURCE_PATH", KRONOS_SOURCE_ROOT))
    import numpy
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "ZERO_SHOT_RUNTIME_PROBE_NO_CUDA: the probe exists to exercise the deployed GPU path"
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
        tokenizer_spec=ZERO_SHOT_TOKENIZER_SPEC,
        model_spec=ZERO_SHOT_MODEL_SPEC,
        device="cuda",
    ) as runtime:
        result = run_zero_shot_runtime_probe(
            codec=runtime.codec,
            model=runtime.model,
            assets=runtime.assets,
            parameter_digest=runtime.parameter_digest,
            environment=environment,
            run_id="zero_shot_runtime_probe",
            deployed_commit=_require_deployed_commit(),
            now=_now(),
        )
    base_cache_volume.commit()
    return result.model_dump(mode="json")


@app.function(
    image=image,
    gpu=GPU_CONFIG,
    volumes={BASE_CACHE_ROOT: base_cache_volume},
    secrets=secrets,
    timeout=2 * 60 * 60,
    retries=0,
)
def run_zero_shot_benchmark(source_commit: str, run_id: str) -> dict[str, Any]:
    from pathlib import Path

    from openalpha_kronos.model.official import official_runtime
    from openalpha_kronos.studies.zero_shot.spec import (
        ZERO_SHOT_MODEL_SPEC,
        ZERO_SHOT_TOKENIZER_SPEC,
    )
    from openalpha_kronos.studies.zero_shot.worker import run_zero_shot_benchmark_worker
    from openalpha_research.providers import YahooDailyProvider

    _register_secrets()
    cache_root = Path(BASE_CACHE_ROOT)
    os.environ.setdefault("HF_HOME", str(cache_root / "huggingface"))
    source_root = Path(os.environ.get("OPENALPHA_KRONOS_SOURCE_PATH", KRONOS_SOURCE_ROOT))

    def resolve_runtime():
        tokenizer_dir, model_dir = _download_base_pair(cache_root)
        return official_runtime(
            source_root=source_root,
            tokenizer_directory=tokenizer_dir,
            model_directory=model_dir,
            tokenizer_spec=ZERO_SHOT_TOKENIZER_SPEC,
            model_spec=ZERO_SHOT_MODEL_SPEC,
            device="cuda",
        )

    result = run_zero_shot_benchmark_worker(
        store=_build_research_store(),
        resolve_runtime=resolve_runtime,
        provider_factory=YahooDailyProvider,
        research_root=Path("/root/research/bridge-v0"),
        source_commit=source_commit,
        deployed_commit=_require_deployed_commit(),
        run_id=run_id,
    )
    base_cache_volume.commit()
    return result.model_dump(mode="json")


@app.function(image=image, secrets=secrets, timeout=CONTROL_TIMEOUT, retries=0)
def inventory_zero_shot_artifacts() -> dict[str, Any]:
    from openalpha_kronos.studies.zero_shot.inventory import inventory_zero_shot_artifacts as build

    _register_secrets()
    result = build(
        _build_research_store(), deployed_commit=_require_deployed_commit(), inspected_at=_now()
    )
    return result.model_dump(mode="json")
