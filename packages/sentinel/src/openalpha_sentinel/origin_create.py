from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
import time
from datetime import UTC, date, datetime
from pathlib import Path

from openalpha_research import EnvironmentMetadata, LocalArtifactStore

from .contracts import ForecastRequest
from .diagnostics import compute_phase_2_diagnostics
from .ensemble import EnsembleMember, assemble_ensemble
from .evidence import EvidenceContext, publish_creation, verify_creation
from .market_data import MarketDataRequest, default_yfinance_provider
from .providers.kronos import KronosSubprocessClient

EXPERIMENT_SHA256 = "587bc36ac1dd941c66d9e94c92d7b26a33faeb8e7e7a38883762f0cb54897950"
MODEL_REVISION = "f4e68697d9d5aed55cef5c96aabc3376bcad9f81"
TOKENIZER_REVISION = "26966d0035065a0cae0ebad7af8ece35bc1fb51c"
SOURCE_REVISION = "67b630e67f6a18c9e9be918d9b4337c960db1e9a"
CUTOFF = date(2024, 7, 5)
FORECAST_SESSIONS = (
    date(2024, 7, 8),
    date(2024, 7, 9),
    date(2024, 7, 10),
    date(2024, 7, 11),
    date(2024, 7, 12),
)
SEEDS = (1729, 2027, 7919)
CONTEXTS = (128, 256, 512)


def create_phase_2_origin(
    *,
    repository_root: Path,
    state_root: Path,
    inference_python: Path,
    source_path: Path,
    model_cache_path: Path,
) -> dict[str, object]:
    started = time.perf_counter()
    state_root.mkdir(parents=True, exist_ok=True)
    store = LocalArtifactStore(state_root / "artifacts")
    snapshot = default_yfinance_provider().fetch(
        MarketDataRequest(
            purpose="forecast_context",
            symbol="SPY",
            start_inclusive=date(2022, 6, 1),
            end_exclusive=date(2024, 7, 6),
            cutoff=CUTOFF,
            minimum_sessions=512,
        )
    )
    observed_source_revision = _git(source_path, "rev-parse", "HEAD").decode().strip()
    if observed_source_revision != SOURCE_REVISION:
        raise ValueError("Kronos source cache is not at the experiment-pinned revision")
    client = KronosSubprocessClient(
        python_executable=inference_python,
        worker_script=repository_root / "scripts" / "kronos_inference_worker.py",
        source_path=source_path,
        cache_path=model_cache_path,
        timeout_seconds=1_800.0,
    )
    probe_requests = (
        _request(snapshot.observations[-512:], 512, 1729),
        _request(snapshot.observations[-512:], 512, 1729),
        _request(snapshot.observations[-512:], 512, 2027),
    )
    probe_batch = client.forecast_many(probe_requests)
    _require_success(probe_batch.responses, "seed reproducibility probe")
    probe_paths = tuple(response.generated_paths[0] for response in probe_batch.responses)

    official_requests = tuple(
        _request(snapshot.observations[-context:], context, seed)
        for context in CONTEXTS
        for seed in SEEDS
    )
    official_batch = client.forecast_many(official_requests)
    _require_success(official_batch.responses, "official nine-path ensemble")
    members = tuple(
        EnsembleMember(
            context_length=request.context_length,
            sampling_seed=request.sampling_seed,
            path=response.generated_paths[0],
            inference_duration_ms=response.inference_duration_ms,
        )
        for request, response in zip(official_requests, official_batch.responses)
    )
    ensemble = assemble_ensemble(
        cutoff_close=snapshot.observations[-1].close,
        forecast_sessions=FORECAST_SESSIONS,
        members=members,
    )
    diagnostics_created_at = datetime.now(UTC)
    diagnostic_vector = compute_phase_2_diagnostics(
        symbol="SPY",
        cutoff=CUTOFF,
        context=snapshot.observations,
        ensemble=ensemble,
        created_at=diagnostics_created_at,
    )
    individual_paths = [
        {
            "context_length": member.context_length,
            "sampling_seed": member.sampling_seed,
            "request_id": response.request_id,
            "path_sha256": member.path.canonical_sha256,
            "predicted_log_return": math.log(
                member.path.observations[-1].close / ensemble.cutoff_close
            ),
            "inference_duration_ms": response.inference_duration_ms,
            "close_path": [row.close for row in member.path.observations],
            "path": [row.model_dump(mode="json") for row in member.path.observations],
            "output_quality_warnings": _path_quality_warnings(member.path.observations),
        }
        for member, response in zip(members, official_batch.responses)
    ]
    model_environment = {
        **official_batch.environment.model_dump(mode="json"),
        "model_repository": "NeoQuasar/Kronos-mini",
        "model_revision": MODEL_REVISION,
        "tokenizer_repository": "NeoQuasar/Kronos-Tokenizer-2k",
        "tokenizer_revision": TOKENIZER_REVISION,
        "source_repository": "shiyu-coder/Kronos",
        "source_revision": SOURCE_REVISION,
    }
    probe = {
        "a_sha256": probe_paths[0].canonical_sha256,
        "b_sha256": probe_paths[1].canonical_sha256,
        "c_sha256": probe_paths[2].canonical_sha256,
        "exact_seed_replay_supported": (
            probe_paths[0].canonical_sha256 == probe_paths[1].canonical_sha256
        ),
        "different_seed_output_differs": (
            probe_paths[0].canonical_sha256 != probe_paths[2].canonical_sha256
        ),
        "paths": [
            [row.model_dump(mode="json") for row in path.observations] for path in probe_paths
        ],
    }
    provider_provenance = snapshot.model_dump(mode="json", exclude={"observations"})
    forecast_payload: dict[str, object] = {
        "schema_version": "1.0",
        "claim_boundary": "DEVELOPMENT PROOF — NOT EMPIRICAL EVIDENCE",
        "symbol": "SPY",
        "cutoff": CUTOFF.isoformat(),
        "forecast_sessions": [session.isoformat() for session in FORECAST_SESSIONS],
        "cutoff_close": ensemble.cutoff_close,
        "provider": provider_provenance,
        "model_environment": model_environment,
        "reproducibility_probe": probe,
        "individual_paths": individual_paths,
        "canonical_close_path": list(ensemble.canonical_close_path),
        "canonical_predicted_log_return": ensemble.canonical_predicted_log_return,
        "baseline_close_path": list(ensemble.baseline_close_path),
        "baseline_predicted_log_return": ensemble.baseline_predicted_log_return,
        "context_summaries": [
            item.model_dump(mode="json") for item in ensemble.context_summaries
        ],
        "diagnostics": [item.model_dump(mode="json") for item in diagnostic_vector.values],
        "forecast_created_at": diagnostics_created_at.isoformat(),
        "total_runtime_seconds": time.perf_counter() - started,
        "request_retries": 1,
        "request_failures": [
            (
                "PRESEAL_ATTEMPT_1: three finite probe paths were rejected by an "
                "incorrect output-OHLC validator; no forecast was sealed and no outcome "
                "was accessed before the boundary was corrected"
            )
        ],
        "limitations": [
            "ONE_ORIGIN_ONLY",
            "UNOFFICIAL_YAHOO_INTERFACE",
            "NOT_POINT_IN_TIME_DATA",
            "RAW_RETURN_OMITS_DIVIDENDS",
            "NO_CROSS_PROVIDER_VERIFICATION",
            "OFFICIAL_KRONOS_PREDICTOR_DERIVES_INTERNAL_AMOUNT",
            "NO_SENTINEL_ACTION_OR_RELIABILITY_SCORE",
            "PRESEAL_IMPLEMENTATION_FAILURE_RETRIED",
        ],
    }
    experiment_path = repository_root / "research" / "sentinel-v0" / "experiment.yaml"
    experiment_bytes = experiment_path.read_bytes()
    if hashlib.sha256(experiment_bytes).hexdigest() != EXPERIMENT_SHA256:
        raise ValueError("experiment bytes do not match the amended Phase 2 hash")
    evidence_context = _evidence_context(
        repository_root=repository_root,
        dependency_lock_path=repository_root / "uv.lock",
        inference_environment=model_environment,
    )
    creation = publish_creation(
        store=store,
        context=evidence_context,
        canonical_spec={
            "schema_version": "1.0",
            "experiment_yaml_sha256": EXPERIMENT_SHA256,
            "experiment_yaml_utf8": experiment_bytes.decode("utf-8"),
        },
        data_snapshot={
            "schema_version": "1.0",
            **provider_provenance,
        },
        data_quality={
            "schema_version": "1.0",
            "normalized_input_sha256": snapshot.normalized_input_sha256,
            "row_count": snapshot.row_count,
            "first_session": snapshot.first_session.isoformat(),
            "last_session": snapshot.last_session.isoformat(),
            "quality_summary": list(snapshot.quality_summary),
            "corporate_action_warnings": [
                warning.model_dump(mode="json")
                for warning in snapshot.corporate_action_warnings
            ],
        },
        forecast=forecast_payload,
        diagnostics={
            "schema_version": "1.0",
            **diagnostic_vector.model_dump(mode="json"),
        },
        created_at=diagnostics_created_at,
    )
    if not verify_creation(store, creation):
        raise ValueError("forecast creation seal did not verify")
    control = {
        "schema_version": "sentinel-phase2-control-v0",
        "state_root": str(state_root.resolve()),
        "evidence_context": evidence_context.model_dump(mode="json"),
        "creation": creation.model_dump(mode="json"),
    }
    _atomic_json_write(state_root / "creation.json", control)
    return {
        "forecast_id": creation.seal_ref.sha256,
        "forecast_ref_sha256": creation.forecast_ref.sha256,
        "diagnostics_ref_sha256": creation.diagnostics_ref.sha256,
        "data_snapshot_sha256": snapshot.normalized_input_sha256,
        "row_count": snapshot.row_count,
        "probe": probe,
        "canonical_predicted_log_return": ensemble.canonical_predicted_log_return,
        "creation_verified": True,
        "control_path": str((state_root / "creation.json").resolve()),
    }


def _request(
    observations: tuple,
    context_length: int,
    seed: int,
) -> ForecastRequest:
    return ForecastRequest(
        model_repository="NeoQuasar/Kronos-mini",
        model_revision=MODEL_REVISION,
        tokenizer_repository="NeoQuasar/Kronos-Tokenizer-2k",
        tokenizer_revision=TOKENIZER_REVISION,
        source_revision=SOURCE_REVISION,
        symbol="SPY",
        cutoff=CUTOFF,
        forecast_sessions=FORECAST_SESSIONS,
        observations=observations,
        context_length=context_length,
        forecast_horizon=5,
        sampling_seed=seed,
        temperature=1.0,
        top_p=0.9,
        sample_count=1,
        data_snapshot_sha256=_observations_sha256(observations),
        experiment_sha256=EXPERIMENT_SHA256,
        created_at=datetime.now(UTC),
    )


def _observations_sha256(observations: tuple) -> str:
    payload = [row.model_dump(mode="json") for row in observations]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _require_success(responses: tuple, label: str) -> None:
    failures = [
        response.failure.model_dump(mode="json")
        for response in responses
        if response.failure is not None
    ]
    if failures:
        raise RuntimeError(f"{label} failed: {failures}")


def _path_quality_warnings(observations: tuple) -> list[str]:
    warnings: list[str] = []
    if any(
        row.high < max(row.open, row.close, row.low)
        or row.low > min(row.open, row.close, row.high)
        for row in observations
    ):
        warnings.append("MODEL_OUTPUT_OHLC_INCONSISTENCY")
    if any(row.volume < 0 for row in observations):
        warnings.append("MODEL_OUTPUT_NEGATIVE_VOLUME")
    return warnings


def _evidence_context(
    *,
    repository_root: Path,
    dependency_lock_path: Path,
    inference_environment: dict[str, object],
) -> EvidenceContext:
    commit = _git(repository_root, "rev-parse", "HEAD").decode().strip()
    status = _git(repository_root, "status", "--porcelain")
    dirty = bool(status.strip())
    diff_sha = _working_tree_sha256(repository_root) if dirty else None
    lock_sha = hashlib.sha256(dependency_lock_path.read_bytes()).hexdigest()
    return EvidenceContext(
        run_id="sentinel_spy_20240705",
        attempt_id="attempt_01",
        code_commit=commit,
        code_dirty=dirty,
        diff_sha256=diff_sha,
        dependency_lock_sha256=lock_sha,
        environment=EnvironmentMetadata(
            os_name=platform.system(),
            os_version=platform.version(),
            architecture=platform.machine(),
            python_version=platform.python_version(),
            node_version=None,
            dependency_lock_sha256=lock_sha,
            container_image=None,
            hardware=(
                f"repository_device=cpu; inference_device={inference_environment.get('device')}"
            ),
        ),
    )


def _working_tree_sha256(repository_root: Path) -> str:
    digest = hashlib.sha256()
    digest.update(_git(repository_root, "diff", "--binary", "HEAD"))
    untracked = _git(
        repository_root,
        "ls-files",
        "--others",
        "--exclude-standard",
        "-z",
    ).split(b"\0")
    for raw_path in sorted(path for path in untracked if path):
        relative = raw_path.decode("utf-8")
        digest.update(relative.encode("utf-8"))
        digest.update((repository_root / relative).read_bytes())
    return digest.hexdigest()


def _git(repository_root: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(repository_root), *arguments],
        check=True,
        capture_output=True,
    )
    return completed.stdout


def _atomic_json_write(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)
