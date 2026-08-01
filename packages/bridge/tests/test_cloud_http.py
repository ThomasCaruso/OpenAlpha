"""Control-plane HTTP tests against the real FastAPI application.

These exercise the routes exactly as the deployed Modal ASGI app serves them:
documented paths, real Authorization headers, and typed failure mapping. No
cloud SDK and no Modal are involved.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from fastapi.testclient import TestClient
from openalpha_bridge.cloud.http import build_control_api, status_for
from openalpha_bridge.cloud.objectstore import InMemoryObjectStore
from openalpha_bridge.cloud.redaction import register_secret
from openalpha_bridge.cloud.service import Phase2ControlService, ServiceConfig
from openalpha_bridge.errors import BridgeFailure, BridgeTransformError, FailureCategory
from openalpha_bridge.phase2.identity import EXPERIMENT_SHA256
from openalpha_bridge.phase2.states import EvidenceClass

TOKEN = "test-bearer-token-abcdef123456"
COMMIT = "a" * 40
RUNS = "/v1/bridge/phase2/runs"


class _Clock:
    def __init__(self) -> None:
        self._now = datetime(2026, 8, 1, 9, 0, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        self._now += timedelta(seconds=1)
        return self._now


class _Backend:
    name = "fake"

    def __init__(self) -> None:
        self.spawns: list[str] = []

    @property
    def image_digest(self) -> str:
        return "sha256:testdigest"

    @property
    def app_version(self) -> str:
        return "1.0.0"

    def spawn_real_run(self, run_id: str, payload: dict[str, Any]) -> str:
        self.spawns.append(run_id)
        return f"fc-real-{run_id}"

    def spawn_synthetic_run(self, run_id: str, payload: dict[str, Any]) -> str:
        self.spawns.append(run_id)
        return f"fc-syn-{run_id}"

    def execution_status(self, cloud_execution_id: str) -> str:
        return "running"

    def cancel(self, cloud_execution_id: str) -> None:
        return None


@pytest.fixture
def store() -> InMemoryObjectStore:
    return InMemoryObjectStore()


@pytest.fixture
def client(store: InMemoryObjectStore) -> TestClient:
    backend = _Backend()
    config = ServiceConfig(
        base_url="https://openalpha.example",
        object_store_namespace="openalpha-artifacts",
        dependency_lock_sha256="b" * 64,
        kronos_revision="26966d0035065a0cae0ebad7af8ece35bc1fb51c",
        provider_identity="yahoo_finance/yfinance==1.5.2",
    )
    service = Phase2ControlService(
        store=store, backend=backend, config=config, clock=_Clock()
    )
    app = build_control_api(
        service_factory=lambda: service, token_provider=lambda: TOKEN
    )
    return TestClient(app, raise_server_exceptions=False)


def _auth() -> dict[str, str]:
    return {"Authorization": f"Bearer {TOKEN}"}


def _payload(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "experiment_hash": EXPERIMENT_SHA256,
        "source_commit": COMMIT,
        "operator": "tester",
        "execution_mode": "synthetic",
    }
    body.update(overrides)
    return body


# ------------------------------------------------------------ authentication


def test_missing_bearer_token_is_rejected(client: TestClient) -> None:
    response = client.post(RUNS, json=_payload())
    assert response.status_code == 401
    assert response.json()["code"] == "UNAUTHORIZED"


def test_invalid_bearer_token_is_rejected(client: TestClient) -> None:
    response = client.post(RUNS, json=_payload(), headers={"Authorization": "Bearer wrong-token"})
    assert response.status_code == 401


def test_non_bearer_scheme_is_rejected(client: TestClient) -> None:
    response = client.post(RUNS, json=_payload(), headers={"Authorization": f"Basic {TOKEN}"})
    assert response.status_code == 401


def test_valid_token_succeeds(client: TestClient) -> None:
    response = client.post(RUNS, json=_payload(), headers=_auth())
    assert response.status_code == 200
    body = response.json()
    assert body["run_id"].startswith("syn_")
    assert body["state"] == "CREATED"
    assert body["evidence_class"] == EvidenceClass.SYNTHETIC_PIPELINE_VALIDATION.value


def test_every_route_requires_authentication(client: TestClient) -> None:
    created = client.post(RUNS, json=_payload(), headers=_auth()).json()
    run_id = created["run_id"]
    for method, path in (
        ("get", f"{RUNS}/{run_id}"),
        ("get", f"{RUNS}/{run_id}/artifacts"),
        ("get", f"{RUNS}/{run_id}/logs"),
        ("post", f"{RUNS}/{run_id}/resume"),
        ("post", f"{RUNS}/{run_id}/cancel"),
    ):
        call = getattr(client, method)
        response = call(path) if method == "get" else call(path, json={"operator": "x"})
        assert response.status_code == 401, f"{method.upper()} {path} was not protected"


# --------------------------------------------------------------- validation


def test_unknown_experiment_hash_is_rejected(client: TestClient) -> None:
    response = client.post(RUNS, json=_payload(experiment_hash="0" * 64), headers=_auth())
    assert response.status_code == 400
    assert response.json()["code"] == "UNKNOWN_EXPERIMENT_HASH"


def test_real_mode_without_confirmation_is_rejected(client: TestClient) -> None:
    response = client.post(RUNS, json=_payload(execution_mode="real"), headers=_auth())
    assert response.status_code == 400
    assert response.json()["code"] == "REAL_EVIDENCE_NOT_CONFIRMED"


def test_unknown_field_is_rejected(client: TestClient) -> None:
    """The request model forbids extras, so no unapproved knob can be smuggled."""
    response = client.post(
        RUNS, json=_payload(architecture="Linear(999,999)"), headers=_auth()
    )
    assert response.status_code in (400, 422)


def test_unknown_run_returns_404(client: TestClient) -> None:
    response = client.get(f"{RUNS}/syn_doesnotexist", headers=_auth())
    assert response.status_code == 404
    assert response.json()["code"] == "RUN_NOT_FOUND"


# -------------------------------------------------------------- idempotency


def test_identical_idempotency_requests_return_the_same_run(client: TestClient) -> None:
    body = _payload(idempotency_key="stable-key-http-1")
    first = client.post(RUNS, json=body, headers=_auth()).json()
    second = client.post(RUNS, json=body, headers=_auth()).json()
    assert first["run_id"] == second["run_id"]
    assert second["idempotent_replay"] is True


def test_conflicting_idempotency_key_returns_409(client: TestClient) -> None:
    client.post(RUNS, json=_payload(idempotency_key="stable-key-http-2"), headers=_auth())
    response = client.post(
        RUNS,
        json=_payload(idempotency_key="stable-key-http-2", operator="someone-else"),
        headers=_auth(),
    )
    assert response.status_code == 409
    assert response.json()["code"] == "IDEMPOTENCY_KEY_CONFLICT"


# ---------------------------------------------------------------- lifecycle


def test_status_reports_a_sealed_test_partition(client: TestClient) -> None:
    run_id = client.post(RUNS, json=_payload(), headers=_auth()).json()["run_id"]
    body = client.get(f"{RUNS}/{run_id}", headers=_auth()).json()
    assert body["test_sealed"] is True
    assert body["test_partition_opened"] is False
    assert body["final_conclusion"] is None


def test_resume_of_a_running_run_returns_409(client: TestClient) -> None:
    run_id = client.post(RUNS, json=_payload(), headers=_auth()).json()["run_id"]
    response = client.post(
        f"{RUNS}/{run_id}/resume", json={"operator": "tester"}, headers=_auth()
    )
    assert response.status_code == 409
    assert response.json()["code"] == "RUN_NOT_RESUMABLE"


def test_cancel_then_status_reports_blocked(client: TestClient) -> None:
    run_id = client.post(RUNS, json=_payload(), headers=_auth()).json()["run_id"]
    cancelled = client.post(
        f"{RUNS}/{run_id}/cancel",
        json={"operator": "tester", "reason": "not needed"},
        headers=_auth(),
    )
    assert cancelled.status_code == 200
    assert cancelled.json()["state"] == "BLOCKED"


def test_artifacts_and_logs_are_served(client: TestClient) -> None:
    run_id = client.post(RUNS, json=_payload(), headers=_auth()).json()["run_id"]

    artifacts = client.get(f"{RUNS}/{run_id}/artifacts", headers=_auth())
    assert artifacts.status_code == 200
    assert artifacts.json()["run_id"] == run_id

    logs = client.get(f"{RUNS}/{run_id}/logs", headers=_auth())
    assert logs.status_code == 200
    assert logs.json()["entries"]


def test_logs_redact_secrets_over_http(client: TestClient) -> None:
    run_id = client.post(RUNS, json=_payload(), headers=_auth()).json()["run_id"]
    register_secret("http-leak-secret-abcdef")
    client.post(
        f"{RUNS}/{run_id}/cancel",
        json={"operator": "tester", "reason": "died with http-leak-secret-abcdef"},
        headers=_auth(),
    )
    logs = client.get(f"{RUNS}/{run_id}/logs", headers=_auth())
    assert "http-leak-secret-abcdef" not in logs.text


def test_no_response_echoes_the_bearer_token(client: TestClient) -> None:
    run_id = client.post(RUNS, json=_payload(), headers=_auth()).json()["run_id"]
    for path in (f"{RUNS}/{run_id}", f"{RUNS}/{run_id}/logs", f"{RUNS}/{run_id}/artifacts"):
        assert TOKEN not in client.get(path, headers=_auth()).text


# ------------------------------------------------------------- attack surface


def test_no_arbitrary_command_or_experiment_path_route_exists(client: TestClient) -> None:
    for path in ("/exec", "/shell", "/v1/bridge/phase2/exec", "/v1/bridge/phase2/experiments"):
        assert client.post(path, json={"cmd": "ls"}, headers=_auth()).status_code == 404


def test_interactive_docs_and_schema_are_disabled(client: TestClient) -> None:
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert client.get(path).status_code == 404


def test_status_mapping_falls_back_to_400() -> None:
    error = BridgeTransformError(
        BridgeFailure(
            category=FailureCategory.INVALID_INPUT,
            code="SOME_UNMAPPED_CODE",
            message="unmapped",
        )
    )
    assert status_for(error) == 400
