"""HTTP binding for the Phase 2 control service.

Kept out of the Modal application so it can be exercised by tests without a
cloud SDK. FastAPI is imported lazily inside the builder, so importing this
module pulls in nothing heavy.

A single ASGI application rather than per-function endpoints: Modal's
``fastapi_endpoint`` binds every declared parameter as a body or query field, so
a ``headers`` parameter never receives HTTP headers and path parameters such as
``{run_id}`` cannot be expressed at all.
"""

# NOTE: deliberately no `from __future__ import annotations`.
#
# FastAPI resolves parameter annotations at decoration time. Because FastAPI is
# imported lazily inside build_control_api, `Request` is a closure local rather
# than a module global, so a deferred string annotation cannot be resolved and
# FastAPI silently demotes `request: Request` to a query parameter. Eager
# annotation evaluation binds the real class. Python 3.13 evaluates `X | None`
# and `dict[str, int]` natively, so nothing else here needs the future import.

from collections.abc import Callable
from typing import Any

from ..errors import BridgeTransformError
from .auth import TokenAuthenticator, log_admin_action
from .models import CancelRunRequest, CreateRunRequest, ResumeRunRequest
from .redaction import redact
from .service import Phase2ControlService

__all__ = ["STATUS_BY_CODE", "build_control_api"]

#: Typed failure codes mapped to HTTP status. Anything unlisted falls back to
#: the status carried on the failure, then to 400.
STATUS_BY_CODE: dict[str, int] = {
    "UNAUTHORIZED": 401,
    "UNAPPROVED_SOURCE_COMMIT": 403,
    "RUN_NOT_FOUND": 404,
    "RUN_NOT_RESUMABLE": 409,
    "RUN_ALREADY_FINALIZED": 409,
    "RUN_LEASE_HELD": 409,
    "IDEMPOTENCY_KEY_CONFLICT": 409,
    "TEST_PARTITION_ALREADY_OPENED": 409,
    "CONCURRENT_JOURNAL_WRITE": 409,
    "RATE_LIMIT_EXCEEDED": 429,
}


def status_for(error: BridgeTransformError) -> int:
    failure = error.failures[0]
    status = STATUS_BY_CODE.get(failure.code)
    if status is None and isinstance(failure.observed_value, (int, float)):
        status = int(failure.observed_value)
    return status or 400


def build_control_api(
    *,
    service_factory: Callable[[], Phase2ControlService],
    token_provider: Callable[[], str | None],
    version: str = "1.0.0",
) -> Any:
    """Build the FastAPI application exposing the documented control routes."""
    from fastapi import Depends, FastAPI, Header, Request
    from fastapi.responses import JSONResponse
    from pydantic import ValidationError

    web_app = FastAPI(
        title="OpenAlpha Bridge Phase 2 Control API",
        version=version,
        # No interactive docs or schema on an authenticated control plane.
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    @web_app.exception_handler(BridgeTransformError)
    async def _typed_failure(_: Request, error: BridgeTransformError) -> JSONResponse:
        failure = error.failures[0]
        return JSONResponse(
            status_code=status_for(error),
            content={
                "error": failure.category.value,
                "code": failure.code,
                "message": redact(failure.message),
            },
        )

    @web_app.exception_handler(ValidationError)
    async def _invalid_request(_: Request, error: ValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "error": "INVALID_INPUT",
                "code": "INVALID_REQUEST_BODY",
                "message": redact(str(error)),
            },
        )

    def authenticate(authorization: str | None = Header(default=None)) -> Any:
        return TokenAuthenticator(token_provider()).authenticate(authorization)

    # The request models are declared strict, which in Python mode rejects the
    # plain strings JSON carries for enums. Validating the raw body keeps
    # strictness while still parsing enum members and timestamps.
    @web_app.post("/v1/bridge/phase2/runs")
    async def create_run(request: Request, context: Any = Depends(authenticate)) -> dict:
        parsed = CreateRunRequest.model_validate_json(await request.body())
        log_admin_action("create_run", principal=context.principal, operator=parsed.operator)
        return service_factory().create_run(parsed).model_dump(mode="json")

    @web_app.get("/v1/bridge/phase2/runs/{run_id}")
    def run_status(run_id: str, context: Any = Depends(authenticate)) -> dict:
        return service_factory().get_run(run_id).model_dump(mode="json")

    @web_app.post("/v1/bridge/phase2/runs/{run_id}/resume")
    async def resume_run(
        run_id: str, request: Request, context: Any = Depends(authenticate)
    ) -> dict:
        parsed = ResumeRunRequest.model_validate_json(await request.body())
        log_admin_action("resume_run", principal=context.principal, run_id=run_id)
        return service_factory().resume_run(run_id, parsed).model_dump(mode="json")

    @web_app.post("/v1/bridge/phase2/runs/{run_id}/cancel")
    async def cancel_run(
        run_id: str, request: Request, context: Any = Depends(authenticate)
    ) -> dict:
        parsed = CancelRunRequest.model_validate_json(await request.body())
        log_admin_action("cancel_run", principal=context.principal, run_id=run_id)
        return service_factory().cancel_run(run_id, parsed).model_dump(mode="json")

    @web_app.get("/v1/bridge/phase2/runs/{run_id}/artifacts")
    def run_artifacts(run_id: str, context: Any = Depends(authenticate)) -> dict:
        return service_factory().list_artifacts(run_id, signed=True).model_dump(mode="json")

    @web_app.get("/v1/bridge/phase2/runs/{run_id}/logs")
    def run_logs(run_id: str, context: Any = Depends(authenticate)) -> dict:
        return service_factory().get_logs(run_id).model_dump(mode="json")

    return web_app
