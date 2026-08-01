"""Bearer-token authentication for the control plane.

The public must not be able to start GPU jobs. Every mutating endpoint requires
a bearer token compared in constant time. Administrative actions are logged
without their credentials.
"""

from __future__ import annotations

import hmac
import logging
from dataclasses import dataclass

from ..errors import BridgeFailure, BridgeTransformError, FailureCategory
from .redaction import redact, register_secret

__all__ = ["AuthContext", "TokenAuthenticator", "log_admin_action"]

_LOGGER = logging.getLogger("openalpha.bridge.cloud.auth")


@dataclass(frozen=True, slots=True)
class AuthContext:
    authenticated: bool
    principal: str


def _unauthorized(message: str) -> BridgeTransformError:
    return BridgeTransformError(
        BridgeFailure(
            category=FailureCategory.INVALID_CONFIGURATION,
            code="UNAUTHORIZED",
            observed_value=401,
            message=message,
        )
    )


class TokenAuthenticator:
    """Constant-time bearer-token check against a Modal-managed secret."""

    def __init__(self, expected_token: str | None, *, principal: str = "service") -> None:
        if expected_token:
            register_secret(expected_token)
        self._expected = expected_token
        self._principal = principal

    def authenticate(self, authorization_header: str | None) -> AuthContext:
        if not self._expected:
            raise _unauthorized(
                "control API is not configured with an authentication token"
            )
        if not authorization_header:
            raise _unauthorized("missing Authorization header")

        scheme, _, presented = authorization_header.partition(" ")
        if scheme.lower() != "bearer" or not presented:
            raise _unauthorized("Authorization header must use the Bearer scheme")
        if not hmac.compare_digest(presented.strip(), self._expected):
            raise _unauthorized("invalid bearer token")
        return AuthContext(authenticated=True, principal=self._principal)


def log_admin_action(action: str, *, principal: str, **fields: object) -> None:
    """Log an administrative action with every value redacted."""
    _LOGGER.info(
        "admin action=%s principal=%s fields=%s",
        redact(action),
        redact(principal),
        redact(dict(fields)),
    )
