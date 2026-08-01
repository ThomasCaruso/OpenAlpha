"""Secret redaction for every outward-facing surface.

Applied to logs, journal entries, status payloads, artifacts, and exception
messages. Secrets are registered once at process start from the environment and
are then scrubbed by value, so an accidental interpolation cannot leak.
"""

from __future__ import annotations

import re
from typing import Any

__all__ = [
    "REDACTED",
    "SECRET_ENVIRONMENT_NAMES",
    "SecretRedactor",
    "redact",
    "register_secret",
]

REDACTED = "[REDACTED]"
_MINIMUM_SECRET_LENGTH = 8

#: Environment variables whose values are always treated as secrets.
SECRET_ENVIRONMENT_NAMES: tuple[str, ...] = (
    "OPENALPHA_API_TOKEN",
    "OPENALPHA_WEBHOOK_SIGNING_SECRET",
    "ALPACA_API_KEY_ID",
    "ALPACA_API_SECRET_KEY",
    "HUGGING_FACE_HUB_TOKEN",
    "R2_ACCESS_KEY_ID",
    "R2_SECRET_ACCESS_KEY",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
)

#: Patterns redacted even when the exact value was never registered.
_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)\b(bearer)\s+[A-Za-z0-9._\-]{8,}"),
    re.compile(r"(?i)\b(authorization\s*[:=]\s*)\S+"),
    re.compile(r"\bhf_[A-Za-z0-9]{16,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9]{16,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bghp_[A-Za-z0-9]{20,}\b"),
    re.compile(r"(?i)\b(api[_-]?key|api[_-]?secret|access[_-]?key|secret[_-]?key|token|password)"
               r"(\s*[:=]\s*)([^\s,;&\"']{6,})"),
    # Query-string credentials in a provider URL.
    re.compile(r"(?i)([?&](?:key|token|secret|apikey|api_key)=)[^&\s]+"),
)


class SecretRedactor:
    """Scrubs known secret values and credential-shaped patterns."""

    def __init__(self) -> None:
        self._values: set[str] = set()

    def register(self, value: str | None) -> None:
        if value and len(value) >= _MINIMUM_SECRET_LENGTH:
            self._values.add(value)

    def register_environment(self, environment: dict[str, str]) -> None:
        for name in SECRET_ENVIRONMENT_NAMES:
            self.register(environment.get(name))

    def clear(self) -> None:
        self._values.clear()

    def scrub_text(self, text: str) -> str:
        result = text
        # Longest first, so a secret containing another is fully removed.
        for secret in sorted(self._values, key=len, reverse=True):
            result = result.replace(secret, REDACTED)
        for pattern in _PATTERNS:
            result = self._apply(pattern, result)
        return result

    @staticmethod
    def _apply(pattern: re.Pattern[str], text: str) -> str:
        def replace(match: re.Match[str]) -> str:
            groups = match.groups()
            if len(groups) >= 3:
                return f"{groups[0]}{groups[1]}{REDACTED}"
            if len(groups) >= 1 and groups[0]:
                return f"{groups[0]} {REDACTED}" if groups[0].lower() == "bearer" else (
                    f"{groups[0]}{REDACTED}"
                )
            return REDACTED

        return pattern.sub(replace, text)

    def scrub(self, value: Any) -> Any:
        """Recursively scrub strings inside mappings, sequences, and scalars."""
        if isinstance(value, str):
            return self.scrub_text(value)
        if isinstance(value, dict):
            return {key: self.scrub(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            scrubbed = [self.scrub(item) for item in value]
            return type(value)(scrubbed) if isinstance(value, tuple) else scrubbed
        return value


_DEFAULT = SecretRedactor()


def register_secret(value: str | None) -> None:
    _DEFAULT.register(value)


def redact(value: Any) -> Any:
    return _DEFAULT.scrub(value)


def default_redactor() -> SecretRedactor:
    return _DEFAULT
