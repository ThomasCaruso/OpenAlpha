from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

__all__ = [
    "REDACTED",
    "SECRET_ENVIRONMENT_NAMES",
    "SecretRedactor",
    "default_redactor",
    "redact",
    "register_secret",
]

REDACTED = "[REDACTED]"
_MINIMUM_SECRET_LENGTH = 8

SECRET_ENVIRONMENT_NAMES: tuple[str, ...] = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "HUGGING_FACE_HUB_TOKEN",
    "GITHUB_TOKEN",
)

_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)\b(bearer)\s+[A-Za-z0-9._-]{8,}"),
    re.compile(r"(?i)\b(authorization\s*[:=]\s*)\S+"),
    re.compile(r"\bhf_[A-Za-z0-9]{16,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9]{16,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bghp_[A-Za-z0-9]{20,}\b"),
    re.compile(
        r"(?i)\b(api[_-]?key|api[_-]?secret|access[_-]?key|secret[_-]?key|token|password)"
        r"(\s*[:=]\s*)([^\s,;&\x22\x27]{6,})"
    ),
    re.compile(r"(?i)([?&](?:key|token|secret|apikey|api_key)=)[^&\s]+"),
)


class SecretRedactor:
    """Scrub registered secret values and common credential-shaped strings."""

    def __init__(self) -> None:
        self._values: set[str] = set()

    def register(self, value: str | None) -> None:
        if value and len(value) >= _MINIMUM_SECRET_LENGTH:
            self._values.add(value)

    def register_environment(
        self,
        environment: Mapping[str, str],
        names: Iterable[str] = SECRET_ENVIRONMENT_NAMES,
    ) -> None:
        for name in names:
            self.register(environment.get(name))

    def clear(self) -> None:
        self._values.clear()

    def scrub_text(self, text: str) -> str:
        result = text
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
            if groups and groups[0]:
                if groups[0].lower() == "bearer":
                    return f"{groups[0]} {REDACTED}"
                return f"{groups[0]}{REDACTED}"
            return REDACTED

        return pattern.sub(replace, text)

    def scrub(self, value: Any) -> Any:
        if isinstance(value, str):
            return self.scrub_text(value)
        if isinstance(value, dict):
            return {key: self.scrub(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self.scrub(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self.scrub(item) for item in value)
        return value


_DEFAULT = SecretRedactor()


def register_secret(value: str | None) -> None:
    _DEFAULT.register(value)


def redact(value: Any) -> Any:
    return _DEFAULT.scrub(value)


def default_redactor() -> SecretRedactor:
    return _DEFAULT
