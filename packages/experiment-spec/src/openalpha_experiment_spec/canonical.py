import hashlib
import json

from .models import ExperimentSpec


def canonical_bytes(spec: ExperimentSpec) -> bytes:
    payload = spec.model_dump(mode="json", exclude_none=False)
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def experiment_id(spec: ExperimentSpec) -> str:
    return f"exp_{hashlib.sha256(canonical_bytes(spec)).hexdigest()}"
