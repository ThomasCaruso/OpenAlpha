from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def render_phase_2_audit(
    *,
    forecast: Mapping[str, Any],
    outcome: Mapping[str, Any],
    forecast_id: str,
    manifest_sha256: str,
    artifact_chain_verified: bool,
    artifact_hashes: Mapping[str, str] | None = None,
) -> str:
    provider = _mapping(forecast.get("provider"))
    environment = _mapping(forecast.get("model_environment"))
    probe = _mapping(forecast.get("reproducibility_probe"))
    individual_paths = _list_of_mappings(forecast.get("individual_paths"))
    diagnostics = _list_of_mappings(forecast.get("diagnostics"))
    contexts = _list_of_mappings(forecast.get("context_summaries"))
    failures = forecast.get("request_failures") or []
    limitations = forecast.get("limitations") or []
    lines = [
        "# OpenAlpha Sentinel Phase 2 Audit",
        "",
        f"**{forecast['claim_boundary']}**",
        "",
        (
            "This single historical origin validates plumbing and auditability only. "
            "It does not show that Kronos or Sentinel works."
        ),
        "",
        "## Origin and provenance",
        "",
        f"- Asset: {forecast['symbol']}",
        f"- Forecast cutoff: {forecast['cutoff']}",
        f"- Forecast sessions: {', '.join(str(item) for item in forecast['forecast_sessions'])}",
        "- Horizon: 5 XNYS trading sessions",
        f"- Provider: {provider.get('provider')} via yfinance {provider.get('client_version')}",
        "- Feed: not applicable (unofficial public interface)",
        "- Adjustment: raw; auto/back adjustment and repair disabled",
        f"- Model: {environment.get('model_repository')}@{environment.get('model_revision')}",
        (
            f"- Tokenizer: {environment.get('tokenizer_repository')}"
            f"@{environment.get('tokenizer_revision')}"
        ),
        (
            f"- Environment: Python {environment.get('python_version')}, "
            f"PyTorch {environment.get('torch_version')}, device {environment.get('device')}"
        ),
        (
            f"- Numerical stack: NumPy {environment.get('numpy_version')}, "
            f"pandas {environment.get('pandas_version')}"
        ),
        f"- Operating system: {environment.get('operating_system')}",
        f"- Cache footprint: {environment.get('cache_size_bytes')} bytes outside Git",
        f"- Forecast ID: {forecast_id}",
        f"- Completed manifest: {manifest_sha256}",
        f"- Artifact chain verified: **{'yes' if artifact_chain_verified else 'no'}**",
        "",
        "## Reproducibility probe",
        "",
        f"- A (512/1729): {probe.get('a_sha256')}",
        f"- B (512/1729): {probe.get('b_sha256')}",
        f"- C (512/2027): {probe.get('c_sha256')}",
        (
            "- Exact same-seed replay supported: "
            f"{str(bool(probe.get('exact_seed_replay_supported'))).lower()}"
        ),
        "",
        "## Nine individual predicted returns",
        "",
        "| Context | Seed | Five-session log return | Latency ms | Output warnings |",
        "|---:|---:|---:|---:|---|",
    ]
    for item in individual_paths:
        lines.append(
            f"| {item.get('context_length')} | {item.get('sampling_seed')} | "
            f"{_number(item.get('predicted_log_return'))} | "
            f"{_number(item.get('inference_duration_ms'))} | "
            f"{', '.join(str(value) for value in item.get('output_quality_warnings', [])) or 'none'} |"
        )
    lines.extend(
        [
            "",
            "## Canonical 512-context forecast",
            "",
            (
                "Only the three 512-context paths form the primary forecast. "
                "The 128/256 paths are stress diagnostics."
            ),
            "",
        ]
    )
    for item in individual_paths:
        if item.get("context_length") == 512:
            lines.append(
                f"- Seed {item.get('sampling_seed')} close path: "
                f"{_path(item.get('close_path'))}"
            )
    lines.extend(
        [
            f"- Timestamp-wise averaged close path: {_path(forecast.get('canonical_close_path'))}",
            (
                "- Canonical predicted raw log return: "
                f"{_number(forecast.get('canonical_predicted_log_return'))}"
            ),
            "",
            "## Context-level summaries",
            "",
        ]
    )
    for item in contexts:
        lines.append(
            f"- Context {item.get('context_length')}: "
            f"{_number(item.get('predicted_log_return'))}"
        )
    lines.extend(["", "## Forecast-time diagnostics", ""])
    for item in diagnostics:
        if item.get("status") == "available":
            lines.append(
                f"- {item.get('name')}: {_number(item.get('value'))} "
                f"({item.get('units')})"
            )
        else:
            lines.append(
                f"- {item.get('name')}: not computable — {item.get('reason')}"
            )
    lines.extend(
        [
            "",
            "## Resolved outcome",
            "",
            f"- Realized raw log return: {_number(outcome.get('realized_log_return'))}",
            (
                "- Kronos absolute return error: "
                f"{_number(outcome.get('kronos_absolute_return_error'))}"
            ),
            (
                "- Baseline absolute error: "
                f"{_number(outcome.get('baseline_absolute_return_error'))}"
            ),
            f"- Direction correct: {str(bool(outcome.get('direction_correct'))).lower()}",
            f"- Closer model at this origin: {outcome.get('closer_model')}",
            "",
            "## Runtime and failures",
            "",
            f"- Total runtime: {_number(forecast.get('total_runtime_seconds'))} seconds",
            f"- Request retries: {forecast.get('request_retries')}",
            f"- Request failures: {len(failures)}",
        ]
    )
    for failure in failures:
        lines.append(f"  - {failure}")
    lines.extend(["", "## Limitations", ""])
    for limitation in limitations:
        lines.append(f"- {limitation}")
    lines.extend(
        [
            (
                "- The official Kronos predictor derives an internal amount feature from OHLCV; "
                "Sentinel neither requested nor stored provider amount data."
            ),
            (
                "- This origin uses modern Yahoo Finance history through an unofficial client, "
                "not an institutional point-in-time dataset."
            ),
            "- Dividend cash return is omitted from the raw close-to-close target.",
            "- Cross-provider verification is deferred.",
            "- One origin cannot establish diagnostic predictiveness, calibration, or efficacy.",
            "",
            "## Downloaded model files",
            "",
        ]
    )
    for item in _list_of_mappings(environment.get("downloaded_files")):
        lines.append(
            f"- {item.get('repository')}@{item.get('revision')} / "
            f"{item.get('relative_path')} — {item.get('size_bytes')} bytes — "
            f"SHA-256 {item.get('sha256')}"
        )
    lines.extend(["", "## Artifact hashes", ""])
    for name, digest in sorted((artifact_hashes or {}).items()):
        lines.append(f"- {name}: {digest}")
    lines.append("")
    return "\n".join(lines)


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _list_of_mappings(value: object) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _number(value: object) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{float(value):.10g}"
    return str(value)


def _path(value: object) -> str:
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_number(item) for item in value) + "]"
    return str(value)
