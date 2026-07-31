from pathlib import Path

from openalpha_sentinel.development_execution import run_reproducibility_probe
from openalpha_sentinel.development_manifest import build_development_manifest
from openalpha_sentinel.inference_cache import InferenceCache
from openalpha_sentinel.providers.kronos import (
    InferenceBatchResult,
    InferenceEnvironment,
)

from packages.sentinel.tests.test_development_origin import (
    NOW,
    FakeInference,
    FakeMarketProvider,
    _snapshot,
)


class FakeBatchClient:
    def __init__(self) -> None:
        self.inference = FakeInference()
        self.calls = 0
        self.environment = InferenceEnvironment(
            python_version='3.11.9',
            torch_version='2.7.1',
            numpy_version='2.2.6',
            pandas_version='2.3.3',
            operating_system='Windows',
            device='cpu',
            cache_path='C:/private/hf',
            cache_size_bytes=32_000_000,
            downloaded_files=(),
            official_predictor_derives_amount=True,
        )

    def forecast_many(self, requests):
        self.calls += 1
        return InferenceBatchResult(
            environment=self.environment,
            responses=self.inference(requests),
        )


def test_replay_probe_preserves_both_outputs_and_populates_verified_cache(
    tmp_path: Path,
) -> None:
    origin = build_development_manifest().origins[0]
    provider = FakeMarketProvider(_snapshot(origin.cutoff))
    client = FakeBatchClient()
    cache = InferenceCache(tmp_path / 'inference-cache')

    probe = run_reproducibility_probe(
        origin=origin,
        state_root=tmp_path,
        market_provider=provider,
        inference_client=client,
        inference_cache=cache,
        clock=lambda: NOW,
    )

    assert probe.deterministic_replay_supported is True
    assert probe.first_path_sha256 == probe.second_path_sha256
    assert probe.request_count == 2
    assert probe.request_success_count == 2
    assert client.calls == 2
    assert (tmp_path / 'reproducibility-probe.json').is_file()
    assert len(probe.output_artifact_sha256) == 2

    repeated = run_reproducibility_probe(
        origin=origin,
        state_root=tmp_path,
        market_provider=provider,
        inference_client=client,
        inference_cache=cache,
        clock=lambda: NOW,
    )
    assert repeated == probe
    assert client.calls == 2
