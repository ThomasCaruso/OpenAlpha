"""Deterministic fakes for the frozen inference diagnostic.

No network, no official asset, no Torch. Each fake is configurable so a test
can construct exactly the scientific situation it wants to distinguish: a
tokenizer that mangles its own round trip, a model whose generated tokens
decode outside the valid region, rollouts whose validity does or does not track
their accuracy, and a model that never emits a valid path at all.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable

from openalpha_bridge.diagnostic.backends import (
    GeneratedPath,
    ResolvedDiagnosticAssets,
    TokenPair,
)
from openalpha_bridge.diagnostic.spec import KRONOS_MINI_SPEC
from openalpha_bridge.diagnostic.validity import DecodedCandle
from openalpha_bridge.phase2.kronos import TOKENIZER_SPEC

__all__ = [
    "FakeCodec",
    "FakeForecastModel",
    "fake_assets",
    "frozen_digest",
]


def fake_assets(*, trainable: int = 0) -> ResolvedDiagnosticAssets:
    return ResolvedDiagnosticAssets(
        tokenizer_repository=TOKENIZER_SPEC.repository,
        tokenizer_revision=TOKENIZER_SPEC.revision,
        tokenizer_config_sha256=TOKENIZER_SPEC.config_sha256,
        tokenizer_weights_sha256=TOKENIZER_SPEC.weights_sha256,
        model_repository=KRONOS_MINI_SPEC.repository,
        model_revision=KRONOS_MINI_SPEC.revision,
        model_config_sha256=KRONOS_MINI_SPEC.config_sha256,
        model_weights_sha256=KRONOS_MINI_SPEC.weights_sha256,
        parameter_sha256="c" * 64,
        trainable_parameter_count=trainable,
        total_parameter_count=4_100_000,
    )


def frozen_digest() -> Callable[[], str]:
    """A parameter digest that never changes, as a frozen model's would not."""
    return lambda: "c" * 64


class FakeCodec:
    """A tokenizer whose round trip is lossless unless told otherwise.

    ``corrupt_round_trip`` makes the decoder emit geometrically invalid candles
    from in-distribution encoder tokens, which is the tokenizer-decoder defect
    the diagnostic must be able to detect.
    """

    def __init__(
        self,
        *,
        corrupt_round_trip: bool = False,
        corrupt_fraction: float = 0.5,
        decode_table: dict[tuple[int, int], DecodedCandle] | None = None,
    ) -> None:
        self.corrupt_round_trip = corrupt_round_trip
        self.corrupt_fraction = corrupt_fraction
        self._decode_table = decode_table or {}
        self._encoded: dict[tuple[int, int], DecodedCandle] = {}
        self.encode_calls = 0
        self.decode_calls = 0

    def encode(self, candles: tuple[DecodedCandle, ...]) -> tuple[TokenPair, ...]:
        self.encode_calls += 1
        tokens: list[TokenPair] = []
        for index, candle in enumerate(candles):
            pair = TokenPair(coarse=index % 1024, fine=(index * 7) % 1024)
            self._encoded[(pair.coarse, pair.fine)] = candle
            tokens.append(pair)
        return tuple(tokens)

    def decode(self, tokens: tuple[TokenPair, ...]) -> tuple[DecodedCandle, ...]:
        self.decode_calls += 1
        out: list[DecodedCandle] = []
        for index, pair in enumerate(tokens):
            key = (pair.coarse, pair.fine)
            if key in self._decode_table:
                out.append(self._decode_table[key])
                continue
            original = self._encoded.get(key)
            if original is None:
                # A token the encoder never produced. Decode it to something
                # plausible so generation paths are exercised.
                base = 100.0 + (pair.coarse % 50)
                out.append(
                    DecodedCandle(
                        open=base, high=base * 1.01, low=base * 0.99, close=base, volume=1e6
                    )
                )
                continue
            if self.corrupt_round_trip and index < len(tokens) * self.corrupt_fraction:
                # high below low: an invalid candle from an in-distribution token.
                out.append(
                    DecodedCandle(
                        open=original.open,
                        high=original.low * 0.5,
                        low=original.high * 1.5,
                        close=original.close,
                        volume=original.volume,
                    )
                )
                continue
            out.append(original)
        return tuple(out)


class FakeForecastModel:
    """A frozen model whose rollouts are decided by a per-seed policy.

    ``path_for`` receives the seed, not a call counter. A deterministic model
    given the same seed must produce the same path, so Method B and the rollout
    sharing its seed agree by construction rather than by accident. The paths
    are registered in the codec's decode table under synthetic tokens, so the
    diagnostic's real decode path is exercised rather than bypassed.
    """

    def __init__(
        self,
        *,
        codec: FakeCodec,
        path_for: Callable[[int, tuple[DecodedCandle, ...]], tuple[DecodedCandle, ...]],
    ) -> None:
        self._codec = codec
        self._path_for = path_for
        self._seen_seeds: list[int] = []
        self.generate_calls = 0
        self.parameter_writes = 0  # never incremented; a frozen model writes nothing

    @property
    def seeds_used(self) -> tuple[int, ...]:
        return tuple(self._seen_seeds)

    def generate(
        self,
        context: tuple[DecodedCandle, ...],
        *,
        steps: int,
        seed: int,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> GeneratedPath:
        self.generate_calls += 1
        self._seen_seeds.append(seed)
        path = self._path_for(seed, context)
        if len(path) != steps:
            raise AssertionError(f"fake produced {len(path)} steps, expected {steps}")

        tokens: list[TokenPair] = []
        for step, candle in enumerate(path):
            # Token ids derived from the seed so distinct rollouts have distinct
            # token sequences, which the diversity metrics then measure.
            digest = hashlib.sha256(f"{seed}:{step}".encode()).digest()
            pair = TokenPair(
                coarse=int.from_bytes(digest[:2], "big") % 1024,
                fine=int.from_bytes(digest[2:4], "big") % 1024,
            )
            self._codec._decode_table[(pair.coarse, pair.fine)] = candle
            tokens.append(pair)

        step_log_probabilities = tuple(-0.5 - (step % 5) * 0.01 for step in range(len(tokens)))
        return GeneratedPath(
            seed=seed,
            tokens=tuple(tokens),
            step_log_probabilities=step_log_probabilities,
            total_log_probability=math.fsum(step_log_probabilities),
        )
