"""Deterministic fakes for the frozen inference diagnostic, v2 contract.

No network, no official asset, no Torch. Each fake is configurable so a test
can construct exactly the situation it wants to distinguish.

Both fakes take the normalization state explicitly and keep none of their own,
which is the property the v2 contract requires of the official adapters too.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable

from openalpha_bridge.diagnostic.backends import (
    GeneratedPath,
    ResolvedDiagnosticAssets,
    StepSampling,
    TokenPair,
)
from openalpha_bridge.diagnostic.normalization import NormalizationState
from openalpha_bridge.diagnostic.official_input import OfficialRow, TimeStamp
from openalpha_bridge.diagnostic.spec import KRONOS_MINI_SPEC, OFFICIAL_SOURCE_FILES
from openalpha_bridge.phase2.kronos import SOURCE_SPEC, TOKENIZER_SPEC

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
        source_revision=SOURCE_SPEC.revision,
        source_as_committed_sha256={
            f.relative_path: f.as_committed_sha256 for f in OFFICIAL_SOURCE_FILES
        },
        source_crlf_normalized_sha256={
            f.relative_path: f.sealed_sha256_crlf_normalized for f in OFFICIAL_SOURCE_FILES
        },
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
    from in-distribution encoder tokens.
    """

    def __init__(
        self,
        *,
        corrupt_round_trip: bool = False,
        corrupt_count: int | None = None,
        corrupt_fraction: float = 0.5,
    ) -> None:
        self.corrupt_round_trip = corrupt_round_trip
        self.corrupt_count = corrupt_count
        self.corrupt_fraction = corrupt_fraction
        self._decode_table: dict[tuple[int, int], OfficialRow] = {}
        self._encoded: dict[tuple[int, int], OfficialRow] = {}
        self.encode_calls = 0
        self.decode_calls = 0
        #: Every normalization state hash this codec was handed. A v2 adapter
        #: must be told the state; it must never remember one.
        self.states_seen: list[str] = []

    def encode(
        self, rows: tuple[OfficialRow, ...], *, state: NormalizationState
    ) -> tuple[TokenPair, ...]:
        self.encode_calls += 1
        self.states_seen.append(state.state_sha256)
        # Exercise the real transform so a wrong state would be observable.
        state.normalize(rows)
        tokens: list[TokenPair] = []
        for index, row in enumerate(rows):
            pair = TokenPair(coarse=index % 1024, fine=(index * 7) % 1024)
            self._encoded[(pair.coarse, pair.fine)] = row
            tokens.append(pair)
        return tuple(tokens)

    def decode(
        self,
        tokens: tuple[TokenPair, ...],
        *,
        state: NormalizationState,
        sessions: tuple,
    ) -> tuple[OfficialRow, ...]:
        self.decode_calls += 1
        self.states_seen.append(state.state_sha256)
        if len(sessions) != len(tokens):
            raise AssertionError(f"decode got {len(tokens)} tokens and {len(sessions)} sessions")
        limit = (
            self.corrupt_count
            if self.corrupt_count is not None
            else int(len(tokens) * self.corrupt_fraction)
        )
        out: list[OfficialRow] = []
        for index, pair in enumerate(tokens):
            key = (pair.coarse, pair.fine)
            session = sessions[index]
            if key in self._decode_table:
                out.append(self._decode_table[key].model_copy(update={"session": session}))
                continue
            original = self._encoded.get(key)
            if original is None:
                base = 100.0 + (pair.coarse % 50)
                out.append(
                    OfficialRow(
                        session=session,
                        open=base,
                        high=base * 1.01,
                        low=base * 0.99,
                        close=base,
                        volume=1e6,
                        amount=1e6 * base,
                    )
                )
                continue
            if self.corrupt_round_trip and index < limit:
                out.append(
                    OfficialRow(
                        session=session,
                        open=original.open,
                        high=original.low * 0.5,
                        low=original.high * 1.5,
                        close=original.close,
                        volume=original.volume,
                        amount=original.amount,
                    )
                )
                continue
            out.append(original.model_copy(update={"session": session}))
        return tuple(out)


class FakeForecastModel:
    """A frozen model whose rollouts are decided by a per-seed policy.

    ``path_for`` receives the seed, not a call counter, so a deterministic
    model given the same seed produces the same path by construction.
    """

    def __init__(
        self,
        *,
        codec: FakeCodec,
        path_for: Callable[[int, tuple[OfficialRow, ...]], tuple[OfficialRow, ...]],
    ) -> None:
        self._codec = codec
        self._path_for = path_for
        self._seen_seeds: list[int] = []
        self.generate_calls = 0
        self.parameter_writes = 0  # never incremented; a frozen model writes nothing
        self.states_seen: list[str] = []
        self.context_lengths: list[int] = []
        self.stamp_shapes: list[tuple[int, int]] = []

    @property
    def seeds_used(self) -> tuple[int, ...]:
        return tuple(self._seen_seeds)

    def generate(
        self,
        context: tuple[OfficialRow, ...],
        *,
        context_stamps: tuple[TimeStamp, ...],
        target_stamps: tuple[TimeStamp, ...],
        target_sessions: tuple,
        state: NormalizationState,
        steps: int,
        seed: int,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> GeneratedPath:
        self.generate_calls += 1
        self._seen_seeds.append(seed)
        self.states_seen.append(state.state_sha256)
        self.context_lengths.append(len(context))
        self.stamp_shapes.append((len(context_stamps), len(target_stamps)))

        if len(context_stamps) != len(context):
            raise AssertionError("one stamp per context row is required")
        if len(target_stamps) != steps:
            raise AssertionError("one stamp per predicted step is required")
        if len(target_sessions) != steps:
            raise AssertionError("one session per predicted step is required")

        path = self._path_for(seed, context)
        if len(path) != steps:
            raise AssertionError(f"fake produced {len(path)} steps, expected {steps}")

        tokens: list[TokenPair] = []
        sampling: list[StepSampling] = []
        for step, row in enumerate(path):
            digest = hashlib.sha256(f"{seed}:{step}".encode()).digest()
            pair = TokenPair(
                coarse=int.from_bytes(digest[:2], "big") % 1024,
                fine=int.from_bytes(digest[2:4], "big") % 1024,
            )
            self._codec._decode_table[(pair.coarse, pair.fine)] = row
            tokens.append(pair)
            sampling.append(
                StepSampling(
                    coarse_log_probability=-0.30 - (step % 5) * 0.01,
                    fine_conditional_log_probability=-0.20 - (step % 3) * 0.01,
                )
            )

        # Decode the official way: context tokens concatenated with the
        # generated ones, whole window decoded, last `steps` rows sliced. The
        # real backend does exactly this; doing it here too means the tests
        # exercise the contract rather than a simplification of it.
        context_tokens = self._codec.encode(context, state=state)
        window = (*context_tokens, *tokens)
        window_sessions = (*(row.session for row in context), *target_sessions)
        decoded_window = self._codec.decode(tuple(window), state=state, sessions=window_sessions)
        raw_suffix = decoded_window[-steps:]

        return GeneratedPath(
            seed=seed,
            tokens=tuple(tokens),
            sampling=tuple(sampling),
            total_path_sampling_log_probability=math.fsum(s.pair_log_probability for s in sampling),
            raw_decoded_suffix=raw_suffix,
        )
