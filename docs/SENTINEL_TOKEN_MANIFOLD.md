# Sentinel v1.1 Token-Manifold Audit

## Claim boundary

**DEVELOPMENT COMPATIBILITY RESEARCH - NOT HOLDOUT EVIDENCE**

Sentinel v1.1 asks whether Kronos structural invalidity first appears in the
tokenizer reconstruction, in autoregressive token selection, or in the amount of
model probability compatible with the financial grammar. It does not refit the
retired v0 reliability model and does not use the untouched holdout.

## Distinctions

- Mathematical validity means a decoded candle satisfies finite, positive OHLC,
  high/low ordering, nonnegative generated volume, and sequence constraints.
- Tokenizer support means a hierarchical token pair appears at least twice in the
  fixed development-only corpus under the same tokenizer.
- Model support means Kronos assigns conditional probability to a pair at a
  particular forecast step.
- Distributional distortion is the probability and path change introduced by
  conditioning or projection.

These concepts are not interchangeable. A mathematically valid pair can be rare,
unsupported, or far from the raw forecast distribution.

## Fixed corpus and round trip

The corpus contains the final 1,536 complete XNYS sessions before 2024-06-29 for
SPY, QQQ, IWM, DIA, TLT, HYG, GLD, EFA, EEM, and XLF. Each instrument contributes
three nonoverlapping 512-session windows. Raw yfinance rows remain ephemeral and
outside Git.

Each valid observed sequence was passed through the pinned preprocessing,
tokenizer encode, and tokenizer decode path without autoregressive generation.
The reconstruction was then validated against the financial grammar.

| Tokenizer | Invalid candles | Total | Fraction | Wilson 95% interval |
|---|---:|---:|---:|---:|
| Kronos-Tokenizer-2k | 4,357 | 15,360 | 28.3659% | 27.6585%-29.0841% |
| Kronos-Tokenizer-base | 3,555 | 15,360 | 23.1445% | 22.4843%-23.8182% |

Both exceed the preregistered material-defect rule by a wide margin. Violations
therefore exist in continuous reconstruction before autoregressive model
generation. This supports TOKENIZER_CONSTRAINT_DEFECT as the primary root-cause
classification; it does not imply every error in a generated path is caused only
by the tokenizer.

## Token support and probability bounds

The generated-token audit reuses the twelve fixed Sentinel v1 development origins,
three seeds, context 512, and five steps. It verifies every raw path hash against
the sealed v1 artifact. Exact pair support is tokenizer-specific and requires a
corpus count of at least two.

At each step the audit records the raw coarse/fine pair, empirical support, model
log probability and ranks, raw structural validity, and nested 64-, 256-, and
1,024-pair probability grids. Reported valid and supported masses are lower bounds.
The corresponding upper bound adds all uncovered tail mass and is capped at one.
Truncated estimates are never presented as exact.

The audit is a secondary characterization. Under the preregistered round-trip
decision, it cannot authorize another decoder after a material tokenizer defect is
confirmed.

All twelve origins and 36 paths completed with exact parity to the sealed Sentinel
v1 raw paths. Of 180 generated candles, 68 were invalid (37.7778%). Fifty-nine raw
pairs were unsupported under the count-two rule. Only 18 of the 68 invalid candles
used unsupported pairs (26.4706%); supported pairs had a 41.3223% invalid rate,
versus 30.5085% for unsupported pairs. Unsupported token combinations therefore do
not explain most observed invalidity in this sample.

At the 1,024-pair grid, the median considered mass was 1.0. Because this is a
median and individual steps can retain uncovered tail mass, the separately
aggregated median valid lower/upper bounds were 0.599689 and 0.602011. The median
valid-and-supported bounds were 0.348095 and 0.350324. These secondary facts do not
override the earlier tokenizer-defect classification.

## Consequence

The released continuous decoder does not guarantee the financial domain contract.
Inference-time terminal projection remains the practical deterministic mitigation:
preserve the raw path, expose every violation, publish a separate projected path,
and never claim that projection improves forecast accuracy.

A true constraint-preserving representation is a separate training research track.
One candidate parameterization models open return, candle body, nonnegative upper
wick, nonnegative lower wick, and nonnegative volume so the inverse transform
guarantees valid OHLC. Sentinel v1.1 does not train that representation.
