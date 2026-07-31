# Sentinel Constraint Compatibility

## Product decision

Sentinel's defensible product is an assurance library, not another unconstrained
decoder experiment.

Its supported responsibilities are:

1. validate financial forecast structure;
2. retain immutable raw model output and provenance;
3. emit structured violations without mutating the raw path;
4. provide a separately labeled deterministic terminal projection;
5. profile tokenizer and generated-token compatibility;
6. compare released model/tokenizer pairs as a model-selection safety check.

## Runtime contract

Raw generated paths are always retained for audit. A path that violates the
financial grammar has structural_status=FAILED and must not be passed to
path-dependent consumers as structurally valid.

Terminal projection keeps predicted open and close unchanged, raises high to the
candle maximum, and lowers low to the candle minimum. It must:

- return a structurally valid path or an explicit failure;
- record each changed field and normalized adjustment;
- preserve the implied close-return forecast;
- remain separate from the raw model output;
- make no accuracy-improvement claim.

The projection is post hoc. It cannot undo later autoregressive conditioning on an
invalid intermediate candle.

## Why constrained decoding stopped

Sentinel v1 showed that terminal projection guarantees final validity, stepwise
projection/re-encoding is operationally unreliable, and generic valid-candidate
resampling degrades high-low range accuracy. Sentinel v1.1 then found that both
released tokenizers materially violate OHLC ordering during encode-decode
reconstruction of valid observed candles.

The preregistered v1.1 rule classifies that evidence as
TOKENIZER_CONSTRAINT_DEFECT. Support-conditioned sampling is authorized only for
off-manifold generation, low valid probability mass, or candidate-search failure.
It is therefore not run. The model-size autoregressive canary is also stopped by
the tokenizer-defect gate; Tokenizer-base itself is not overwhelmingly valid.

## Evidence limits

The result applies only to the pinned Kronos source and checkpoints, the tested
tokenizers, daily ETF sequences, the fixed development corpus, and the five-step
feasibility design. It does not establish:

- that Kronos is broadly unusable;
- that structural invalidity predicts return error;
- that terminal projection improves forecasting;
- that another tokenizer or frequency behaves the same way;
- provider-independent market evidence;
- holdout performance.

No untouched holdout origin was accessed. No model or tokenizer weights were
modified, trained, or committed.

## Future boundary

Any constraint-preserving tokenizer or candle parameterization requires a separate
training experiment and new data/governance decisions. It is not another
inference-only Sentinel phase and must not reuse the untouched holdout for
iteration.
