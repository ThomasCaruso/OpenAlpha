# Sentinel v1.1 Token-Manifold Compatibility

**DEVELOPMENT COMPATIBILITY RESEARCH - NOT HOLDOUT EVIDENCE**

## Conclusion

Root cause: TOKENIZER_CONSTRAINT_DEFECT.

Decision: STOP_CONSTRAINED_DECODING_RESEARCH_SHIP_ASSURANCE_LIBRARY.

## Tokenizer round trip

- tokenizer_2k: 4357 of 15360 reconstructed candles invalid (28.3659%); Wilson 95% CI [27.6585%, 29.0841%].
- tokenizer_base: 3555 of 15360 reconstructed candles invalid (23.1445%); Wilson 95% CI [22.4843%, 23.8182%].

## Autoregressive compatibility characterization

- Raw invalid candles: 68 of 180 (37.7778%).
- Unsupported raw token pairs: 59 (32.7778%).
- Invalid raw candles using unsupported pairs: 26.4706%.

## Truncated probability-mass bounds

- Median considered mass: 1.000000.
- Median valid mass: [0.599689, 0.602011].
- Median valid-and-supported mass: [0.348095, 0.350324].
- These are truncated bounds, not exact full-vocabulary mass.

## Governance

The tokenizer-defect gate did not authorize another decoder. No realized outcomes were needed for the classification, and the untouched holdout was not accessed.
