# openalpha-experiment-spec

Loading, validation, canonicalization and migration of experiment specification
documents.

| Module | Purpose |
| --- | --- |
| `models.py` | The `ExperimentSpec` model |
| `canonical.py` | `canonical_bytes` and `experiment_id` — a specification's stable identity |
| `validation.py` | `load_yaml` / `load_json` with schema validation |
| `migrations.py` | Forward migration of older specification mappings |

Canonicalization is what makes preregistration enforceable: a specification hashes
to the same digest regardless of formatting, so the digest sealed before execution
can be re-verified inside the container at run time.
