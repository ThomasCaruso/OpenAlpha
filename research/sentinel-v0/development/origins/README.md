# Development origin index

The verified `development_table.jsonl` in the parent directory is the compact,
deterministically ordered one-row-per-origin index for all 104 Phase 3A
development origins.

The complete content-addressed forecast, path, projection, and outcome chains
remain in the private Phase 3A state root outside Git. They are not duplicated
here because doing so would increase repository size and risk redistributing
provider-derived data. Each committed analysis row retains the origin ID,
cutoff, data hash, model revision, diagnostics, status, outcome, errors, and
runtime metadata required to reconcile it with the private verified chain.
