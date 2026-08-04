# openalpha-research-core

Provenance primitives shared by every study: run manifests, artifact references
and an append-only run-state journal.

| Module | Purpose |
| --- | --- |
| `manifest.py` | `RunManifest` with git, environment and methodology metadata; canonical bytes and publication |
| `artifacts.py` | `ArtifactRef` and a local artifact store |
| `run_state.py` | `RunState` plus an append-only `RunStateJournal` |
| `errors.py` | Typed failures |

Manifests serialize to canonical bytes so a run's identity is a stable digest
rather than whatever key order a JSON encoder happened to choose.

```bash
uv run pytest packages/research-core/tests -q
```
