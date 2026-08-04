"""Verify every committed terminal artifact against its recorded digest.

The artifacts under `research/artifacts/` are the exact object-store bodies. This
recomputes each SHA-256 and compares it against the digest recorded at
publication, so a reader can confirm the committed evidence is the same evidence
the study produced.

Also checks that each artifact's own payload agrees with the manifest about its
run ID, experiment, outcome and source commit — a file could otherwise carry the
right digest under the wrong description.

Exits non-zero on any mismatch, so it is usable as a CI or make gate.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ARTIFACTS = Path(__file__).resolve().parents[1] / "research" / "artifacts"


def main() -> int:
    manifest_path = ARTIFACTS / "manifest.json"
    if not manifest_path.is_file():
        print(f"missing manifest: {manifest_path}")
        return 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    committed = manifest["artifacts"]
    recorded_only = manifest.get("not_committed", [])
    failures = 0

    print(f"Verifying {len(committed)} committed terminal artifacts\n")
    for entry in committed:
        name = entry["committed_file"]
        path = ARTIFACTS / name
        if not path.is_file():
            print(f"  MISSING  {name}")
            failures += 1
            continue

        body = path.read_bytes()
        observed = hashlib.sha256(body).hexdigest()
        expected = entry["artifact_sha256"]
        if observed != expected:
            print(f"  MISMATCH {name}")
            print(f"           expected {expected}")
            print(f"           observed {observed}")
            failures += 1
            continue
        if len(body) != entry["size_bytes"]:
            print(f"  SIZE     {name}: {len(body)} bytes, manifest says {entry['size_bytes']}")
            failures += 1
            continue

        # The digest proves the bytes; this proves the manifest describes them.
        payload = json.loads(body)
        for field, key in (
            ("run_id", "run_id"),
            ("experiment_id", "experiment_id"),
            ("outcome", "outcome"),
            ("source_commit", "source_commit"),
            ("schema_version", "schema_version"),
        ):
            if payload.get(field) != entry[key]:
                print(f"  DESCRIB  {name}: payload {field}={payload.get(field)!r} "
                      f"but manifest says {entry[key]!r}")
                failures += 1
                break
        else:
            print(f"  OK       {name}")
            print(f"           {observed}")
            print(f"           {entry['run_id']}  {entry['outcome']}  {len(body):,} bytes")

    if recorded_only:
        print(f"\nRecorded but not committed ({len(recorded_only)}):")
        for entry in recorded_only:
            print(f"  {entry['run_id']}  {entry['artifact_sha256']}")
            print(f"    {entry['status']}")

    print()
    if failures:
        print(f"{failures} artifact check(s) FAILED")
        return 1
    print(f"all {len(committed)} committed artifacts verify against their recorded digests")
    return 0


if __name__ == "__main__":
    sys.exit(main())
