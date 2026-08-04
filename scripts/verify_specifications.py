"""Verify every sealed preregistration digest in the repository.

Each specification YAML has a `.sha256` sidecar recording the digest that was
sealed before that study executed. This re-derives the digest from the committed
bytes and compares.

Deliberately not a `sha256sum -c` wrapper: the sidecars were written on Windows
and some carry CRLF, which the coreutils checker rejects outright. A reviewer
should be able to check provenance on any platform without that mattering.

Exits non-zero if any digest disagrees, so it is usable as a CI or make gate.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

RESEARCH = Path(__file__).resolve().parents[1] / "research" / "bridge-v0"


def main() -> int:
    sidecars = sorted(RESEARCH.glob("*.sha256"))
    if not sidecars:
        print(f"no .sha256 sidecars found under {RESEARCH}")
        return 1

    failures = 0
    print(f"Verifying {len(sidecars)} sealed specification digests\n")
    for sidecar in sidecars:
        # Tolerate CRLF, stray whitespace and the two-space sha256sum separator.
        text = sidecar.read_text(encoding="utf-8").strip()
        if not text:
            print(f"  EMPTY    {sidecar.name}")
            failures += 1
            continue
        expected, _, name = text.partition(" ")
        name = name.strip().lstrip("*").strip()
        if not name:
            name = sidecar.name.removesuffix(".sha256") + ".yaml"

        target = RESEARCH / name
        if not target.is_file():
            print(f"  MISSING  {name}")
            failures += 1
            continue

        observed = hashlib.sha256(target.read_bytes()).hexdigest()
        if observed == expected:
            print(f"  OK       {name}  {observed}")
        else:
            print(f"  MISMATCH {name}")
            print(f"           expected {expected}")
            print(f"           observed {observed}")
            failures += 1

    print()
    if failures:
        print(f"{failures} of {len(sidecars)} digests FAILED")
        return 1
    print(f"all {len(sidecars)} sealed digests verify")
    return 0


if __name__ == "__main__":
    sys.exit(main())
