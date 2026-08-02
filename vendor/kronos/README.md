# Vendored Kronos source, for conformance verification only

Verbatim copies of two files from the pinned official revision:

- Repository: <https://github.com/shiyu-coder/Kronos>
- Revision: `67b630e67f6a18c9e9be918d9b4337c960db1e9a`
- Files: `model/kronos.py`, `model/module.py`

## Why these are here

The diagnostic's inference settings, normalization formulas, sampling order and
tensor shapes are derived from this source rather than from the README. Deriving
them from a copy that is fetched at test time would make the test suite depend
on the network and on GitHub continuing to serve the same bytes. These copies
let the conformance tests run offline and deterministically, and their digests
are verified against the values sealed in `experiment.yaml`.

## What they are not

They are never imported, never executed, and never shipped in the Modal image.
The image clones the upstream repository at the pinned revision itself. Nothing
in `packages/` imports anything under `vendor/`.

## Digests

| File | As committed upstream | Line ending | Bytes | Sealed (CRLF-normalized) |
| --- | --- | --- | --- | --- |
| `model/kronos.py` | `0a5f9028…1032` | LF | 30133 | `638a56e0…9f3a` |
| `model/module.py` | `a07edbad…a409f` | CRLF | 23426 | `a07edbad…a409f` |

The digests sealed in `experiment.yaml` are over CRLF-normalized content.
`module.py` is committed upstream with CRLF, so its as-committed digest already
equals the sealed one. `kronos.py` is committed with LF, so the two differ and
the sealed value is the digest of its CRLF-normalized form. One rule explains
both sealed values, and both digests are verified independently.

`.gitattributes` marks these files `-text` so git never rewrites their line
endings on checkout. Without it, a Windows checkout would convert `kronos.py`
to CRLF and the as-committed digest would stop matching — which is precisely
how the sealed digest came to be a CRLF one in the first place.
