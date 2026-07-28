# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] — unreleased

A complete rebuild. **1.0.0 did not provide the tamper-evidence it documented**;
see [SECURITY.md](SECURITY.md) for what to assume if you deployed it.

Numbered 2.0.0 rather than continuing from 1.0.0's lineage because the API is
entirely different — and because pip resolves to the highest version, so a lower
number would have left `pip install signledger` serving the withdrawn release.

### Security

- **A hash chain is now actually built on every backend.** In 1.0.0, `metadata`
  defaulted to `{}`, which is falsy, so SQLite and PostgreSQL stored SQL `NULL`;
  reading the row back failed validation; and the append path caught that
  exception, logged a warning, and continued with `previous_hash = None`. On the
  documented default configuration, every row was an unlinked genesis block.
- **Signatures are verified.** In 1.0.0 `Signer.verify()` had no call sites at
  all. Verification now resolves the key through a `KeyResolver` and checks every
  signature, over a domain-separated payload
  (`b"signledger/v2\x00" + ledger_id + b"\x00" + entry_hash`) so a signature
  cannot be replayed into another ledger.
- **Truncation and wholesale rewrite are detectable**, via checkpoints and
  external anchoring. 1.0.0 anchored nothing, and because the chain is unkeyed,
  an actor with write access could recompute a fabricated chain using the
  library's own code.
- **Merkle inclusion proofs are checked against a caller-supplied root.** 1.0.0's
  `verify_proof()` validated a proof against the root stored inside that same
  proof, so it returned `True` for every input. Interior and leaf hashes are now
  domain-separated per RFC 6962, which also closes the CVE-2012-2459
  duplicate-leaf forgery.
- **`ledger.append()` no longer leaks its payload into error messages.** A raw
  pydantic `ValidationError` renders `input_value=...` verbatim; a rejected entry
  therefore printed the data it was rejecting into any log that caught it.
  Validation errors are now redacted to field location and error type, raised
  `from None` so a traceback cannot re-expose the value.
- **Backend construction errors are translated.** `SQLiteBackend(...)` raised raw
  `sqlite3.OperationalError`, which `except SignLedgerError` did not catch and
  whose message disclosed the database path.

### Added

- `Ledger.check_integrity()` returns a `VerificationReport` carrying an
  `AssuranceLevel` (`NONE` / `CHAIN_ONLY` / `SIGNED` / `ANCHORED` / `WORM`)
  alongside the verdict, plus typed `VerificationFailure` records with a
  `FailureKind`. A chain that verifies is not the same as a chain nobody could
  have rewritten, and the report now says which one you have.
- SLCS/1 canonical serialization (RFC 8785 JCS profile): UTF-16 code-unit key
  ordering, integer epoch-microsecond datetimes, rejection of NaN/Infinity and
  non-string keys, and explicit size and nesting bounds. Pinned by a test-vector
  file so a future change to the encoder is a visible diff.
- `format_version` inside the hash preimage, so the encoding can change without
  silently invalidating old ledgers.
- Atomic append on every backend — `BEGIN IMMEDIATE`, `SELECT ... FOR UPDATE`,
  and a `findAndModify` counter respectively — with `UNIQUE(sequence)` and
  `UNIQUE(previous_hash)` as the database-level backstop against a forked chain.
- Checkpoints and `AnchorSink`, for publishing the chain head somewhere the
  writer cannot reach.
- A `BackendConformanceSuite` that every backend runs, so a custom backend can
  prove it is interchangeable with the shipped ones.
- `signledger` CLI: `init`, `append`, `verify`, `checkpoint`, `anchor`,
  `export`, `migrate`. In 1.0.0 the console script was dead on arrival — an empty
  `cli/` directory shadowed `cli.py`.
- `signledger.contrib` integrations for Flask, FastAPI and Django, all rebuilt.
  The 1.0.0 versions were shadowed the same way and never ran.
- Field redaction (`signledger.contrib.redaction`) for keeping secrets out of
  audit payloads, applied before hashing so the hash covers what is stored.
- `BatchProcessor` with an explicit atomic-vs-best-effort choice and a
  dead-letter path.
- `signledger.observability`: metrics and structured events that report real
  verification state rather than a hardcoded `True`.
- `docs/ARCHITECTURE.md` — the normative contract. Every guarantee, encoding rule
  and interface in this release is specified there first.

### Changed

- **The entire public API.** Nothing from 1.0.0 is source-compatible.
- Packaging moved to `pyproject.toml` (PEP 621) with an explicit `packages` list.
  `find_packages()` is deliberately not used: it is what shipped 1.0.0's four
  empty directories.
- `import signledger` no longer imports `cryptography`, `psycopg`, `pymongo` or
  any web framework. Optional extras stay optional.
- Minimum Python is 3.9.
- Each backend has exactly one module name (`signledger.backends.postgresql`,
  `signledger.backends.mongodb`) — no aliases.

### Removed

- The empty `cache/`, `compression/`, `consensus/`, `distributed/`, `pool/` and
  `integrations/*` package directories. They contained no working code and
  shadowed their own sibling modules.
- `setup.py`, which hardcoded `version="1.0.0"`, declared dependencies the code
  never used, and fought `pyproject.toml` during the wheel build.
- The 1.0.0 README and examples, which documented `RedisBackend`,
  `signledger.compression.Compressor` and `ledger.get_entries()` — none of which
  have ever existed — and quoted invented performance numbers.

### Fixed

- A genuine circular import between `crypto.canonical`, `core.exceptions` and
  `core.entry`, live but masked by package `__init__` ordering. Reordering two
  lines would have broken every `import signledger.crypto.*` at once.
- The Django integration's migration was missing from the built wheel, so
  `manage.py migrate` created no table and every audited request failed under the
  default fail-closed policy.

### Testing

2,810 tests. 100% line **and** branch coverage across all 37 modules, measured
against live PostgreSQL and MongoDB rather than mocks — 582 of those tests need a
real server, and CI fails if any of them is skipped. `ruff` and `mypy --strict`
are clean. Property-based tests use Hypothesis; mutation testing runs in CI.

Stated plainly because the last release did not: 1.0.0 shipped **zero** tests
while its README claimed "100% test coverage".

## [1.0.0] — 2025-10-21 — WITHDRAWN

Do not use. See [SECURITY.md](SECURITY.md).

[2.0.0]: https://github.com/0xV8/signledger/releases/tag/v2.0.0
[1.0.0]: https://github.com/0xV8/signledger/releases/tag/v1.0.0
