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
- `signledger` CLI: `verify`, `stats`, `inspect`, `checkpoint`, `anchor`, `export`,
  `verify-bundle` and `migrate`. In 1.0.0 the console script was dead on arrival —
  an empty `cli/` directory shadowed `cli.py`. Note that `migrate` moves a ledger
  that *verifies* between stores; it is not a 1.0.0 upgrade path, and no such path
  exists — see [SECURITY.md](SECURITY.md).
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
- **Minimum Python is 3.9. 1.0.0 supported 3.8; this release drops it.** The code was
  measured to run on 3.8, so this is a policy decision rather than a technical one: 3.8
  stopped receiving security fixes in October 2024, and a library whose only purpose is
  producing evidence should not run on an unpatched interpreter. Note that a 3.8 user
  running `pip install -U signledger` is not upgraded and is not warned — the resolver
  simply keeps 1.0.0, which is why yanking it matters.
  Verified working: CPython 3.9 through 3.14. PyPy is **not** supported.
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

Found by building four real consumer projects — a Flask service, a FastAPI
service, a Django project and a plain/CLI project — installing the published
wheel into each, and driving them with real traffic. Everything below was
reproduced by execution.

- **PostgreSQL reported `append_only_enforced` on a table with no trigger.** The
  catalog probe matched `pg_class.relname` with no schema join, so once any schema
  held a `ledger_entries` trigger, a backend opened against a *different* schema
  both skipped creating its own guard and kept claiming the capability. With
  signatures and an anchor, `assurance_level()` therefore reported `worm` about a
  table anyone could freely `UPDATE`. Resolution is now by relation OID.
- **Request bodies leaked whole categories of credential.** Two rule sets shipped:
  `?api_key=` was redacted and `{"api_key": ...}` was written verbatim. Unified
  onto one rule, then widened after fuzzing 146,000 generated field names found
  one-time codes, crypto material, bearer tokens, certificates and PII
  identifiers (`otp`, `passcode`, `salt`, `mnemonic`, `jwt`, `pem`, `iban`, …)
  still passing through. A fuzz corpus now guards the rule in CI.
- **A field name could stall the audit write.** Name-splitting was quadratic, so a
  4 KB JSON key — client-controlled — took 79 seconds on a per-request path. Found
  by the new fuzz test while it was being written.
- **An unpaired UTF-16 surrogate anywhere in a request was an unauthenticated
  DoS.** It passed redaction and was then refused by the canonicalizer, so one
  request could 500 every audited endpoint, or under `fail_open` silently suppress
  its own audit record — letting an attacker choose whether to be logged. Fixed at
  the single choke point all three integrations share.
- **`verify` wrote to the store it was verifying.** Every read-only command
  provisioned tables, triggers and indexes, so it could not run as a read-only
  role or against a hot standby — and a mistyped DSN reported "your ledger is
  empty" (exit 1, an integrity verdict) instead of "that store does not exist"
  (exit 2). Backends now accept `create=False`; SQLite additionally sets
  `query_only` and skips the WAL pragma, which is itself a write.
- **`verify` on `mongodb://host/database` ignored the database** and reported
  VERIFIED, exit 0, about an entirely different ledger.
- **No CLI command could address a non-default table**, so `signledger verify ||
  alert` was unavailable to anyone who used the documented `entries_table=`
  argument. All four projects hit this.
- **Django recorded `status: 200` for requests that returned 500.** An `AuditIndex`
  mirror-write failure re-raised *after* the ledger entry was already permanent,
  making the tamper-evident log assert an outcome the client never received.
- Opening a PostgreSQL or MongoDB store silently re-armed dropped triggers,
  validators and indexes, repairing the evidence that someone had removed them.
  Both now report rather than repair, matching SQLite.
- `USER_RESOLVER` accepted only a live callable while `LEDGER_FACTORY` accepted a
  dotted path, so the natural settings idiom passed `manage.py check` and then
  raised `TypeError` on every request. Both forms now work, with a system check.
- `export` exited 0 on a ledger that failed verification, while `checkpoint`
  correctly refused.
- A genuine circular import between `crypto.canonical`, `core.exceptions` and
  `core.entry`, live but masked by package `__init__` ordering. Reordering two
  lines would have broken every `import signledger.crypto.*` at once.
- The Django integration's migration was missing from the built wheel, so
  `manage.py migrate` created no table and every audited request failed under the
  default fail-closed policy.
- The Django admin registration the models promised but never shipped.

### Testing

3,691 tests. 100% line **and** branch coverage across every module, measured
against live PostgreSQL and MongoDB rather than mocks — CI fails if any database
test is skipped, because a skipped one is indistinguishable from a passing one in
the summary line. `ruff`, `mypy --strict` and `bandit` are clean. Property-based
tests use Hypothesis; mutation testing runs in CI.

Two things the test suite alone did not catch, and what now covers them:

- Mutation testing over the crypto core found four security-relevant gaps that
  100% coverage did not — signature malleability, an off-by-one in Merkle
  inclusion, a consistency-proof size guard, and two canonical-number boundaries.
  Coverage proves a line ran; mutation proves an assertion would notice it
  changing.
- Four real consumer projects, installing the wheel and following only the
  published documentation, found everything in **Fixed** above. A green unit suite
  is not evidence that a deployment works, and neither is a green suite evidence
  that anyone can figure out how to use the thing.

Every fenced `python` block in the README — including all the framework wiring —
is executed on every test run.

Stated plainly because the last release did not: 1.0.0 shipped **zero** tests
while its README claimed "100% test coverage".

## [1.0.0] — 2025-10-21 — WITHDRAWN

Do not use. See [SECURITY.md](SECURITY.md).

[2.0.0]: https://github.com/0xV8/signledger/releases/tag/v2.0.0
[1.0.0]: https://github.com/0xV8/signledger/releases/tag/v1.0.0
