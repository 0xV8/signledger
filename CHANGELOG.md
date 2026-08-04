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
- **Checkpoints are verified, not believed.** A checkpoint is the only record that
  sees a truncated tail, which makes it the next thing an attacker goes for — and
  it was signed on the way out and taken on sight on the way back in.
  Verification now resolves the `key_id`, checks the signature over the signing
  payload, requires the `ledger_id` to be this chain's and the `format_version` to
  be one this build can read, and recomputes the `merkle_root` over the entries up
  to the checkpointed sequence. Six new `FailureKind` members name which part gave
  way, instead of collapsing into `head_mismatch` and blaming the chain for a
  fault in the record being compared against it.
- **Both copies of a checkpoint are read, and the newer wins.** Preferring the
  anchor unconditionally made anyone who could reach it a downgrade oracle:
  replace it with an older, genuinely signed checkpoint and the store's
  contradicting copy was discarded unread. [SECURITY.md](SECURITY.md) states the
  limit that remains — an attacker holding *both* the store and the anchor can
  roll both back with nothing to detect it, because what is left is a true
  statement about a chain that really did exist.
- **The audit console is excluded from the trail it serves.** On all three
  integrations the read-only console was installed on the same application as the
  auditor and was never excluded from it, so every read of the ledger was a write
  to it: `/verify` answered `entries_verified=5` while appending the sixth, and a
  caller with no credential appended a signed, unprunable row on every rejected
  probe — against the docstring that requires an authenticator precisely because an
  open endpoint is a denial-of-service surface.
- **`skip_paths` is matched by path segment.** All three integrations used a bare
  `str.startswith`, so with the value the README recommends —
  `skip_paths=("/health",)` — a request to `/healthz-admin-secret` was served with
  real data and written nowhere at all.

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
- Streamed responses settle. The entry written when the status line goes out says
  `completion="provisional"`, because the status is a promise at that moment and
  the generator may still raise on its third chunk; a second entry
  (`kind="response.completion"`, same `request_id`) records how it actually ended.
  All three integrations agree on which responses get one, and
  `tests/test_contrib_shared.py` exists to keep that true.
- Django runs under ASGI, and an async streamed response settles there too. The
  hook is not the WSGI one and this was established by execution rather than by
  reading: `send_response` closes Django's own iterator wrapper rather than ours
  and awaits `response.close()` only afterwards, so on the one path where a
  settlement matters most there is no close call at all. The outcome is recorded
  from inside the iteration, in a `finally`, and put back on a thread because it
  writes an ORM row.
- Migration `0002` gives the Django `AuditIndex` mirror the fields this release
  added — `kind`, `completion` and `request_id` — and makes `status` nullable,
  which the record builder has always typed it as. `models.py` is documented as
  the queryable mirror and ships a read-only admin, so the ORM is the surface
  Django users are told to use, and it could express none of those fields.
- A Django system check warns when `RedactionPolicy.max_body_bytes` and
  `DATA_UPLOAD_MAX_MEMORY_SIZE` disagree. It is a warning and not an error because
  the configuration is survivable, and a check that can stop `manage.py migrate`
  has to be reserved for the ones that are not.
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
- Settlement entries go through `build_audit_record` like every other entry. Both
  `_settle` implementations hand-assembled a dict, so a
  `kind="response.completion"` row carried no `completion`, `client_ip`, `headers`,
  `query`, `body` or `user` — refuting the identical comment above both of them
  promising completion is never absent so an auditor need not know which framework
  wrote the entry. It also skipped `redact_payload`, the one choke point that
  closes the unpaired-surrogate hole, on exactly the entries written after the
  response has gone and a refused write has nowhere left to be reported.

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

#### Compatibility across the supported matrix

Every item here was found by CI rather than locally, and each was the same shape:
something true in one environment and assumed universal. They are listed first
because they are the ones a consumer on a different Python or a different FastAPI
would have hit.

- **The FastAPI inspection console stopped excluding itself on FastAPI 0.141.**
  The exclusion found its own mount by scanning `app.routes` for a stamped
  endpoint, which assumes `include_router` copies an included router's routes up
  into the application's table. FastAPI 0.141 with Starlette 1.3 leaves a wrapper
  object holding them instead, so the scan found no console, concluded there was
  none, and audited every unrouted probe against it — which is the defect the
  exclusion exists to prevent. The mount is now resolved through `url_path_for`,
  public API that both generations support, and the three inspection routes carry
  namespaced names so an application route called `status` cannot move the
  console's apparent mount. Verified against Starlette 0.52/FastAPI 0.135 and
  Starlette 1.3/FastAPI 0.141.
- **`uvicorn` was imported by the test suite and declared by nothing.** It passed
  wherever it happened to be installed and failed collection on all six supported
  Python versions and all three operating systems at once. It is now a declared
  `test` dependency.
- **The coverage gate failed on Python 3.9 for a branch 3.9 cannot measure.**
  coverage.py does not record the normal-exit arc of an `async for` inside an
  async generator on 3.9, so the same commit measured 100% on 3.11 and 99.98%
  there. The behaviour is reachable and tested on every version; the arc carries a
  narrow `no branch` pragma documented as the measurement workaround it is, rather
  than lowering the gate.


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
- **Flask recorded every error response as provisional and then settled it.** The
  predicate was `Response.is_streamed`, which werkzeug defines as "`len()` raises",
  and `make_response` runs every `HTTPException` through `force_type`, which swaps
  the body list for a `ClosingIterator`. So every 400, 404, 405 and 500 from an
  error handler was two signed rows from any anonymous caller hitting bad URLs, the
  first of them untrue because the body went out under a `Content-Length`. A
  declared `Content-Length` is the test now, which is what FastAPI already applied.
- **A Flask exception that propagated left no record at all.** `after_request` does
  not run when an exception leaves the application, and `TESTING`, `DEBUG` and
  `PROPAGATE_EXCEPTIONS` each arrange that. Measured over a real socket: the client
  got a 500, the ledger gained nothing, and `check_integrity()` reported ok on a
  chain with no memory of the request. `fail_closed` cannot help, because the
  failure is upstream of the audit write; `teardown_request` is the backstop, and
  it fires only when there is an exception and nothing has been recorded yet.
- **`error` reached `build_audit_record` from one Flask call site out of three**, so
  `WHERE error IS NOT NULL` was structurally empty however many times a view had
  blown up — even the handled 500 that did earn an entry named no cause.
- **A Flask client that hung up mid-stream left a provisional entry forever.**
  werkzeug puts the request-socket drain and `application_iter.close()` in one
  `finally` block, drain first, so a peer RST makes the drain raise and `close()`
  is never reached. Measured against a real server: a 400×64 KB stream, the client
  reading 4 KB and then closing with `SO_LINGER(1, 0)`, left exactly one
  `completion="provisional"` entry for as long as the ledger existed. A finalizer
  on the body wrapper is the fallback, routed through the `close()` that already
  refuses to settle twice — a backstop rather than a guarantee, since an
  implementation without reference counting settles whenever it next collects.
- **A streamed Django response asserted a success the client never received.** A
  `StreamingHttpResponse` whose generator raised on its second chunk produced one
  entry reading `status=200`, `completion="final"`, `error=None`, and under
  `wsgiref` the client sees a clean EOF, so both ends believed it.
- **Django's own body limit produced a 400 that nothing audited.**
  `RedactionPolicy.max_body_bytes` and `DATA_UPLOAD_MAX_MEMORY_SIZE` are
  independent numbers describing the same limit with nothing comparing them, and
  the line that reads the body runs before `get_response`, so the `except` in
  `__call__` never saw the failure. Measured with
  `DATA_UPLOAD_MAX_MEMORY_SIZE=1024` and a 2 KB JSON POST: three requests produced
  two ledger entries, and the 400 the auditor itself had caused had none at all.
  Capture degrades to a marker, which is what redaction already does for every
  other refusal, and the request is served as it would have been with no audit
  middleware installed.
- **A chunked upload and an empty GET recorded the same marker**, so a client could
  switch capture off for itself with `Transfer-Encoding: chunked` and leave an entry
  indistinguishable from the millions of GETs around it — reachable on gunicorn and
  uWSGI, which dechunk without setting `CONTENT_LENGTH`.
- **The FastAPI inspection router audited itself**, under a plain application, under
  `app.mount("/api", subapp)` and under `FastAPI(root_path="/gw")` alike. The
  exclusion is decided after routing, from the endpoint that answered, because
  `route.path` is application-relative and `scope["path"]` is not: deriving a prefix
  up front and comparing the two silently excludes nothing under a mount, while the
  deployer has been told the exclusion is automatic and so does not set `skip_paths`
  either.
- **The coverage gate measured a denominator nobody had checked.** Its
  `exclude_lines` carried a bare three-dot regex, and coverage.py drops the whole
  multi-line construct a matching line belongs to — PEP 484 signatures contain a
  literal ellipsis, so the rule matched the `def` line of ordinary functions and
  deleted their bodies from measurement. The entire `audited()` decorator went that
  way. The rule is anchored to a line that is nothing but an ellipsis.
- **The mutation gate could never have run**, in four independent ways. Its baseline
  died because a test resolved the repository root by `parent.parent` while mutmut
  copies only source and tests into `mutants/`. The guard counted survivors and
  demanded at least a hundred of them, because `mutmut results` prints only
  non-killed mutants — so a perfect sweep printed nothing, scored zero and failed.
  The all-option it needed is not a flag and a bare `--all` exits with an error. And
  the exit-code tolerance let 1 through on the stated grounds that mutmut exits
  non-zero when mutants survive, which is false: it returns 0 with every mutant
  alive and uses 1 for setup failures, which is the failure this repository had. The
  replacement gates on a mutation score counted from the per-mutant result lines,
  and was checked against a real sweep.
- Two CI steps piped into `tee` without `pipefail`. GitHub runs `bash -e {0}`, and
  `-e` sees only the last command of a pipeline, so a failing pytest and a failing
  wheel build both reached their `grep` and passed.

### Testing

4,087 tests, of which 653 need a live PostgreSQL or MongoDB and are marked as
such. Coverage is **gated** at 100% line and branch across every module, measured
with all four backends live rather than against mocks — CI fails if any database
test is skipped, because a skipped one is indistinguishable from a passing one in
the summary line. Both counts are `pytest --collect-only` on this tree rather
than a remembered figure; the gate is `fail_under = 100` in `pyproject.toml`, so
what the release ships is whatever that gate lets through, and it is the gate
rather than a number in this file that is worth reading. `ruff`,
`mypy --strict` and `bandit` are clean. Property-based tests use Hypothesis;
mutation testing gates on a score.

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

[unreleased]: https://github.com/0xV8/signledger/compare/v2.0.0...HEAD
[2.0.0]: https://github.com/0xV8/signledger/releases/tag/v2.0.0
[1.0.0]: https://github.com/0xV8/signledger/releases/tag/v1.0.0
