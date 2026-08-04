# SignLedger v2.0.0 production readiness — design

Status: awaiting maintainer review; not yet implemented
Branch: `rebuild/v2.0.0`, baseline commit `6681640`
Date: 2026-08-02

## Why this document exists

An evidence-based audit of `rebuild/v2.0.0` reproduced ~40 defects by execution, across
security, the three framework integrations, the quality gates and release mechanics. Every
finding below was proved by running it, and then independently re-run by a second pass
whose brief was to refute it. Findings that did not survive refutation are recorded in
"Refuted" so nobody spends time on them again.

This repo has a documented history of shipping claims that were never true. Accordingly:
**no claim in this spec may be restated in code comments, docstrings, README or CHANGELOG
until it has been demonstrated by a test that fails without the fix.**

## Scope decisions

Taken by the maintainer before design:

1. **Django `FileResponse` is not wrapped.** It stays on the `sendfile` zero-copy path and
   is recorded `completion="final"`. The limitation is documented rather than hidden.
2. **Flask gets a fallback settlement plus a real socket-level abort test**, replacing the
   existing test that fakes the abort by calling `close()` in-process.
3. **Delivery is blockers-first in reviewable commits**, each verified before the next.
4. **No authorship or tool attribution in any commit message, PR body, changelog entry or
   code comment.**

Taken during design, open to reversal:

5. **Request-body capture stays divergent between frameworks.** FastAPI records
   `[OMITTED:not-read]` where Flask and Django capture the body, because making the ASGI
   middleware actively drain `receive` would change ASGI semantics and can hang against a
   client that sends no body. The `body_reason` marker mechanism exists precisely to name
   this. It becomes documented and pinned by a parity test instead of accidental.

## Workstream A — checkpoint verification

### The defect

`_check_checkpoint` (`signledger/core/verification.py:153-211`) reads only
`checkpoint.sequence`, `head_hash` and `entry_count`. It never touches `signature`,
`key_id`, `merkle_root`, `ledger_id` or `format_version`. Checkpoints are signed at
`core/ledger.py:518` and verified nowhere in product code.

Nine reproduced attacks, all yielding `ok=True`:

| # | Attack | Result | Notes |
|---|--------|--------|-------|
| A1 | Forge anchor JSONL line: garbage signature, unresolvable `key_id` | `ok=True, assurance=anchored` | *Upgrades* assurance from `none` |
| A2 | `UPDATE ledger_checkpoints` — no anchor, no DDL, no key | `ok=True, assurance=signed` | Checkpoint table has no append-only triggers |
| A3 | Roll back to an older **genuine** signed checkpoint | `ok=True, assurance=anchored` | Defeats signature + ledger_id + merkle checks |
| A4 | Delete every checkpoint and the anchor file | `ok=True, assurance=signed` | `verification.py:355` skips the check entirely |
| A5 | Truncate tail, re-append to restore the count | `ok=True, assurance=anchored` | Count check is `stored < entry_count` only |
| A6 | `merkle_root` set to `"00"*32` / `"ff"*32` | Accepted | Never recomputed |
| A7 | Checkpoint from a different `ledger_id`, `format_version=999` | Accepted | Never bound |
| A8 | Roll back the anchor file only, leaving the local table intact | `ok=True` | Anchor preferred unconditionally = downgrade oracle |
| A9 | Drop and re-create the SQLite append-only triggers | `ok=True, assurance=worm` | `capabilities()` still reports `append_only_enforced` |

A2 is threat class B ("DB writer / insider") which `SECURITY.md` places explicitly in
scope. A3 is the load-bearing one: it passes signature verification under the legitimate
public key, matches `ledger_id`, `format_version`, `head_hash`, `entry_count` **and** the
recomputed Merkle root, because the prefix root is stable under growth. Authentication
alone does not fix it.

The shipped CLI agrees with all of them:
`signledger verify -b sqlite:… --anchor … --key <legitimate-pubkey> --json` returns
`{"ok": true, "assurance": "worm", "failures": []}`, exit 0.

### The fix

**A-1. Authenticate the checkpoint.** In `_check_checkpoint`, mirroring how entry
signatures are handled at `verification.py:107-150`:

- resolve `checkpoint.key_id` through the same resolver entries use;
- verify `checkpoint.signature` over `checkpoint_signing_payload(checkpoint)`;
- reject a checkpoint whose `ledger_id` is not the chain under verification;
- reject `format_version > CURRENT_FORMAT_VERSION`;
- recompute the Merkle root over `backend.iter_entries(end_seq=checkpoint.sequence)`
  using `chain.algorithm` and compare against `checkpoint.merkle_root`. `end_seq` is
  inclusive and the prefix root is stable under growth — both verified.

**Correction, found while implementing (A-1).** The design originally called for a
bidirectional count check. That is wrong and was dropped: `stored > entry_count` is
ordinary growth past the checkpoint, and failing it would break every live ledger. The
attack it was meant to catch — truncate, then re-append to restore the count — is caught
by the head digest whenever the checkpoint is current, and is a freshness problem, not a
counting one, when the checkpoint is old.

**Correction, found while implementing (A-3).** `CHECKPOINT_ROLLBACK` and
`CHECKPOINT_MISSING` were specified as failure kinds. Both were removed. A publish that
fails leaves the local checkpoint ahead of the anchor, which is *indistinguishable* from a
rollback, so reporting either as an integrity failure turns an anchor-service outage into
`ok=False` — and `tests/test_anchor.py::test_a_checkpoint_the_anchor_service_refused_
confers_no_anchored_assurance` already, correctly, required `SIGNED` in that case. Both
conditions are now expressed as `anchored=False`, so they cost the `ANCHORED`/`WORM`
label rather than claiming damage that did not happen.

**Refinement (A-1).** The Merkle check runs only when the head and count checks pass. A
truncated tail fails all three at once, and a report naming three symptoms of one cause is
harder to act on than one naming the two that locate it. The Merkle check earns its place
on the case the other two cannot see: a head and count that reconcile over a tree that
does not.

**A-2. Selection becomes `max(local, external)` by sequence.** `ledger.py:589-600` and
`cli.py:567-577` currently prefer the anchor unconditionally. Taking the highest sequence
across both sources means a rollback requires compromising *both* stores. Relatedly,
`FileAnchorSink.fetch_latest` returns the highest-sequence entry for the `ledger_id`
rather than the last line, so deleting lines no longer wins.

**A-3. Absence of evidence becomes a failure.** `verification.py:355` is
`if checkpoint is not None:` — no checkpoint means no check and no failure. When an anchor
sink is configured and yields nothing, that is now `CHECKPOINT_MISSING`.

**A-4. New `FailureKind` members:** `CHECKPOINT_UNSIGNED`, `CHECKPOINT_UNKNOWN_KEY`,
`CHECKPOINT_BAD_SIGNATURE`, `CHECKPOINT_LEDGER_MISMATCH`,
`CHECKPOINT_FORMAT_UNSUPPORTED`, `CHECKPOINT_MERKLE_MISMATCH`, `CHECKPOINT_ROLLBACK`,
`CHECKPOINT_MISSING`.

**A-5. Assurance downgrade.** `_assurance()` must refuse `ANCHORED`/`WORM` when any
checkpoint failure is present.

### Documented limitation

An attacker with write access to **both** the anchor and the checkpoint table can still
roll back to a genuine older checkpoint. The defence is operational — an anchor the ledger
host cannot write — not algorithmic. `SECURITY.md` and `docs/ARCHITECTURE.md` must say so
plainly rather than implying the chain self-defends.

Likewise A9: the SQLite append-only triggers are droppable and re-creatable by any writer
of the file, and the `worm` label survives the attack. Document the trigger as a speed bump
against accidental and casual modification, not as WORM in the storage sense.

### Regression tests

`tests/test_threat_model.py` gains one test per attack A1–A9, each asserting the specific
`FailureKind` and that assurance does not exceed `CHAIN_ONLY`. Note that the existing
`test_deleting_the_local_checkpoints_too_does_not_hide_the_truncation` passes today only
because its fixture uses "a path the adversary in these tests does not touch" — it must be
rewritten to touch it.

## Workstream B — framework parity

### B-1. Shared layer (`signledger/contrib/__init__.py`)

Three things are re-implemented per framework today and diverge. They move into the shared
module:

**Settlement records go through `build_audit_record`** via a new `outcome=` parameter.
Today both `_settle()` implementations hand-assemble a dict, so every
`kind="response.completion"` entry lacks `completion`, `client_ip`, `headers`, `query`,
`body` and `user`, and skips the `redact_payload` choke point that
`contrib/__init__.py:150-161` calls "one choke point instead of three". This directly
refutes the identical comments at `fastapi.py:375` and `flask.py:249` claiming `completion`
is "never absent". It is also a hard prerequisite for Django — see B-3.

**One `will_send_body(method, status)` helper** replaces Flask's `_sends_a_body` and
FastAPI's status-only check at `fastapi.py:546`. They disagree on HEAD: a HEAD to a
streaming route yields 2 entries and `provisional` on FastAPI, 1 entry and `final` on Flask
and Django.

**One segment-aware `skip_paths` matcher.** All three integrations use bare
`str.startswith` (`flask.py:190`, `fastapi.py:341`, `django/middleware.py:63`). With the
README's own `skip_paths=("/health",)`, a request to `/healthz-admin-secret` returns 200
with real data and writes **zero** audit entries. New rule:
`path == prefix or path.startswith(prefix.rstrip("/") + "/")`.

### B-2. Flask

| Defect | Site | Fix |
|--------|------|-----|
| Every HTTPException 4xx/5xx written as `provisional` + a spurious settlement | `flask.py:230` | Replace `response.is_streamed` with a materialized-body test. `is_streamed` is `len()` raises, which is true for every `ClosingIterator` that `force_type` produces — measured: 404/405/500 all carry a `Content-Length` yet report `is_streamed=True` |
| A propagated 500 leaves zero entries | `flask.py:213` | `teardown_request` backstop — verified to fire in all five exception configs and to receive the exception |
| `error=` never recorded | `flask.py:233` | Pass it |
| Client hard-disconnect leaves a permanently unsettled provisional | `flask.py:296` | Fallback settlement + socket-level abort test |

On the disconnect: `werkzeug.serving` puts the socket drain and
`application_iter.close()` in the same `finally`, drain first, so a peer RST means `close()`
is never reached and `_SettlingBody.close` never runs. The existing test at
`tests/test_contrib_flask.py:390` hand-calls `iterable.close()` in-process and therefore
passes over the real failure.

Capacity note: the 4xx double-write is also an anonymous append amplifier — `skip_paths`
defaults to `()`, so a scanner writes two signed rows per probe into a store that by design
cannot be pruned.

### B-3. Django

Ordered, because two of these are landmines under the others:

1. **Fix the `_index` try boundary first.** `middleware.py:138-149` builds the row dict —
   including `record["user"]` — *outside* the `try`, while its docstring promises the method
   "must never raise". Verified: calling `_index` with a settlement-shaped record raises an
   uncaught `KeyError: 'user'` straight out of `__call__`. `INDEX_ENTRIES` defaults to
   `True`, so the moment Django writes settlement records every streamed request 500s.
2. **Migration** adding `kind`, `completion` and `request_id` to `AuditIndex`, and making
   `status` nullable (`build_audit_record` types it `Optional[int]`; the column is NOT NULL).
   Without this the ORM mirror and admin — the query surface Django users are told to use —
   cannot express either v2.0.0 field.
3. **Exception capture:** `try/except` around `get_response()` that records `error=` and
   re-raises. **Not** `process_exception` — verified on Django 5.2.16 that it fires *and*
   the exception still propagates, so a hook recording in both places double-counts. It is
   also never called for an exception raised by a middleware below this one.
4. **Streaming settlement:** wrap `response.streaming_content`, excluding `FileResponse`
   per scope decision 1. Assigning a wrapper to `streaming_content` flips `file_to_stream`
   to `None`, which disables the `wsgi.file_wrapper` zero-copy path for every download; and
   where `file_wrapper` *is* used the server iterates the raw file object, so `_complete`
   stays `False` and a fully-delivered file would settle as `incomplete`.
5. **`RequestDataTooBig` in `_body`** (`middleware.py:107`): raised before `get_response`,
   client gets a 400, ledger gets nothing — the audit middleware is itself the cause of an
   unaudited failed request. `RedactionPolicy.max_body_bytes` (default 8192) and
   `DATA_UPLOAD_MAX_MEMORY_SIZE` (default 2621440) are independent with no cross-check.
6. **ASGI:** declare `async_capable`, cover the async path. Driving `ASGIHandler` directly
   shows an async `StreamingHttpResponse` that raises after chunk 1 recorded as
   `status=200, completion=final, error=None` — the identical blocker, with zero tests.
7. **Observability:** unlike Flask, the Django middleware instance is owned by
   `BaseHandler` and unreachable from application code, so `recorder.audit_failures` cannot
   be read in production. Expose it via a logger warning, mirroring `_index` at
   `middleware.py:156`.

### B-4. The audit inspection endpoint audits itself

`audit_blueprint` is registered on the same app as the app-wide hooks and is never
excluded. Measured: every `/_audit/*` request appends an entry, **including unauthenticated
401s**. So every read of the trail mutates the trail, `/verify` cost rises per inspection,
and an anonymous caller appends a signed row on every rejected probe — against a docstring
(`flask.py:373-381`) that justifies the authenticator as anti-DoS. Fix: exclude the
blueprint's own prefix, and document it.

### B-5. Parity suite

New `tests/test_contrib_parity.py`, replacing `tests/test_contrib_django.py:1247-1285`,
which compares key *sets* on `entry[0]` of one 200 GET, Flask vs Django only, FastAPI
absent, via in-process test clients that never consume a streamed body.

Scenario matrix × {Flask, Django, FastAPI}, driven over **real sockets**:
200 GET · 404 · 405 · propagated 500 · completed stream · stream that raises mid-body ·
mid-stream client disconnect · HEAD on a streaming route · 204 · POST with a JSON body.

Invariants asserted across all three: entry count per scenario; `kind` and `completion`
values; presence of `error` on a failed request; the full key set on both entry kinds;
`request_id` correlation; no orphan settlements; no unsettled provisionals except the
documented disconnect case; `check_integrity().ok`.

Positive controls already verified as passing — do not re-litigate: `fail_closed` behaves
identically on all three (500, endpoint body suppressed); concurrency is chain-safe (8×20
requests, 80 streamed, 240 entries, 0 orphans, `ok=True`); query-parameter and header
redaction are already byte-identical across frameworks.

Methodology warning for whoever writes this: the backend hook to stub for failure injection
is `append_atomic`, not `append` (`backends/memory.py:75`). Stubbing `append` yields 200s
and looks like a blocker.

### B-6. Unsettled provisional entries are undiscoverable

There is no way to find one: the CLI exposes verify/stats/inspect/checkpoint/anchor/export/
verify-bundle/migrate and no query surface, and `check_integrity()` ignores them. A worker
killed mid-stream leaves an entry indistinguishable from one still in flight. Minimum for
v2.0.0: a documented retention/timeout semantic for provisional entries. A
`signledger query --unsettled` command is the fuller answer and is deferred unless the
maintainer wants it in scope.

## Workstream C — quality gates

**Coverage.** `pyproject.toml:178`'s `"\\.\\.\\."` matches literal `...` in PEP 484
signatures, and coverage.py drops the whole multi-line `def` construct — including the
entire `audited()` decorator (`fastapi.py:431-506`) and `_call_record()` (509-536).
Measured on the same data file: shipped config `TOTAL 4165 … 920`, with only that rule
removed `TOTAL 4276 … 946`. Fixes:

- narrow to `^\s*\.\.\.\s*$`;
- **also** drop `class .*\(Protocol\):` — same class of over-broad line regex, hides 29
  statements and 36 branches across `backends/base.py`, `crypto/signatures.py`,
  `anchor/sinks.py`, `crypto/keys.py`, all of which are currently covered, so removing it
  costs nothing;
- drop `@overload`, which matches nothing — evidence the list was never validated;
- add a sync raising `@audited` test to cover `fastapi.py:498-500`, the one genuinely
  uncovered path. Verified behaviourally correct when executed, so this is a coverage hole,
  not a latent bug;
- re-measure with `fail_under = 100` and confirm it passes honestly. `backends/mongodb.py`
  reads 38% with 129 missing statements when Mongo is down — the release measurement needs
  a live-Mongo run.

**Mutation gate.** Four independent breakages, all reproduced:

1. Baseline fails — `tests/test_examples.py:22` resolves `REPO_ROOT` via `parent.parent`
   but mutmut copies only `source_paths` + `tests/` + `pyproject.toml`. `mutmut run` on a
   pristine `git archive HEAD` tree exits 1 with zero mutants checked. This is the identical
   defect `6681640` fixed in `test_docs_execute.py` and left in its twin.
2. The guard is inverted — `mutmut results` prints only non-killed mutants, so
   `ci.yml:541`'s `grep -cvE 'not checked'` counts survivors and `ci.yml:544` passes only
   when ≥100 mutants survive. A perfect sweep scores 0 and fails.
3. `ci.yml:526-529` tolerates rc ≤ 1 on the stated grounds that mutmut exits non-zero when
   mutants survive. It does not: mutmut 3.6.0 returns 0 with 360/360 survivors, and uses 1
   for setup failures — so the tolerance swallows exactly the failure this repo has.
4. The job gates on `schedule || workflow_dispatch`, which GitHub honours only from the
   default branch's copy, and `main` has no `.github/`.

Fixes: repair the root resolution; `mutmut results --all true` (bare `--all` errors — the
option takes a value); parse `mutants/mutmut-stats.json` for a real killed/survived count
and gate on a mutation *score*, not a bare `checked` floor against 9,595 mutants; land the
workflow on `main`; raise the `mutmut>=2.4.0` pin to `>=3.0` (2.4.0 does not understand any
of the 3.x config keys and fails with the same rc=1).

**Two more green-over-nothing CI steps.** `ci.yml:126-132` and `ci.yml:459-466` pipe into
`tee` without `set -o pipefail`, and GitHub's default `bash -e` sees only the pipeline's
last command — so a failing pytest run and a failing `python -m build` both pass. Add
`set -o pipefail`.

**Verified clean, do not re-check:** `ruff check .`, `mypy --strict -p signledger`,
`bandit` all pass.

## Workstream D — documentation honesty

Each correction below is `file:line`, claimed → measured.

| Site | Claimed | Measured |
|------|---------|----------|
| `CHANGELOG.md:169` | 3,691 tests | 3,763 |
| `README.md:696`, `ci.yml:67` | 582 skip without DBs | 645 |
| `README.md:695`, `CHANGELOG.md:169` | 100% line and branch | 99.93% |
| `docs/ARCHITECTURE.md:352`, `MANIFEST.in:22` | every python block executes | 0 of ARCHITECTURE's 8 fences; executor covers README.md and examples/README.md only |
| `docs/ARCHITECTURE.md:348` | 16 threads × 500 appends | 16 × 200 |
| `docs/ARCHITECTURE.md:270` | `len(path) == ceil(log2(n))` | false for every ragged tree (n=3 → [2,2,1]); the implementation deliberately does not assert it |
| `README.md:546`, `CHANGELOG.md:125` | 146,000 fuzzed field names | 12,910 (12,803 unique) |
| `crypto/merkle.py:435-441` | 20,000 hostile size pairs | no such harness; a real sweep exists but over sizes 1–17 |
| `pyproject.toml:46-47` | floors measured on every run | measured zero times — no CI job installs `uv`; 44 of 102 packaging tests skip |
| `pyproject.toml:23` | `Development Status :: 4 - Beta` | live 1.0.0 declares `5 - Production/Stable` |
| `README.md:458` | streams recorded at the status they started with | false for all three integrations once B lands |
| `CHANGELOG.md:173` | mutation testing runs in CI | never run — workstream C |

`CHANGELOG.md:6` reads `## [2.0.0] — unreleased` and is grafted into the sdist and is the
target of `Project-URL: Changelog`. Add a test asserting the top CHANGELOG heading matches
`signledger.__version__` and carries a date — nothing checks this today.

### Refuted — do not act on these

- **Werkzeug version determines a 404's entry count.** False. `is_streamed` and
  `run_wsgi_app` are identical across werkzeug 1.0.1, 2.0.3, 2.3.8, 3.0.6, 3.1.8, and
  Flask's `make_response` coercion is identical across 1.1.4 → 3.1.3.
- **The README Deployment table is unbacked.** False. `ci.yml:245-343` is a tracked
  harness: it installs gunicorn, writes a Flask `wsgi.py`, starts `gunicorn -w 4`, drives
  300 POSTs through a 32-thread pool and asserts sequence contiguity.
- **PyPI Project-URLs survive only on a rename redirect.** False. `vipin08` is a live,
  distinct account (id 227882960) from `0xV8` (id 8843310); no redirect is involved.
- **The auditor SQL in the docs does not run against the schema.** Not a real finding —
  no such SQL appears anywhere in the repo. The underlying schema facts are true (tables
  are `ledger_entries`/`ledger_checkpoints`; the record is JSON in a `data` column), so the
  docs should stop describing queries in prose that cannot be run.

## Workstream E — release mechanics

Current state, verified: no tags locally or on origin, no PRs ever, no releases, one
workflow file (`ci.yml`) with no tag or release trigger and no publish step. `main` is the
v1.0.0 tree — 43 files, no `SECURITY.md`, `CHANGELOG.md`, `pyproject.toml`, `docs/` or
`.github/`. All 12 workflow runs are `event=push` on `rebuild/v2.0.0`; the mutation job is
`skipped` in every one. No branch protection, no rulesets. Default workflow permission is
`read`, so a publish job must declare `id-token: write` explicitly.

**1.0.0 is still live and unyanked on PyPI** — `info.yanked=False`, and
`pip download signledger` resolves it today. Its description still reads "production-ready".
This is the only finding currently harming users and it costs a login, not a commit.

`blob/main/…` URLs 404 today, including `Project-URL: Changelog` baked into the wheel
METADATA, which PyPI renders as a sidebar link.

**Do not upload `dist/`.** It is gitignored and predates HEAD by 39 minutes; its sdist
ships the pre-`6681640` `pyproject.toml`, `test_docs_execute.py` and `ARCHITECTURE.md` —
the exact regressions HEAD fixed. The wheel's runtime code is byte-identical to HEAD, so
the damage is confined to the sdist, but the artifact was never inspected by CI (the
`package` job builds, `twine check --strict`s, and discards).

Deliverables: `RELEASING.md`, `CONTRIBUTING.md`, a trusted-publishing workflow (tag
trigger, `id-token: write`, `pypa/gh-action-pypi-publish`), CHANGELOG date and version,
the `5 - Production/Stable` classifier, and the `main` PR prepared but not merged.

Ordered human actions, all left to the maintainer:

1. Yank 1.0.0 on PyPI.
2. Review and merge the `main` PR — closes four problems at once (the withdrawn code stops
   being the default clone, the `blob/main` URLs resolve, the mutation trigger becomes
   reachable, and a PR exists for the first time).
3. Enable branch protection with the existing CI jobs as required checks.
4. Tag `v2.0.0`; the publish workflow builds fresh artifacts and uploads via OIDC.

## Sequencing

Blockers first, each commit verified before the next:

1. Checkpoint verification (A-1 … A-5) + threat-model tests for A1–A9
2. Shared contrib layer (B-1)
3. Flask (B-2)
4. Django, in the stated internal order (B-3)
5. Audit-endpoint self-audit (B-4)
6. Parity suite (B-5) + provisional-entry semantics (B-6)
7. Quality gates (C)
8. Documentation honesty (D)
9. Release preparation (E)

## Verification standard

No commit in this sequence may claim a behaviour that is not demonstrated by a test failing
without it. Test names must describe the behaviour, not the implementation. Where a test
must simulate a server-level event (a disconnect, a `close()`), it must do so over a real
socket — the in-process shortcut is what let the current defects ship.
