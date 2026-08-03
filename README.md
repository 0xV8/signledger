# SignLedger

A tamper-evident, append-only audit log for Python.

SignLedger answers one question: **has this log been altered since it was written?**

It does not claim to answer that unconditionally, because no library can. How much the
answer is worth depends on how you deploy it — so SignLedger *computes* that and tells
you, instead of asserting it in prose.

```bash
pip install signledger
```

> **If you are on 1.0.0, upgrade.** 1.0.0 is withdrawn: it did not provide the
> tamper-evidence it documented. On SQLite and PostgreSQL — the documented default —
> it built no hash chain at all, and its signatures were never verified.
> [SECURITY.md](https://github.com/0xV8/signledger/blob/main/SECURITY.md) explains what to assume about a log written by it, and why
> a 1.0.0 ledger cannot be migrated with its integrity intact — it never had any.

## Quick start

```python
from signledger import Ledger

with Ledger() as ledger:
    ledger.append({"actor": "alice", "action": "deleted-record", "record_id": 42})
    ledger.append({"actor": "bob", "action": "changed-role", "target": "carol"})

    report = ledger.check_integrity()

print(report.ok)                # True
print(report.entries_verified)  # 2
print(report.assurance.value)   # chain_only
```

`Ledger()` with no arguments keeps entries in memory, which is useful for a test and
useless for an audit trail. Point it at a store to keep anything:

```python
import os, tempfile
from signledger import Ledger
from signledger.backends import SQLiteBackend

path = os.path.join(tempfile.mkdtemp(), "audit.db")

with Ledger(SQLiteBackend(path)) as ledger:
    entry = ledger.append({"actor": "alice", "action": "login"})

print(entry.sequence)  # 0
```

## Read this before relying on it

A hash chain on its own is **not** tamper evidence. SHA-256 is a public function, so
anyone who can write to your database can edit a row and recompute every hash after it.
The chain verifies perfectly afterwards.

Only two things actually stop that — a secret the attacker cannot forge, or a record kept
somewhere they cannot reach. SignLedger reports which of them you have:

| Level | What you have | What it detects |
|---|---|---|
| `none` | chain broken, or nothing verified | — |
| `chain_only` | hash chain only | accidental corruption, and naive edits |
| `signed` | every entry signed with a key the attacker does not hold | a database insider rewriting rows |
| `anchored` | signed, **plus** a head published outside the database | deletion of the tail, which nothing else can see |
| `worm` | anchored, **plus** a store that refuses `UPDATE`/`DELETE` | rewriting at all |

```python
from signledger import AssuranceLevel, Ledger

with Ledger() as ledger:
    ledger.append({"event": "something happened"})
    level = ledger.assurance_level()

assert level is AssuranceLevel.CHAIN_ONLY
```

If `assurance_level()` returns `chain_only`, someone with write access to your database
can rewrite your audit log undetectably. That is not a bug — it is what a bare chain is.
The rest of this README is about getting above it.

### Signing

A signature is a secret the attacker cannot forge. The key's identity is part of the
hashed preimage, so it cannot be swapped without breaking the chain.

```python
from signledger import AssuranceLevel, Ledger
from signledger.crypto.signatures import Ed25519Signer

signer = Ed25519Signer.generate()

with Ledger(enable_signatures=True, signer=signer) as ledger:
    ledger.append({"actor": "alice", "action": "wired-funds", "amount": "10000.00"})
    level = ledger.assurance_level()

assert level is AssuranceLevel.SIGNED
```

Signing needs the optional extra:

```bash
pip install 'signledger[signatures]'
```

`Ledger(enable_signatures=True)` without a signer fails at construction, not on the first
append.

### Anchoring

Signatures do not detect **deletion**. Remove the last 200 entries and what remains is a
shorter log that is internally flawless — every hash and every signature still checks out.
Nothing inside the database can tell you those entries ever existed.

An anchor is a record of the chain head kept where the attacker does not control it.

```python
import os, tempfile
from signledger import Ledger
from signledger.anchor import FileAnchorSink
from signledger.backends import SQLiteBackend

directory = tempfile.mkdtemp()
store = os.path.join(directory, "audit.db")
anchors = os.path.join(directory, "anchors.jsonl")

with Ledger(SQLiteBackend(store), anchor_sink=FileAnchorSink(anchors)) as ledger:
    for index in range(5):
        ledger.append({"event": "payment", "index": index})
    checkpoint = ledger.checkpoint()

print(checkpoint.entry_count)  # 5
```

Take a checkpoint periodically and keep it somewhere else — another host, an append-only
bucket, a transparency log. `check_integrity()` compares the head against the newest
anchored checkpoint, and a truncated tail then shows up as `HEAD_MISMATCH` and
`COUNT_MISMATCH`.

## Verification

`check_integrity()` never raises because the log is damaged; it returns a report. It
collects **every** failure rather than stopping at the first, because someone looking at a
corrupted audit log needs the whole picture.

```python
from signledger import Ledger
from signledger.backends import MemoryBackend

backend = MemoryBackend()
with Ledger(backend) as ledger:
    ledger.append({"event": "one"})
    ledger.append({"event": "two"})

    # Rewrite a stored row behind the ledger's back, the way an insider would.
    stored = backend._entries[0]
    backend._entries[0] = stored.model_copy(update={"data": {"event": "tampered"}})

    report = ledger.check_integrity()

print(report.ok)  # False
print(sorted({failure.kind.value for failure in report.failures}))
```

Failures name what broke and where: `broken_link`, `bad_hash`, `bad_signature`,
`missing_signature`, `sequence_gap`, `genesis_mismatch`, `head_mismatch`,
`count_mismatch`, `unknown_key`, `format_unsupported`.

An empty range is reported as **not ok**. A verifier that returns success having checked
nothing is worse than no verifier.

## Storage backends

```python
from signledger.backends import MemoryBackend, SQLiteBackend

memory = MemoryBackend()            # tests only; nothing survives the process
sqlite = SQLiteBackend(":memory:")  # or a path, for WAL + synchronous=FULL on disk
memory.close()
sqlite.close()
```

PostgreSQL and MongoDB need their own extras, and are imported by path so a base install
never needs their drivers:

```bash
pip install 'signledger[postgresql]'   # signledger.backends.postgresql.PostgresBackend
pip install 'signledger[mongodb]'      # signledger.backends.mongodb.MongoBackend
```

Every backend runs the same conformance suite, so they behave identically. Ask one what it
actually guarantees rather than assuming:

```python
from signledger.backends import SQLiteBackend

backend = SQLiteBackend(":memory:")
print(sorted(backend.capabilities()))
backend.close()
```

An append is one database transaction — the tip read and the insert together — plus
`UNIQUE` constraints on `sequence` and `previous_hash`, and a bounded retry on conflict.
Two workers cannot fork the chain.

## Batching

```python
from signledger import Ledger
from signledger.core.batch import BatchItem, BatchProcessor

with Ledger() as ledger:
    result = BatchProcessor(ledger).append_batch([
        {"event": "one"},
        {"event": "two"},
        BatchItem(data={"event": "three"}, metadata={"source": "importer"}),
    ])

print(result.ok, result.committed)  # True 3
```

Everything a caller can get wrong — an unencodable payload, a naive timestamp, a repeated
`entry_id` — is checked before the first write, so with `atomic=True` (the default) one
bad item means nothing at all is written.

Honest number: batching is roughly **1.1–1.2x** faster than a loop of `append()`, not an
order of magnitude. Most of the cost is canonical encoding, which both paths pay.

## What goes in an entry

Payloads must be JSON-native: `dict`, `list`, `str`, `int`, `bool`, `None`, `float`.
`Decimal`, `datetime`, `bytes`, `set` and `tuple` are **rejected**, with the JSON path in
the message:

```python
from decimal import Decimal
from signledger import Ledger
from signledger.core.exceptions import CanonicalizationError

with Ledger() as ledger:
    try:
        ledger.append({"amount": Decimal("1.50")})
    except CanonicalizationError as err:
        print(err)  # data.amount: Decimal is not JSON-native; convert to str
```

That strictness is deliberate: silently coercing values is how two different payloads end
up with one hash. Convert to `str` and keep the exactness.

Error messages never contain your payload, your connection string or your credentials.
They carry field names, JSON paths, sequence numbers and type names only — an audit
library's exceptions end up in logs its payloads were deliberately kept out of.

Every error derives from `SignLedgerError`:

```python
from signledger import Ledger, SignLedgerError

with Ledger() as ledger:
    try:
        ledger.append({})               # data must not be empty
    except SignLedgerError as err:
        print(type(err).__name__)       # ValidationError
```

## Command line

```bash
signledger verify -b sqlite:audit.db
signledger stats -b sqlite:audit.db
signledger inspect -b sqlite:audit.db --forks
signledger checkpoint -b sqlite:audit.db --anchor anchors.jsonl
signledger export -b sqlite:audit.db -o bundle.json
signledger verify-bundle bundle.json
```

Commands that make an integrity claim refuse to run on a ledger that does not verify:
publishing a checkpoint over a rewritten chain would launder the damage into the one record
an auditor is meant to trust.

### Exit codes

The contract with `signledger verify || alert`. The distinction between 1 and 2 is the
whole point — an operator must be able to tell *"the log was tampered with"* from *"I typed
the DSN wrong"*, and until 2.0.0 they could not.

| Code | Meaning |
|---|---|
| `0` | it verified |
| `1` | **an integrity verdict**: tampering, truncation, a broken link — or an empty store, because a verifier that reports success having checked nothing is worse than no verifier |
| `2` | **operational**: the store does not exist, a bad URI, a missing driver, an unreadable key |

A store that does not exist is `2`, never `1`. It used to be answered by provisioning an
empty ledger and reporting that it held nothing — telling an operator their audit log was
gone when they had mistyped a name.

### Reading a store you must not write to

Every read-only command (`verify`, `stats`, `inspect`, `export`, `anchor --show`) opens the
store with `create=False`, which provisions nothing at all: no tables, no triggers, no
indexes, and on SQLite not even `PRAGMA journal_mode = WAL`, which is itself a write to the
database header. So an integrity check can run as a read-only database role, against a hot
standby, or against a copy on read-only media — which is what an auditor is usually handed.

The same argument is available directly:

```python
import os, tempfile
from signledger import Ledger
from signledger.backends import SQLiteBackend

path = os.path.join(tempfile.mkdtemp(), "audit.db")
with Ledger(SQLiteBackend(path)) as writer:
    writer.append({"event": "recorded"})

reader = SQLiteBackend(path, create=False)
with Ledger(reader) as auditor:
    print(auditor.check_integrity().ok)  # True
```

A `create=False` backend refuses `append()`, and refuses to open a store that is not there
rather than bringing one into being. `memory://` has nothing to read, so it is refused
outright.

Two honest caveats, because "reads nothing" is a stronger claim than SQLite lets anyone
make:

- **The directory may still gain a `-wal` and `-shm`.** Opening a WAL database read-only
  in a *writable* directory lets SQLite build its own wal-index beside the file. That is
  SQLite's doing, not the library's; the database file is byte-identical afterwards and no
  stored row changes. On genuinely read-only media nothing is created at all.
- **A store whose `-wal` cannot be read is refused, not guessed at.** If a snapshot arrives
  with uncheckpointed commits still in its write-ahead log and the log cannot be replayed,
  reading the `.db` alone would return a valid-looking *prefix* of the chain — it hashes,
  it links, it verifies — while silently omitting everything the log still held.
  `signledger` stops instead, and tells you to checkpoint at the source or copy the store
  somewhere writable.

If you point the CLI at a non-default table, name **both** — `--entries-table` alone now
fails rather than silently provisioning a default-named `ledger_checkpoints` beside it and
reporting success.

## Web framework integration

```bash
pip install 'signledger[flask]'    # signledger.contrib.flask
pip install 'signledger[fastapi]'  # signledger.contrib.fastapi
pip install 'signledger[django]'   # signledger.contrib.django
```

Every example below is taken from a working project that was built against the published
wheel and driven with real traffic under `gunicorn -w 4` or `uvicorn --workers 4`.

One rule applies to all three: **build the `Ledger` inside your application factory, not at
module scope.** A pre-forking server imports your module once and then forks, so a ledger
built at import time hands every worker the same database handle. On PostgreSQL that
corrupts the wire protocol between processes. Build it per worker, after the fork, and pass
the same `ledger_id` to each — see [Deployment](#deployment).

### Flask

```python
from flask import Flask
from signledger import Ledger
from signledger.backends import SQLiteBackend
from signledger.contrib.flask import SignLedgerAudit, audit_blueprint
from signledger.contrib.redaction import RedactionPolicy


def create_app():
    app = Flask(__name__)

    ledger = Ledger(SQLiteBackend("audit.db"), ledger_id="billing-api")

    SignLedgerAudit(
        app,
        ledger=ledger,
        redaction=RedactionPolicy(capture_body=True),
        audit_failure_policy="fail_closed",
        skip_paths=("/health",),
        user_resolver=lambda request: getattr(request, "actor", None),
    )
    return app
```

`audit_failure_policy` is the decision that matters. `fail_closed` (the default) means a
request whose audit entry cannot be written **fails** — the client gets a 500 and nothing
happens unrecorded. `fail_open` serves the request anyway and drops the record. For a
compliance log the first is almost always right; choosing the second should be a decision
someone wrote down.

`skip_paths` keeps health checks out of the ledger. A load balancer polling `/health` twice
a second will otherwise bury every real event.

`audit_blueprint(ledger, authenticator=...)` mounts a read-only console at `/_audit/status`,
`/_audit/entries` and `/_audit/verify`. It exposes hashes, sequences and timestamps — never
payloads. There is deliberately no default authenticator: it is your audit log, and the
library will not guess who may read it.

### FastAPI

```python
from fastapi import FastAPI, Request
from signledger import Ledger
from signledger.backends import SQLiteBackend
from signledger.contrib.fastapi import SignLedgerAuditMiddleware, audit_router
from signledger.contrib.redaction import RedactionPolicy


def create_app():
    ledger = Ledger(SQLiteBackend("audit.db"), ledger_id="payments-api")

    app = FastAPI()
    app.add_middleware(
        SignLedgerAuditMiddleware,
        ledger=ledger,
        redaction=RedactionPolicy(capture_body=True),
        skip_paths=("/health",),
        user_resolver=lambda request: request.headers.get("x-actor"),
    )
    app.include_router(
        audit_router(
            ledger,
            authenticator=lambda request: request.headers.get("x-audit-key") == "…",
        )
    )
    return app
```

`audit_router(ledger, authenticator=...)` mounts the same read-only console as the Flask
blueprint, at `/_audit/status`, `/_audit/entries` and `/_audit/verify`. The requests it
answers are **not** audited, so reading the trail is not a write to it — and that includes
the `401` an unauthenticated caller gets, the `422` from a malformed `limit`, and the `404`
for any unrouted path under its mount, which is the cheapest probe on the application and
the only one needing no credential at all. The exclusion is worked out per request from the
path the router actually answers on, so it holds under a custom `prefix`, under
`include_router(..., prefix=...)`, under `app.mount("/api", subapp)`, under
`FastAPI(root_path="/gw")`, and for a router included after the server started. There is no
`skip_paths` entry to remember, and writing one by hand would be wrong in all five of those
shapes.

A route of your own that merely shares the prefix — `/_audit/dashboard`, or a single-page
catch-all — is still audited: the exclusion covers the router's own endpoints, not every
path that starts with the same characters. An empty `prefix` is refused outright, because a
console at the root cannot be told apart from the site it reports on.

The `audited` decorator records a *business* event rather than an HTTP request, and composes
with the middleware rather than replacing it:

```python
from fastapi import APIRouter
from signledger import Ledger
from signledger.contrib.fastapi import audited

ledger = Ledger()
router = APIRouter()


@router.post("/transfer")
@audited(lambda: ledger, action="funds.transfer")
async def transfer(amount: str) -> dict:
    return {"transferred": amount}


ledger.close()
```

A route carrying both writes **two** entries — one describing the request, one describing the
action. That is intentional, but know that it is happening. Pass a callable (as above) rather
than a `Ledger` instance so the ledger is resolved per request, which is what lets the
decorator work with a per-worker ledger.

Streaming responses are audited when the status line is sent, because a fail-closed entry has
to be written before the first byte leaves. At that moment the status is a *promise*, not a
fact — the generator may still raise on its third chunk — so the entry says so with
`completion="provisional"`, and a second entry (`kind="response.completion"`, same
`request_id`) records how it actually ended: `outcome="complete"`, or `outcome="incomplete"`
plus `error` naming the exception if one was raised.

**All three integrations behave identically here**, which is what
`tests/test_contrib_shared.py` exists to keep true. So an auditor counting successful
responses wants `completion="final"` plus the provisional entries that settled complete —
`status: 200` on a provisional entry means *200 was sent*, not *200 was delivered*.

A response whose body is already built is `completion="final"` and gets no second entry. So
is one whose body a server will never send (`HEAD`, `204`, `304`), and so is a file response
sent under a declared `Content-Length`, because a receiver can detect a truncated one itself
as a short read.

### Django

```python
# settings.py
INSTALLED_APPS = [
    ...,
    "signledger.contrib.django",
]

MIDDLEWARE = [
    ...,
    "signledger.contrib.django.middleware.SignLedgerAuditMiddleware",
]

SIGNLEDGER = {
    "LEDGER_FACTORY": "myapp.audit.build_ledger",   # required, no default
    "USER_RESOLVER": "myapp.audit.acting_user",     # optional
    "CAPTURE_BODY": True,
    "AUDIT_FAILURE_POLICY": "fail_closed",
    "TRUSTED_PROXY_COUNT": 0,
}
```

```python
# myapp/audit.py
from signledger import Ledger
from signledger.backends import SQLiteBackend


def build_ledger():
    return Ledger(SQLiteBackend("audit.db"), ledger_id="orders-service")


def acting_user(request):
    user = getattr(request, "user", None)
    if user is None or not user.is_authenticated:
        return None
    return user.get_username()
```

`LEDGER_FACTORY` is **required and has no default**. It and `USER_RESOLVER` both accept a
dotted path or a callable; a dotted path is usually better in `settings.py`, because a
callable forces the import at settings-import time.

Then run migrations — the app ships one, and you should never be asked to generate it:

```bash
python manage.py migrate
python manage.py check     # fails if LEDGER_FACTORY is missing or unimportable
```

`manage.py check` validates the configuration at deploy time rather than letting the first
request discover it. Take it seriously in CI.

The app also creates an `AuditIndex` table. It is a **queryable mirror**, not the evidence —
the ledger is the evidence. It exists so you can find a record with the ORM or the Django
admin (a read-only `ModelAdmin` ships) without walking the chain. An index write that fails
is logged and never fails the request; an entry that is in the ledger but missing from the
index is a stale mirror, not a broken audit trail.

If you repoint `LEDGER_FACTORY` at a fresh store while keeping the same `ledger_id`, the
index rows from the old store are stale and are replaced as new entries arrive.

## What gets redacted, and what does not

Sensitive fields are replaced before anything is written — in the body, the headers **and**
the query string, by one shared rule so the three can never disagree. Matching is by
substring on the field *name*, and covers passwords, tokens, secrets, keys, one-time codes,
crypto material, certificates and common PII identifiers.

```python
from signledger.contrib.redaction import RedactionPolicy

policy = RedactionPolicy()
print(policy.payload({"user": "alice", "password": "hunter2"}))
```

**Request bodies are not captured at all unless you ask.** `capture_body` defaults to
`False`; headers and query strings are always redacted.

Three limits worth understanding before you turn body capture on, because this data goes
somewhere nobody can edit it afterwards:

- **A name denylist can never be complete.** The rule was widened after fuzzing a generated
  corpus of ~12,900 field names found whole categories it missed. Yours may use a name
  nobody anticipated. Add them with
  `RedactionPolicy(extra_sensitive_fields=("customer_email",))`.
- **Free text is never scanned.** A field called `notes` containing *"the customer said
  their password is hunter2"* is stored verbatim. Redaction reads field names, not values.
- **It deliberately over-redacts.** Short stems are matched as whole words, so `shipping`,
  `pinned`, `single`, `business`, `seeded` and `concert` survive. Longer ones are matched as
  substrings, so `keyword`, `keyboard`, `author`, `discard` and `wildcard` **are** blanked.
  For a store nobody can edit afterwards, that is the correct direction to be wrong in.

If a payload must never be stored even redacted, do not send it through the audited path.

**Upgrading an existing ledger:** the rule got wider in 2.0.0, and old entries are not
rewritten — they cannot be. A ledger that spans the upgrade holds the plaintext spelling of
a name like `otp` or `account_number` before the boundary and `[REDACTED]` after it,
permanently. If the earlier entries contain something that should not be there, the fix is
to retire that ledger and start a new one, not to edit it.

## Deployment

Everything below was measured, on **macOS arm64** (CPython 3.12, gunicorn 26.0.0,
uvicorn 0.52, PostgreSQL 14). There is no Linux measurement yet, so nothing here is
claimed for Linux. Topologies that are not in the table were not tested.

### Every worker must be given the same `ledger_id`

`Ledger()` without `ledger_id=` mints one. That is fine for a single process and wrong
for a pre-forked server: identity then depends on which worker reached an empty store
first. Read one stable value from configuration and pass it in every worker.

```python
import os, tempfile
from signledger import Ledger
from signledger.backends import SQLiteBackend

store = os.path.join(tempfile.mkdtemp(), "audit.db")
LEDGER_ID = "billing-api"  # from config, identical in every worker

for worker in range(4):  # what four workers each do, serialised here
    with Ledger(SQLiteBackend(store), ledger_id=LEDGER_ID) as ledger:
        ledger.append({"worker": worker, "event": "request"})

with Ledger(SQLiteBackend(store), ledger_id=LEDGER_ID) as ledger:
    report = ledger.check_integrity()

print(report.ok, report.entries_verified)  # True 4
```

Every clean run in the table below started against a store that already held entry 0.
Deploy the same way: create the ledger once, then start the workers.

### Open the database connection inside the worker, never before the fork

A PostgreSQL connection is not usable from more than one process. Under
`gunicorn --preload` a module-level backend is created in the master and inherited by
every child, which puts several processes on one socket.

That configuration used to fail outright — 139 of 300 requests served, with failed
appends and read timeouts — and it no longer does: the pool became fork-aware, and three
`--preload` runs on the current build came out clean (300/300 at `-w 4`, 300/300 at
`-w 12`, 800/800 at `-w 4`, chains contiguous and verifying). **Follow the rule anyway.**
It is a property of `fork()` rather than of any pooling, so it holds whatever a backend
does internally, and the version that made it safe is one you have to keep:

- with gunicorn, either drop `--preload`, or build the backend in a `post_fork` hook;
- uvicorn `--workers` was clean against PostgreSQL — each worker imports the application
  itself instead of inheriting an open connection.

SQLite is unaffected — `gunicorn --preload -w 8` on SQLite was clean at 800/800 — but
the rule costs nothing there either.

### Verified clean

Chain verified means `check_integrity()` returned ok, sequences were contiguous from 0,
one `ledger_id`, and no duplicate `sequence` or `previous_hash`.

| Server | Workers | Store | Requests | 2xx | Chain |
|---|---|---|---|---|---|
| gunicorn / Flask | `-w 1 --threads 8` | SQLite | 300 | 300 | 301 verified |
| gunicorn / Flask | `-w 4` | SQLite | 300 | 300 | 301 verified |
| gunicorn / Flask | `-w 8` | SQLite | 800 | 800 | 801 verified |
| gunicorn / Flask | `-w 12` | PostgreSQL | 300 | 300 | 301 verified |
| uvicorn / FastAPI | `--workers 4` | SQLite | 300 | 300 | 301 verified |
| uvicorn / FastAPI | `--workers 4` | PostgreSQL | 1000 | 1000 | 1001 verified |
| gunicorn / Django | `-w 4` | PostgreSQL | 300 | 300 | 301 verified |

The Django row also had 300 distinct rows in its `AuditIndex` table, each tying back to a
chain hash.

A cold start — `gunicorn -w 4` against a database file that **did not exist yet** — was
also measured clean at 300/300. It was not, before: two separate defects meant four
workers racing to create one store produced a total outage. Both are fixed and both are
pinned by tests, but seeding the store first is still the cheaper habit.

### Which store

SQLite is a real answer for multi-worker on **one host**: the `-w 4` and `-w 8` rows above
are separate processes writing one file, and the chain came out intact. It stops being an
answer the moment a second host writes — a network filesystem does not give SQLite the
locking it depends on.

PostgreSQL is the answer across hosts, and the only one tested for that.

## Compatibility

SemVer across three independent surfaces: the Python API, the on-disk schema, and the hash
`format_version`. Verification supports every historical `format_version` forever; writing
always uses the newest.

## Limitations

Stated plainly, because a security library that hides these is worse than useless.

- **Without an anchor, tail truncation is undetectable.** Delete the last N entries and
  what remains verifies perfectly. Use `checkpoint()` and keep the result elsewhere.
- **`chain_only` does not survive a database insider.** See the table above.
- **A process compromise beats signing.** Anyone who can read the signing key can produce
  valid entries. Only external anchoring and an HSM/KMS signer help.
- **`Entry.data` is not deeply frozen.** Mutating a returned entry's `data` dict in place
  changes what it hashes to.
- **Timestamps are limited to roughly 1684–2255**, a consequence of encoding them as
  integer microseconds within the ±2^53 safe-integer bound.
- **Merkle consistency proofs** are one element longer than RFC 6962's when the old tree
  size is a power of two, because SignLedger binds the leaf count into the root.
- **On SQLite and PostgreSQL, `anchored` is a damage signal, not a configuration.** Both
  arm their append-only triggers when they create the entries table, so a healthy signed
  and anchored store reports `worm` — the level above. Seeing `anchored` on one of them
  means the triggers are *gone*, which is worth an alert rather than a shrug. MongoDB
  never reports `append_only_enforced`, so `anchored` is its normal ceiling.
- **On Python 3.14, `fork()` in a pre-forking server warns** if the parent is
  multi-threaded — which it is whenever `auto_verify=True` runs the background verifier.
  It is a `DeprecationWarning` from CPython, not from SignLedger, and it does not appear
  on 3.9–3.13. Building the ledger after the fork (which you should do anyway, see
  Deployment) avoids both the warning and the reason for it.

## Development

```bash
python3 -m pytest --cov=signledger --cov-branch --cov-report=term-missing
```

PostgreSQL and MongoDB tests skip unless a live server is reachable; set
`SIGNLEDGER_TEST_POSTGRES_DSN` / `SIGNLEDGER_TEST_MONGO_URI` to point at your own. Nothing
in the suite mocks a database.

Every fenced `python` block in this file is executed by `tests/test_docs_execute.py`, and
every `signledger` line in a `bash` block is parsed and run there too — a renamed flag
breaks the suite instead of breaking the docs.

Coverage is gated at 100% line *and* branch. That figure is only real with both databases
running — without them 582 tests skip and it falls to ~91%, so CI runs them as services
and fails if any database test is skipped.

## Security

Report vulnerabilities privately — see [SECURITY.md](https://github.com/0xV8/signledger/blob/main/SECURITY.md). Release history and the
full account of what was wrong with 1.0.0 are in [CHANGELOG.md](https://github.com/0xV8/signledger/blob/main/CHANGELOG.md).

## License

MIT
