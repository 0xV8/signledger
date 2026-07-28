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
> [SECURITY.md](SECURITY.md) explains what to assume about a log written by it, and
> `signledger migrate` carries an existing ledger forward without breaking its history.

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

`verify` exits non-zero when the log does not verify **and** when the store is empty, so it
works as a health check. Commands that make an integrity claim refuse to run on a ledger
that does not verify: publishing a checkpoint over a rewritten chain would launder the
damage into the one record an auditor is meant to trust.

## Web framework integration

```bash
pip install 'signledger[flask]'    # signledger.contrib.flask
pip install 'signledger[fastapi]'  # signledger.contrib.fastapi
pip install 'signledger[django]'   # signledger.contrib.django
```

Request data is redacted before it is written: field names containing `password`, `token`,
`secret`, `ssn`, `card`, `cvv` and friends are replaced — in the body, the headers **and**
the query string. Matching is by substring, so it over-redacts rather than under-redacts.
For a store nobody can edit afterwards, that is the correct direction to be wrong in.

```python
from signledger.contrib.redaction import RedactionPolicy

policy = RedactionPolicy()
print(policy.payload({"user": "alice", "password": "hunter2"}))
```

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

## Development

```bash
python3 -m pytest --cov=signledger --cov-branch --cov-report=term-missing
```

PostgreSQL and MongoDB tests skip unless a live server is reachable; set
`SIGNLEDGER_TEST_POSTGRES_DSN` / `SIGNLEDGER_TEST_MONGO_URI` to point at your own. Nothing
in the suite mocks a database.

Every fenced `python` block in this file is executed by `tests/test_docs_execute.py`.

Coverage is gated at 100% line *and* branch. That figure is only real with both databases
running — without them 582 tests skip and it falls to ~91%, so CI runs them as services
and fails if any database test is skipped.

## Security

Report vulnerabilities privately — see [SECURITY.md](SECURITY.md). Release history and the
full account of what was wrong with 1.0.0 are in [CHANGELOG.md](CHANGELOG.md).

## License

MIT
