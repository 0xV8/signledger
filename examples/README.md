# SignLedger examples

Three runnable examples. Each one is executed end to end by `tests/test_examples.py` on
every test run, and CI fails if any of them stops working or stops printing what it
promises.

That test exists because v1.0.0 shipped three examples and all three crashed on first
use. An example nobody runs is an example nobody knows is broken.

```bash
python examples/quickstart.py
python examples/banking_sox.py
python examples/healthcare_hipaa.py
```

No arguments, no setup, no database to provision. Each example builds everything it needs
in a temporary directory and deletes it on exit, so you can run them repeatedly and they
leave nothing behind.

Only `quickstart.py` runs on a bare install. The other two need the signing extra:

```bash
pip install 'signledger[signatures]'
```

---

## The assurance ladder

SignLedger reports what a deployment's evidence is actually worth, computed at runtime by
`Ledger.assurance_level()`. It is never asserted in prose and never hardcoded. The three
examples deliberately land on three different rungs, and each says out loud what its rung
does and does not buy:

| Example | Reaches | What that means |
|---|---|---|
| `quickstart.py` | `chain_only` | Accidental corruption and naive edits are detected. Someone with write access can still rewrite history and recompute every digest. |
| `healthcare_hipaa.py` | `signed` | Entries cannot be forged or altered without the signing key. The newest entries can still be **deleted** without trace. |
| `banking_sox.py` | `worm` | Signed, anchored outside the database, and the store itself refuses `UPDATE` and `DELETE`. |

A bare hash chain is not tamper-evidence. SHA-256 is public, so the chain alone only ever
proves internal consistency. The two things that actually stop an insider are a secret
they cannot forge (signatures) and a witness they do not control (an external anchor).

---

## 1. `quickstart.py` — append, verify, get caught

The five-minute tour, on SQLite.

It appends a handful of payment events, verifies the chain, and then attacks it twice:

1. A plain `UPDATE` against the entries table. SQLite **refuses it** — the schema carries
   `BEFORE UPDATE` and `BEFORE DELETE` triggers, so the rows are append-only at the
   database level.
2. The attack that works: drop the trigger first, then rewrite a refund amount. The row
   ends up looking completely ordinary to anyone reading the table.

Verification then reports `bad_hash` against the exact sequence number, because the digest
is recomputed from the row rather than trusted from it.

Read this one first. It is also the shortest path to understanding why the other two
examples bother with keys and anchors.

---

## 2. `banking_sox.py` — signatures, checkpoints, external anchoring

A SOX-style financial audit trail. Deliberately a script and not a web service: v1's
flagship example was a Flask app that returned HTTP 500 from every endpoint while bound to
every network interface with the debug console enabled. A script runs in CI, so it stays
working.

Demonstrates:

- **A signing key loaded from a file**, as an encrypted PKCS#8 PEM, the way it would arrive
  from a secrets mount. Public keys are resolved from a directory by `FileKeyResolver`,
  which indexes them by fingerprint — so renaming a key file cannot re-point a `key_id`.
- **Ledger-first writes.** The balance moves only *after* the append succeeds. v1 moved the
  money first, so a failed append left funds transferred with no record of it.
- **Refusals are logged too.** An overdraw attempt is an auditable event; a control that
  only records its successes cannot be shown to have rejected anything.
- **Bulk posting** of end-of-day fees through `BatchProcessor`, including what an atomic
  batch does when one item in the submission is unusable (nothing is committed).
- **Checkpointing and anchoring** to a file kept outside the database directory. This is
  the control that makes a truncated tail visible. In production the anchor belongs on a
  different host, under a different account, or with a notarization service — the point is
  segregation of duties, not the file format.

It reaches `worm` and then explains precisely how much that is worth: the append-only
trigger stops the application and a careless operator, but someone with file-level write
access can drop it in one statement. What survives that attacker is the signature they
cannot forge and the anchor they do not control.

---

## 3. `healthcare_hipaa.py` — PHI access logging, redacted

HIPAA-style audit logging for access to Protected Health Information, under
45 CFR 164.312(b).

The interesting work happens **before `append()` is ever called**. SignLedger redacts
nothing on your behalf — it preserves exactly what you hand it, forever, in a store built
to make deletion detectable. That is the whole reason the filtering has to happen at the
call site.

The example prints both halves side by side:

- **Captured:** who accessed the record, their role, the action, the purpose of use, the
  field *names* touched, and the outcome.
- **Not captured:** the medical record number, name, date of birth, SSN, diagnosis,
  medications and clinical notes.

Entries carry a pseudonymous subject id — an HMAC of the medical record number under a key
that lives with the covered entity and never enters the log. Every access to one patient
stays correlatable, which is what makes an access history useful, while the log on its own
cannot name anybody.

It then scans every stored payload for the known PHI values and proves none of them
leaked, rather than asserting it in a comment.

---

## Using the bulk path

`BatchProcessor` (shown in `banking_sox.py`) validates, hashes and signs an entire batch
before writing anything, so a bad item aborts the whole submission with nothing committed:

```python
from signledger import Ledger
from signledger.core.batch import BatchItem, BatchProcessor

with Ledger() as ledger:
    result = BatchProcessor(ledger).append_batch([
        {"event": "fee.posted", "account": "ACC-1001", "amount": "2.50"},
        BatchItem(data={"event": "interest.accrued"}, metadata={"run": "eod"}),
    ])

if not result.ok:
    for letter in result.dead_letters:
        print(f"item {letter.index} refused: {letter.error} - {letter.message}")
```

A bare `dict` is always the entry's `data`. Use `BatchItem` to attach metadata, pin a
timestamp or supply your own `entry_id`.

There is no parallel mode, and there will not be one. A hash chain is serial at the tip —
entry *n* cannot exist until entry *n-1* has a digest — so extra threads can only contend
for it. v1 shipped the flag anyway and it measured 0.67x a plain `append` loop.
