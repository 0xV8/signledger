# SignLedger Architecture Contract (v2 / format_version = 2)

This document is **normative**. Every module in `signledger/` implements against it.
If code and this document disagree, the code is wrong.

> Why this file exists: v1.0.0 shipped ~3,200 LOC written against an `Entry.sequence` /
> `Ledger.get_latest()` / `backend.append()` API that never existed in the codebase.
> No module may invent an API. Anything not defined here does not exist.

---

## 1. What SignLedger guarantees

SignLedger is a **tamper-evident append-only log**. It answers one question:

> Has this log been altered since it was written?

The guarantee is **deployment-dependent** and is reported at runtime by
`Ledger.assurance_level()` and `StorageBackend.capabilities()`. It is never asserted in prose.

| Adversary | Capability | Detected by |
|---|---|---|
| A. API caller | Can call public API only | Hash chain (always) |
| B. DB writer / insider | Direct `UPDATE`/`DELETE` on the store | Signatures (if `enable_signatures`) + anchored checkpoints + WORM enforcement |
| C. Process compromise | Can read the signing key in-process | Only external anchoring + HSM/KMS signer |

**A bare hash chain is not tamper-evidence.** SHA-256 is public, so adversary B can edit a row
and re-derive every subsequent hash. Only two things stop that:

1. **A secret** — signatures the adversary cannot forge (`key_id` bound into the preimage).
2. **An external anchor** — a chain head published somewhere the adversary does not control.

Both are first-class here. Neither existed in v1.

### Assurance levels

`Ledger.assurance_level() -> AssuranceLevel`, computed, never hardcoded:

| Level | Condition |
|---|---|
| `NONE` | chain broken or unverified |
| `CHAIN_ONLY` | hash chain valid; no signatures, no anchor. Detects accidental corruption and naive edits **only**. |
| `SIGNED` | every entry has a signature verified against a resolved key |
| `ANCHORED` | `SIGNED` + at least one checkpoint anchored to an external sink |
| `WORM` | `ANCHORED` + backend reports `append_only_enforced` |

---

## 2. Canonical serialization — SLCS/1

Module: `signledger/crypto/canonical.py`. Spec: `docs/SPEC.md`. Vectors: `tests/vectors/*.json`.

The hash preimage is a **frozen byte encoding**. It is not `json.dumps` defaults.

**Rules**

1. UTF-8, no insignificant whitespace: separators `(',', ':')`.
2. Object keys sorted by **UTF-16 code unit** (RFC 8785 §3.2.3), not Python's default `str` order.
3. `datetime` → **integer microseconds since Unix epoch (UTC)**. Never an ISO string.
   Naive datetimes are **rejected**; tz-aware are converted to UTC first.
   *(v1 embedded the UTC offset in the digest, so a non-UTC Postgres session re-hashed
   every row differently and reported the whole log as tampered.)*
4. `NaN`, `Infinity`, `-Infinity` → **rejected** (`CanonicalizationError`). `allow_nan=False`.
5. Only JSON-native types in `data`/`metadata`: `dict`, `list`, `str`, `int`, `bool`, `None`, `float`.
   - Non-`str` dict keys → **rejected** (v1 collided `{1:...}` and `{"1":...}`).
   - `tuple`, `set`, `bytes`, `Decimal`, `datetime` inside `data` → **rejected** with the JSON path
     in the message (e.g. `data.amount: Decimal is not JSON-native; convert to str`).
     Strict rejection is deliberate: silent coercion is how two different payloads get one hash.
6. `int` outside ±2^53 → rejected (cross-language float coercion hazard).
7. Limits, enforced at `append()`: `MAX_ENTRY_BYTES = 1_048_576`, `MAX_NESTING_DEPTH = 64`.

```python
def canonicalize(obj: Any) -> bytes            # raises CanonicalizationError
def entry_preimage(entry: Entry) -> bytes      # SLCS/1 encoding of §3 field set
```

---

## 3. `Entry` — `signledger/core/entry.py`

Immutable (`model_config = {"frozen": True}`), pydantic v2.

| Field | Type | In preimage | Notes |
|---|---|---|---|
| `format_version` | `int` | ✅ | `2`. **First field.** Enables all future format changes. |
| `ledger_id` | `str` | ✅ | UUID of the owning ledger. Binds entries to one chain. |
| `sequence` | `int` | ✅ | Monotonic from `0`, **no gaps**. The ordering authority. |
| `entry_id` | `str` | ✅ | UUID4. |
| `timestamp` | `datetime` | ✅ | tz-aware UTC. |
| `data` | `dict` | ✅ | User payload. Non-empty. |
| `metadata` | `dict` | ✅ | Defaults `{}`. **Never NULL on disk.** |
| `previous_hash` | `str` | ✅ | Genesis hash when `sequence == 0`. Never `None`. |
| `key_id` | `str \| None` | ✅ | Signing key identity. In preimage so it cannot be swapped. |
| `signature_algorithm` | `str \| None` | ✅ | Ditto. |
| `hash` | `str` | ❌ | Hex digest of the preimage. |
| `signature` | `str \| None` | ❌ | Base64 over domain-separated hash bytes. |

`sequence` is the ordering authority. **No query may `ORDER BY timestamp`.**
*(v1 ordered by wall clock: equal timestamps returned reversed, and an NTP step-back
broke a pristine ledger.)*

`Entry.from_dict` must not mutate its argument — operate on `dict(data)`.
`data`/`metadata` are deep-copied on construction and on storage.

### Genesis

`sequence == 0` sets `previous_hash = genesis_hash(ledger_id, created_at, format_version)`.
Verification asserts entry 0 links to it, so the chain prefix cannot be replaced.

---

## 4. `StorageBackend` — `signledger/backends/base.py`

A `typing.Protocol` + `ConformanceTestMixin`. Any backend failing conformance fails at test time.

```python
class StorageBackend(Protocol):
    def append_atomic(self, build: Callable[[Entry | None], Entry]) -> Entry: ...
    def get_entry(self, entry_id: str) -> Entry | None: ...
    def get_by_sequence(self, sequence: int) -> Entry | None: ...
    def iter_entries(self, *, start_seq: int | None = None, end_seq: int | None = None,
                     start_time: datetime | None = None, end_time: datetime | None = None,
                     limit: int | None = None, offset: int = 0) -> Iterator[Entry]: ...
    def get_head(self) -> Entry | None: ...
    def get_oldest(self) -> Entry | None: ...
    def count(self) -> int: ...
    def size_bytes(self) -> int: ...
    def capabilities(self) -> frozenset[str]: ...
    def save_checkpoint(self, cp: Checkpoint) -> None: ...
    def get_latest_checkpoint(self) -> Checkpoint | None: ...
    def iter_checkpoints(self) -> Iterator[Checkpoint]: ...
    def close(self) -> None: ...
```

### `append_atomic` is the whole concurrency story

The tip read and the insert are **one database transaction**:

- SQLite — `BEGIN IMMEDIATE`
- Postgres — `SELECT … ORDER BY sequence DESC LIMIT 1 FOR UPDATE`
- Mongo — `findAndModify` counter document
- Memory — a `threading.Lock`

Plus `UNIQUE(sequence)` and `UNIQUE(previous_hash)` in every schema, and a bounded retry
(max 5, exponential backoff) on constraint violation.

*(v1 did read-modify-write under a process-local lock: two gunicorn workers forked the chain
within seconds, permanently, with no repair path.)*

**`append` must fail closed.** A backend error while reading the head raises `StorageError`.
`previous_hash = genesis` is permitted only when `count() == 0` is affirmatively confirmed.
*(v1 caught the exception, logged a warning, and silently wrote an unlinked entry.)*

### Capability tokens

`{"atomic_append", "unique_chain_link", "append_only_enforced", "worm", "durable_fsync"}`
— returned from `capabilities()`, surfaced by `get_stats()` and the CLI.

### Storage rules

- `metadata` persists as `json.dumps(entry.metadata or {})` — **never NULL**.
  *(This one line broke the entire v1 SQLite and Postgres path.)*
- Round-trip is byte-exact: `store → load → re-hash` equals the stored hash. Enforced by
  `tests/test_roundtrip_hash.py` on every backend.
- `limit`/`offset`: test `is not None`, never truthiness (`limit=0` means zero rows).
- Offset applies **after** filters.
- SQLite emits `LIMIT -1 OFFSET ?` when offset is given without limit.
- Identifiers validated against `^[A-Za-z_][A-Za-z0-9_]*$`; Postgres uses `psycopg.sql.Identifier`.
- Append-only DDL: SQLite `BEFORE UPDATE/DELETE` triggers raising `ABORT`; Postgres rules +
  a `REVOKE` script; Mongo `$jsonSchema`.

---

## 5. Verification — `signledger/core/verification.py`

**Verification never raises for integrity reasons.** It returns a report.

```python
@dataclass(frozen=True)
class VerificationFailure:
    kind: FailureKind   # BROKEN_LINK | BAD_HASH | BAD_SIGNATURE | MISSING_SIGNATURE
                        # | SEQUENCE_GAP | GENESIS_MISMATCH | HEAD_MISMATCH
                        # | COUNT_MISMATCH | UNKNOWN_KEY | FORMAT_UNSUPPORTED
    sequence: int | None
    entry_id: str | None
    expected: str | None
    actual: str | None
    message: str

@dataclass(frozen=True)
class VerificationReport:
    ok: bool
    entries_verified: int
    first_sequence: int | None
    last_sequence: int | None
    failures: tuple[VerificationFailure, ...]
    assurance: AssuranceLevel
    checkpoint_used: Checkpoint | None
```

`Ledger.check_integrity(...) -> VerificationReport` is primary.
`Ledger.verify_integrity(...) -> bool` is a thin wrapper kept for compatibility; it raises
`IntegrityError` on failure.

**It must collect all failures, not stop at the first** — an operator needs the full damage report.

Rules:
1. `entries_verified == 0` ⇒ `ok = False`. *(v1's CLI printed ✓ on an empty ledger.)*
2. `sequence` must start at the range start and increment by exactly 1 — gaps are `SEQUENCE_GAP`.
3. Entry 0 must link to the genesis hash.
4. When a checkpoint exists, head hash and entry count must match it — **this is what detects
   tail truncation**, which a bare chain cannot see. The checkpoint itself is authenticated
   first; see section 8.
5. When `enable_signatures`, a missing signature is `MISSING_SIGNATURE`, not a pass.
6. `check_integrity(from_sequence=…)` seeds `previous_hash` from that entry's own
   `previous_hash` and verifies the start entry itself.
   *(v1's `start_entry` raised `IntegrityError` on every healthy ledger.)*
7. All digest comparisons use `hmac.compare_digest`.
8. Reads for verification always hit the backend, **never the cache**.
   *(v1 served unverified cache hits, masking on-disk tampering.)*

---

## 6. Signatures — `signledger/crypto/signatures.py`

Signing and verification are **symmetric**. A signer without a verifier is not a feature.

```python
class Signer(Protocol):
    key_id: str                                        # SHA-256 fingerprint of the public key
    algorithm: str                                     # "rsa-pss-sha256" | "ecdsa-p256-sha256" | "ed25519"
    def sign(self, payload: bytes) -> str: ...
    def public_key_pem(self) -> str: ...

class KeyResolver(Protocol):
    def resolve(self, key_id: str) -> PublicKey | None: ...
```

- **Signed payload is domain-separated bytes**, never a hex `str`:
  `b"signledger/v2\x00" + ledger_id.encode() + b"\x00" + bytes.fromhex(entry.hash)`
  *(v1 called `signer(hash_hex_str)`; `Signer` was not callable and `.sign()` took bytes —
  the documented path raised `TypeError`.)*
- `key_id` and `signature_algorithm` are **in the hash preimage**, so an attacker cannot
  swap the key identity without breaking the hash.
- `Ledger(enable_signatures=True)` **without a signer fails at construction**, not on first append.
- Never `except Exception` around a verify — catch `InvalidSignature` explicitly.
- Private keys: `load_pem_private_key(..., password=...)` supported; `get_private_key_pem`
  defaults to `BestAvailableEncryption`. Never plaintext PKCS8 by default.
- `ECDSASigner` asserts the loaded key's curve matches the requested curve.

---

## 7. Merkle — `signledger/crypto/merkle.py` (RFC 6962)

Rewritten. The v1 implementation was forgeable.

- Leaf: `H(0x00 || data)`. Interior: `H(0x01 || left || right)`. **Domain separation is mandatory**
  *(v1 had none, so a 64-byte hex data item collided with an interior node).*
- Unpaired nodes are **promoted unchanged**, never duplicated
  *(v1 duplicated: `root([A,B,C]) == root([A,B,C,C])` — CVE-2012-2459).*
- Leaf count is bound into the root.

```python
def verify_inclusion(proof: InclusionProof, data_item: bytes, expected_root: str) -> bool
```

**`expected_root` is a required caller argument.** The root is never read from the proof.
*(v1's `verify_proof(proof)` compared against `proof.root_hash` — a field of the proof being
verified — so every forged proof returned `True`.)*

Asserts that path directions match the bits of `leaf_index`.

*(This line used to claim `len(proof.path) == ceil(log2(n))` as well. That is false for every
ragged tree — at n=3 the three proof paths have lengths [2, 2, 1] — and the implementation
deliberately does not assert it; `expected_path_length` documents `ceil(log2(n))` as the
**longest** path, not every path. Verified over sizes 1–17: every proof verifies, and the
short paths are correct.)*

---

## 8. Checkpoints & anchoring — `signledger/anchor/`

This is what makes tail truncation detectable.

```python
@dataclass(frozen=True)
class Checkpoint:
    ledger_id: str; sequence: int; head_hash: str; entry_count: int
    merkle_root: str; timestamp: datetime
    signature: str | None; key_id: str | None; format_version: int
```

```python
class AnchorSink(Protocol):
    name: str
    def publish(self, cp: Checkpoint) -> AnchorReceipt: ...
    def fetch_latest(self, ledger_id: str) -> Checkpoint | None: ...
```

Shipped sinks: `FileAnchorSink` (append-only JSONL, fsync), `NullAnchorSink` (tests).
`HTTPAnchorSink` / notarization are documented extension points.

`Ledger.checkpoint()` creates + signs + persists + publishes.

`check_integrity()` compares the head against a checkpoint, and a truncated tail yields
`HEAD_MISMATCH` + `COUNT_MISMATCH`. Two rules govern *which* checkpoint and *whether it is
believed*:

- **Both copies are consulted — the store's and the anchor's — and the newer wins.**
  Preferring the anchor unconditionally is a downgrade oracle: anyone who can reach the
  anchor rolls it back to an older genuine checkpoint and the store's contradicting copy is
  discarded unread. Requiring both to be rolled back is the point.
- **The checkpoint is authenticated before anything is compared against it.** `key_id`
  resolves, the signature verifies over the signing payload, `ledger_id` must be this
  chain's, `format_version` must be one this build reads, and `merkle_root` is recomputed
  over the entries up to the checkpointed sequence. Each has its own `FailureKind`
  (`CHECKPOINT_*`) rather than collapsing into `HEAD_MISMATCH`, which would report the
  chain as damaged when the damage is in the record it is being measured against.

An anchor that is behind, or empty, is **not** a failure — the commonest cause is a publish
that did not land, and an integrity report is the wrong place for an operational alarm. It
costs the `ANCHORED`/`WORM` label instead.

Limit, stated in `SECURITY.md` too: an attacker holding *both* the store and the anchor can
roll both back with nothing to detect it. Freshness cannot be established from inside a
system the attacker fully controls.

---

## 9. `Ledger` — `signledger/core/ledger.py`

```python
Ledger(backend=None, *, ledger_id=None, hash_algorithm="sha256",
       enable_signatures=False, signer=None, key_resolver=None,
       auto_verify=False, verify_interval=3600, on_integrity_failure=None,
       anchor_sink=None, max_cache_entries=1000)
```

- `auto_verify` defaults to **`False`**. When `True`, `on_integrity_failure` is **required**.
  *(v1 defaulted it on, then swallowed `IntegrityError` into a log line.)*
- `get_stats()` reports **real** state: `integrity_ok`, `last_verification_time`, `last_failure`,
  `assurance_level`, `capabilities`. It propagates backend errors instead of returning
  healthy-looking zeros. *(v1 hardcoded `integrity_verified = True`.)*
- Subscribers are notified **outside** the write lock, from a dedicated notifier thread with a
  bounded queue. *(v1 called them while holding a non-reentrant lock: a subscriber that
  appended deadlocked the ledger forever.)*
- Cache: validates a cached entry's hash before returning it and evicts on mismatch;
  `_load_cache` fetches the **newest** N *(v1 fetched the oldest)*; byte-budget eviction actually
  enforced.
- `close()` is idempotent, stops the notifier and verifier, then closes the backend.
  Registered via `weakref.finalize`.

---

## 10. Errors — `signledger/core/exceptions.py`

`SignLedgerError` ← `IntegrityError`, `ValidationError`, `StorageError`, `SignatureError`,
`CanonicalizationError`, `ConfigurationError`, `AnchorError`.

**No exception message may contain entry payloads, connection strings, or credentials.**
Enforced by `tests/test_error_redaction.py`. Use `Dict[str, Any]`, never PEP 585 `dict[...]`
(the package targets ≥3.9).

---

## 11. Testing methodology

Industry standard, layered. **Every test is written before its implementation and observed
failing first.**

| Layer | File | Proves |
|---|---|---|
| Contract | `test_backend_conformance.py` | all 4 backends behave identically (parametrized fixture) |
| Property | `test_tamper_detection.py` | Hypothesis: mutate *any* field ⇒ detected, and the failure names it |
| Round-trip | `test_roundtrip_hash.py` | `store → load → re-hash` is byte-identical per backend |
| Concurrency | `test_concurrency.py` | 16 threads × 200 appends + 2-process SQLite: no gaps, no forks |
| Vectors | `test_canonical_vectors.py` | SLCS/1 matches frozen `tests/vectors/*.json` |
| Adversary | `test_threat_model.py` | truncation, rewrite, signature stripping, cache masking |
| Packaging | `test_packaging.py` | build wheel → fresh venv → import every module → run console script |
| Docs | `test_docs_execute.py` | every ```python block in `README.md` and `examples/README.md` executes |
| Parity | `test_contrib_parity.py` | identical traffic against Flask, Django and FastAPI produces the same record shape |

**Gates** (CI-blocking):
- `pytest --cov=signledger --cov-branch --cov-fail-under=100`
- `mutmut run` — **nightly and on demand, not per-push.** A sweep is ~9,600 mutants, each
  needing its own test run; it does not fit in a per-push job. The job fails unless real
  verdicts come back, because it previously reported success for eight consecutive runs
  while executing nothing: `[tool.mutmut]` declared its paths as strings, mutmut 3.x
  concatenates them onto lists and died loading the config, and the job ended in
  `|| true`. Surviving mutants are reported, not auto-failed — the score is read by a
  person, since an equivalent mutant is not a defect.
  Coverage proves lines ran; **mutation proves the assertions mattered**.
  Two exclusions, both because the test cannot see a mutant rather than because it is
  inconvenient: `test_packaging.py` exercises the built artifact, and two `test_cli.py`
  tests `chdir` away from the source root, which mutmut's own `resolve(strict=True)`
  cannot survive.
- `mypy --strict -p signledger`, `ruff check`, `bandit -r signledger`
- Postgres/Mongo: real servers (local binaries / CI service containers), never mocks.

---

## 12. Compatibility policy

SemVer over **three independent surfaces**: Python API, on-disk schema, hash `format_version`.

Verification supports **every historical `format_version` forever**; writing always uses the newest.
`HashChain.calculate_hash` dispatches on `format_version` via `_PREIMAGE_ENCODERS`, and an
encoder once registered is never removed. `format_version` 2 is the only one that has ever
existed — v1.0.0 had no such field — so the dispatch table has one entry today. A v3 encoder is
added by registering it alongside 2, never by replacing it.

### No v1.0.0 upgrade path

**This section previously specified one. It was withdrawn as unimplementable, and the reasoning
is kept here so it is not proposed again.**

The original text said `migrate` would verify a v1 ledger under v1 rules and bind its head hash
and entry count into the v2 genesis, keeping history cryptographically continuous. It cannot,
because a v1 ledger on SQLite or PostgreSQL **has no head hash**: `previous_hash` is `NULL` on
every row (§3, and the note at line 152). There is no chain to verify under any rules, and
binding a fabricated head into a v2 genesis would manufacture evidence — the precise failure
this document exists to prevent.

`signledger migrate` therefore does one honest thing: it copies a ledger that **verifies** from
one store to another, refusing one that does not. A v1 ledger does not.

Operators upgrading from v1 read the old rows out as data and append them to a fresh v2 ledger.
The result attests that *these were the records held at migration time*, not that they are
unaltered since being written. `SECURITY.md` states this to users in those terms.
