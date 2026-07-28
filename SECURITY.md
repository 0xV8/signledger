# Security Policy

## Reporting a vulnerability

Report privately through [GitHub Security Advisories](https://github.com/0xV8/signledger/security/advisories/new).
Please do not open a public issue for a security problem.

Include a reproduction if you can. A working reproduction against a released
version is the fastest path to a fix.

Expect an acknowledgement within 7 days and an assessment within 30.

## Supported versions

| Version | Supported | Notes |
|---------|-----------|-------|
| 2.0.x   | yes       | current |
| 1.0.0   | **no**    | withdrawn — see below |

## About version 1.0.0

**1.0.0 does not provide the tamper-evidence it documents. Do not use it, and do
not present its output as evidence of anything.**

It was published before the project had any automated tests, while its README
claimed "100% test coverage". Three independent defects were found during a full
audit of the codebase:

1. **No hash chain was built on SQLite or PostgreSQL — the documented default
   path.** `Entry.metadata` defaults to `{}`, which is falsy, so the backend
   stored SQL `NULL`. Reading the row back then failed validation, and the
   append path caught that exception, logged a warning, and continued with
   `previous_hash = None`. Every row was written as an unlinked genesis block
   while `append()` reported success. `verify_integrity()` on such a ledger could
   not read its own data back.

2. **Signatures were generated but never verified.** `Signer.verify()` had no
   call sites. The ledger held no verifier, no public key and no key registry.
   Replacing every signature in the database with garbage left verification
   returning `True`.

3. **Tail truncation and full rewrite were undetectable.** Nothing anchored the
   chain head. Because the chain was unkeyed, an actor with write access could
   recompute a fabricated chain using the library's own code, with no secret
   required.

Additionally, `MerkleTree.verify_proof()` validated a proof against the root
carried *inside that same proof*, so every inclusion proof returned `True`.

### If you deployed 1.0.0

- **On SQLite or PostgreSQL:** treat the log as having **no integrity
  protection**. The rows are ordinary mutable rows with a `NULL` link column.
  They are still readable as data; they carry no evidence of being unaltered.
- **On the in-memory backend:** the chain was built correctly and detects
  in-place edits, but truncation and wholesale rewrite remain undetectable.
- **If you relied on signatures:** they were never checked.
- **In a compliance context (HIPAA, SOX, PCI-DSS):** do not represent 1.0.0
  output as tamper-evident.

### There is no integrity-preserving upgrade path, and there cannot be

Stated plainly rather than papered over: **a 1.0.0 ledger cannot be migrated into
2.0.0 in a way that carries its integrity forward, because it has no integrity to
carry.** On SQLite and PostgreSQL its rows are unlinked, so there is no chain
head to bind into a new genesis entry. Any tool that claimed otherwise would be
manufacturing evidence.

`signledger migrate` copies a ledger that *verifies* from one store to another.
It refuses a ledger that does not verify, and a 1.0.0 ledger does not.

The honest path is:

1. Read the old rows out as ordinary data.
2. Start a fresh 2.0.0 ledger.
3. Append the old rows to it, and treat the result as *"these are the records we
   held on this date"* — not as *"these records are unaltered since they were
   written"*. Only the second claim is tamper-evidence, and nothing can
   retroactively establish it for data written by 1.0.0.

If your retention obligations require attesting to the earlier period, that
attestation has to come from outside the library — backups, database WAL
archives, or whatever else your environment kept independently.

## What this library does and does not defend against

Read `docs/ARCHITECTURE.md` section 1 for the full statement. In short, a hash
chain by itself is **not** tamper-evidence: SHA-256 is public, so anyone who can
write to the store can edit a row and re-derive every hash after it. Detecting
that requires either a secret the attacker cannot forge (signatures) or a head
published where the attacker cannot reach it (anchoring).

`Ledger.check_integrity()` returns an `assurance` level alongside its verdict for
exactly this reason — it tells you what the verdict is worth:

| Level        | Detects                                         |
|--------------|-------------------------------------------------|
| `NONE`       | nothing                                         |
| `CHAIN_ONLY` | accidental corruption and naive edits           |
| `SIGNED`     | edits by anyone without the signing key         |
| `ANCHORED`   | truncation and rewrite, up to the last anchor   |
| `WORM`       | the above, plus storage-enforced immutability   |

A report that says `ok=True, assurance=CHAIN_ONLY` means "nothing here is
inconsistent", not "nobody tampered with this".

## Scope

In scope: anything that lets a ledger pass verification after being modified,
key handling, canonicalization ambiguity that permits two payloads to hash the
same, and injection through backend identifiers.

Out of scope: attacks that require the signing key or the anchor destination to
already be compromised, denial of service through resource exhaustion, and
vulnerabilities in optional dependencies (report those upstream).
