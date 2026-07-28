"""A SOX-style financial audit trail: signed entries, checkpoints and external anchoring.

Run it::

    python examples/banking_sox.py

Everything happens in a throwaway directory that is deleted on exit.

This is deliberately not a web application. v1's flagship example was a Flask app that
returned HTTP 500 from every endpoint, bound to every network interface with the debug
console enabled - which is a remote shell, not an example. A script runs end to end in
CI, so it is a script, and it stays working.

The three controls it demonstrates
----------------------------------
1. **Signatures.** The signing key is loaded from an encrypted PEM on disk, exactly as it
   would be from a secrets mount. ``key_id`` and ``signature_algorithm`` live inside the
   hash preimage, so an insider cannot re-point an entry at a key they control without
   breaking the digest.

2. **External anchoring.** Checkpoints are published to a file *outside* the directory
   holding the database. This is the control that makes a truncated tail visible: lop the
   last N entries off a hash chain and what remains verifies perfectly, because the chain
   has no way to know it used to be longer. Only a statement made earlier, and kept
   somewhere the database administrator cannot reach, can contradict it. In production
   that means a different host, a different account, or a notarization service - the
   point is segregation of duties, not the file format.

3. **Ledger-first writes.** The balance moves only after the ledger has accepted the
   entry. v1 mutated the balance first and appended afterwards, so any failure in the
   append left money moved with no record of it - the exact opposite of an audit trail.

Honesty about the assurance level
---------------------------------
This run reaches ``worm``, the top rung, because the ledger is signed, anchored, and
SQLite reports ``append_only_enforced`` via its BEFORE UPDATE/DELETE triggers. Read that
precisely: a trigger stops the application and a careless operator, and it is a real
control. It does not stop someone with file-level write access, who can drop the trigger
in one statement - ``quickstart.py`` does exactly that. What survives that attacker is the
signature they cannot forge and the anchor they do not control, which is why all three
controls are here rather than just the one that prints the nicest word.
"""

import os
import pathlib
import tempfile
from decimal import Decimal
from typing import Dict

from cryptography.hazmat.primitives.asymmetric import ed25519

from signledger.anchor.sinks import FileAnchorSink
from signledger.backends.sqlite import SQLiteBackend
from signledger.core.batch import BatchProcessor
from signledger.core.exceptions import SignLedgerError
from signledger.core.ledger import Ledger
from signledger.crypto.keys import FileKeyResolver
from signledger.crypto.signatures import Ed25519Signer, get_private_key_pem

#: In production this comes from the environment or a secrets manager and never from
#: source. It is here so the example runs unattended; the important part is that the key
#: on disk is encrypted at all - ``get_private_key_pem`` refuses to write a plaintext one
#: unless you pass ``allow_unencrypted=True`` at the call site, where review can see it.
KEY_PASSWORD = os.environ.get("SIGNLEDGER_EXAMPLE_KEY_PASSWORD", "example-passphrase").encode()

OPENING_BALANCES = {"ACC-1001": 250_000, "ACC-2002": 12_500, "ACC-3003": 0}

TRANSFERS = [
    ("ACC-1001", "ACC-2002", 45_000, "quarterly settlement"),
    ("ACC-2002", "ACC-3003", 7_500, "vendor payout"),
    ("ACC-3003", "ACC-1001", 900_000, "attempted overdraw"),
    ("ACC-1001", "ACC-3003", 1_250, "expense reimbursement"),
]

#: End-of-day fee postings, written through the bulk path in one prepared chain.
DAILY_FEES = [
    {"event": "fee.posted", "account": "ACC-1001", "amount": "2.50", "kind": "maintenance"},
    {"event": "fee.posted", "account": "ACC-2002", "amount": "2.50", "kind": "maintenance"},
    {"event": "fee.posted", "account": "ACC-3003", "amount": "2.50", "kind": "maintenance"},
    {"event": "interest.accrued", "account": "ACC-1001", "amount": "18.04", "kind": "interest"},
]


def rule(title: str) -> None:
    print(f"\n--- {title} " + "-" * max(0, 62 - len(title)))


def money(cents: int) -> str:
    return f"{cents // 100}.{cents % 100:02d}"


def provision_keys(key_directory: pathlib.Path) -> pathlib.Path:
    """Write an encrypted signing key and its public half, as a deployment would."""
    key_directory.mkdir(parents=True, exist_ok=True)
    public_directory = key_directory / "public"
    public_directory.mkdir(exist_ok=True)

    private_key = ed25519.Ed25519PrivateKey.generate()
    private_path = key_directory / "signing-key.pem"
    private_path.write_text(get_private_key_pem(private_key, KEY_PASSWORD), encoding="ascii")
    private_path.chmod(0o600)

    signer = Ed25519Signer(private_key)
    # Named by fingerprint, not by role: the resolver derives a key's id from the key
    # material itself, so renaming a file can never silently re-point a key_id.
    (public_directory / f"{signer.key_id[:16]}.pem").write_text(
        signer.public_key_pem(), encoding="ascii"
    )
    return private_path


def open_ledger(workspace: pathlib.Path) -> Ledger:
    """Load the signer from disk and wire up verification and anchoring."""
    rule("1. keys and anchoring")
    key_directory = workspace / "keys"
    private_path = provision_keys(key_directory)

    signer = Ed25519Signer.from_pem(private_path.read_bytes(), KEY_PASSWORD)
    print(f"  loaded signing key from {private_path.name} (encrypted PKCS#8)")
    print(f"  algorithm: {signer.algorithm}   key_id: {signer.key_id[:32]}...")

    resolver = FileKeyResolver(key_directory / "public")
    print(f"  verification keys on disk: {len(resolver)}")

    # Deliberately not under the database directory. An anchor stored beside the ledger
    # is controlled by whoever controls the ledger, and protects against nothing.
    anchor_path = workspace / "offsite" / "anchors.jsonl"
    anchor_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  anchor sink: {anchor_path.parent.name}/{anchor_path.name} (separate from the db)")

    database = workspace / "db" / "banking.db"
    database.parent.mkdir(parents=True, exist_ok=True)
    return Ledger(
        SQLiteBackend(str(database)),
        enable_signatures=True,
        signer=signer,
        key_resolver=resolver,
        anchor_sink=FileAnchorSink(str(anchor_path)),
    )


def post_transfers(ledger: Ledger, balances: Dict[str, int]) -> None:
    """Record every transfer, and move money only once the record exists."""
    rule("2. post transfers (ledger first, balance second)")
    for source, target, cents, reason in TRANSFERS:
        if balances[source] < cents:
            # A refusal is itself an auditable event. Logging only the successes is how a
            # control gets described as working when nobody can show what it rejected.
            ledger.append(
                {
                    "event": "transfer.refused",
                    "from": source,
                    "to": target,
                    "amount": money(cents),
                    "reason": "insufficient_funds",
                    "available": money(balances[source]),
                },
                metadata={"control": "SOX-404", "system": "core-banking"},
            )
            print(f"  refused: insufficient funds  {source} -> {target} {money(cents)}")
            continue

        entry = ledger.append(
            {
                "event": "transfer.posted",
                "from": source,
                "to": target,
                "amount": money(cents),
                "memo": reason,
            },
            metadata={"control": "SOX-404", "system": "core-banking"},
        )
        # Only now. If the append had raised, no money would have moved.
        balances[source] -= cents
        balances[target] += cents
        print(
            f"  posted   sequence {entry.sequence}  {source} -> {target} "
            f"{money(cents):>10}  signed {entry.signature[:12]}..."
        )
    print("  balance updated after the ledger accepted the entry, never before")

    # Prove it rather than assert it. A Decimal is exactly the kind of value an ORM hands
    # you and SLCS/1 refuses, so this append fails - and the balance below is the evidence
    # that the failure happened before any money moved.
    rule("2b. an append that fails moves no money")
    before = balances["ACC-1001"]
    try:
        ledger.append(
            {
                "event": "transfer.posted",
                "from": "ACC-1001",
                "to": "ACC-2002",
                "amount": Decimal("10.00"),
            }
        )
        balances["ACC-1001"] -= 1_000
        balances["ACC-2002"] += 1_000
    except SignLedgerError as refusal:
        print(f"  ledger rejected the entry: {refusal}")
    print(f"  ACC-1001 balance unchanged: {money(balances['ACC-1001'])} "
          f"(was {money(before)})")
    print("  v1 moved the money first, so this path left funds transferred with no record")


def post_daily_fees(ledger: Ledger) -> None:
    """End-of-day postings through the bulk path: one prepared chain, then one write."""
    rule("3. end-of-day batch")
    result = BatchProcessor(ledger).append_batch(DAILY_FEES)
    print(f"  submitted {result.submitted}, committed {result.committed}, "
          f"refused {result.failed}")
    print(f"  took {result.duration_seconds * 1000:.1f} ms, every entry signed on the way in")
    if not result.ok:
        raise SystemExit(f"batch did not commit cleanly: {result.dead_letters}")

    # An atomic batch refuses the whole submission if any single item is unusable.
    rejected = BatchProcessor(ledger).append_batch(
        [{"event": "fee.posted", "amount": "1.00"}, {"event": "fee.posted", "amount": object()}]
    )
    print(f"  a batch containing one unusable item committed {rejected.committed} entries")
    for letter in rejected.dead_letters:
        print(f"    item {letter.index} refused: {letter.error}")


def anchor(ledger: Ledger, workspace: pathlib.Path) -> None:
    rule("4. checkpoint and anchor")
    checkpoint = ledger.checkpoint()
    anchor_path = workspace / "offsite" / "anchors.jsonl"
    print(f"  checkpoint published to {anchor_path.name}")
    print(f"  sequence {checkpoint.sequence}, {checkpoint.entry_count} entries")
    print(f"  head hash   {checkpoint.head_hash[:32]}...")
    print(f"  merkle root {checkpoint.merkle_root[:32]}...")
    print("  this record is what makes a deleted tail detectable; the chain alone cannot"
          " see one")


def audit(ledger: Ledger, balances: Dict[str, int]) -> None:
    rule("5. audit")
    report = ledger.check_integrity()
    print(f"  entries verified: {report.entries_verified}")
    print(f"  integrity: {'OK' if report.ok else 'FAILED'}")
    if not report.ok:
        raise SystemExit(f"integrity check failed: {[f.message for f in report.failures]}")

    stats = ledger.get_stats()
    print(f"  backend capabilities: {', '.join(sorted(stats['capabilities']))}")
    print(f"  assurance level: {report.assurance.value}")
    print("  worm here means: signed, anchored offsite, and the database refuses UPDATE")
    print("  and DELETE. It does not mean the trigger cannot be dropped by someone with")
    print("  file-level write access - it means doing so cannot forge a signature or")
    print("  rewrite the anchor, so the tampering still surfaces.")

    rule("closing balances")
    for account, cents in sorted(balances.items()):
        print(f"  {account}  {money(cents):>12}")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="signledger-banking-") as directory:
        workspace = pathlib.Path(directory)
        balances = dict(OPENING_BALANCES)
        ledger = open_ledger(workspace)
        try:
            post_transfers(ledger, balances)
            post_daily_fees(ledger)
            anchor(ledger, workspace)
            audit(ledger, balances)
        except SignLedgerError as failure:
            raise SystemExit(f"ledger error: {failure}") from failure
        finally:
            ledger.close()


if __name__ == "__main__":
    main()
