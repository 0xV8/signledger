"""SignLedger in five minutes: append, verify, and watch tampering get caught.

Run it::

    python examples/quickstart.py

Everything happens in a throwaway directory that is deleted on exit, so you can run it
as often as you like.

What this example is honest about
---------------------------------
It reaches assurance level ``chain_only``, and that is a *low* rung. A hash chain proves
the log is internally consistent. It does not prove nobody rewrote it, because SHA-256 is
public: anyone who can edit a row can also recompute every digest after it and hand you a
chain that verifies perfectly. Stopping that needs a secret (signatures) or a witness
outside the database (an anchor). See ``banking_sox.py`` for both.

What a bare chain *does* buy you is real, and it is what you see below: accidental
corruption, a truncated field, a well-meaning "just fix this one row in production" - all
caught, and located to the exact entry.
"""

import pathlib
import sqlite3
import tempfile

from signledger.backends.sqlite import SQLiteBackend
from signledger.core.ledger import Ledger

PAYMENTS = [
    {"event": "payment.authorized", "order": "A-1001", "amount": "49.99", "currency": "USD"},
    {"event": "payment.captured", "order": "A-1001", "amount": "49.99", "currency": "USD"},
    {"event": "refund.issued", "order": "A-1001", "amount": "10.00", "currency": "USD"},
    {"event": "payment.authorized", "order": "A-1002", "amount": "129.00", "currency": "USD"},
]


def rule(title: str) -> None:
    print(f"\n--- {title} " + "-" * max(0, 62 - len(title)))


def write_the_log(database: str) -> str:
    """Append a handful of entries and confirm the ledger is sound."""
    rule("1. append")
    with Ledger(SQLiteBackend(database)) as ledger:
        for payment in PAYMENTS:
            entry = ledger.append(payment, metadata={"service": "checkout", "region": "eu-west-1"})
            print(f"  sequence {entry.sequence}  {entry.data['event']:<22} {entry.hash[:16]}...")
        print(f"  {ledger.count()} entries appended")

        rule("2. verify")
        report = ledger.check_integrity()
        print(f"  entries verified: {report.entries_verified}")
        print(f"  integrity: {'OK' if report.ok else 'FAILED'}")
        # ``.value`` rather than the enum: str-Enum formatting differs across Python
        # versions, and this string is the whole point of the line.
        print(f"  assurance level: {report.assurance.value}")
        print("  chain_only means: accidental corruption and naive edits are detected.")
        print("  It does NOT mean nobody with write access could have rewritten history.")
        return ledger.ledger_id


def try_the_front_door(database: str) -> None:
    """Attempt the edit an insider would try first, and get refused by the database."""
    rule("3. tamper, the easy way")
    connection = sqlite3.connect(database)
    try:
        connection.execute("UPDATE ledger_entries SET data = '{\"x\":1}' WHERE sequence = 1")
    except sqlite3.IntegrityError as refusal:
        print(f"  the database refused the UPDATE: {refusal}")
        print("  SQLite carries a BEFORE UPDATE trigger, so the rows are append-only.")
    finally:
        connection.close()


def force_the_edit(database: str) -> None:
    """Now do it properly: drop the trigger first, the way a real attacker would."""
    rule("4. tamper, the way an attacker actually would")
    connection = sqlite3.connect(database)
    with connection:
        connection.execute("DROP TRIGGER ledger_entries_no_update")
        connection.execute(
            "UPDATE ledger_entries SET data = ? WHERE sequence = ?",
            ('{"amount": "9999.00", "currency": "USD", "event": "refund.issued",'
             ' "order": "A-1001"}', 2),
        )
    connection.close()
    print("  dropped the append-only trigger and rewrote the refund amount to 9999.00")
    print("  the row now looks completely ordinary to anyone reading the table")


def audit(database: str) -> None:
    """Re-open the ledger later, as an auditor would, and check it."""
    rule("5. verify again")
    with Ledger(SQLiteBackend(database)) as ledger:
        report = ledger.check_integrity()
        print(f"  integrity: {'OK' if report.ok else 'FAILED'}")
        if report.ok:
            raise SystemExit("expected the rewritten row to be detected")
        print(f"  tampering detected: {len(report.failures)} failure(s)")
        for failure in report.failures:
            print(f"    [{failure.kind.value}] sequence {failure.sequence}: {failure.message}")
        print(f"  assurance level: {report.assurance.value}")
        print("  the digest is recomputed from the row, so the edit cannot survive it.")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="signledger-quickstart-") as workspace:
        database = str(pathlib.Path(workspace) / "audit.db")
        write_the_log(database)
        try_the_front_door(database)
        force_the_edit(database)
        audit(database)

    rule("what to read next")
    print("  banking_sox.py     signatures, checkpoints and external anchoring")
    print("  healthcare_hipaa.py  what to put in an audit record, and what to leave out")


if __name__ == "__main__":
    main()
