"""HIPAA-style PHI access logging: what belongs in an audit record, and what does not.

Run it::

    python examples/healthcare_hipaa.py

Everything happens in a throwaway directory that is deleted on exit.

The point of this example
-------------------------
An audit log of PHI access must not itself become a second, less protected copy of the
PHI. 45 CFR 164.312(b) asks you to record *that* a record was accessed, by whom, and why.
It does not ask you to record the diagnosis. A log that copies the chart into every entry
turns one protected system into two, and the second one is usually the one with the
loosest access controls, the longest retention and the widest replication.

So the interesting work here happens **before** ``ledger.append`` is ever called.
SignLedger does not redact anything for you - it faithfully preserves whatever you hand
it, forever, in a store designed to make deletion detectable. That is precisely why the
redaction has to happen at the call site, and why this example prints both halves: what
was captured, and what was deliberately dropped on the floor.

Re-identification
-----------------
Entries carry a *pseudonymous* subject id: an HMAC of the medical record number under a
key that lives with the covered entity and never enters the log. That keeps every access
to one patient correlatable - which is the whole point of an access history - while
leaving the log itself unable to name anybody. Someone who steals only the ledger gets a
list of opaque identifiers.

Honesty about the assurance level
---------------------------------
This run reaches ``signed`` and stops there, on purpose. Every entry carries a signature
that verifies against a resolved public key, so an insider with database write access
cannot forge an entry. They can still *delete the tail*, and a hash chain physically
cannot see that: what remains after a truncation is internally perfect. Detecting it
needs an external anchor, which this example does not configure - see ``banking_sox.py``.
"""

import hashlib
import hmac
import json
import pathlib
import tempfile
from typing import Any, Dict, List

from signledger.backends.sqlite import SQLiteBackend
from signledger.core.ledger import Ledger
from signledger.crypto.keys import InMemoryKeyResolver
from signledger.crypto.signatures import Ed25519Signer

#: Lives with the covered entity, alongside the chart itself - never in the audit log and
#: never in a backup of it. Without it the subject ids below cannot be reversed or even
#: tested against a guessed medical record number.
PSEUDONYM_KEY = b"per-deployment-secret-kept-outside-the-audit-log"

#: A chart, as the clinical system holds it. None of this belongs in an audit trail.
CHART = {
    "mrn": "MRN-4417-2290",
    "name": "Dana Okonkwo",
    "date_of_birth": "1974-03-19",
    "ssn": "512-88-7361",
    "diagnosis": "Type 2 diabetes mellitus with diabetic neuropathy",
    "medications": ["metformin 1000mg", "gabapentin 300mg"],
    "notes": "Patient reports intermittent numbness in both feet since January.",
}

ACCESSES = [
    ("dr.mensah", "physician", "view", "treatment", ["diagnosis", "medications"], "granted"),
    ("n.alvarez", "nurse", "view", "treatment", ["medications"], "granted"),
    ("b.kowalski", "billing", "view", "payment", ["diagnosis"], "granted"),
    ("t.ferris", "contractor", "export", "operations", ["notes", "ssn"], "denied"),
    ("dr.mensah", "physician", "amend", "treatment", ["notes"], "granted"),
]

#: The values that must never appear anywhere in the ledger. Checked for real below.
PHI_VALUES = [
    CHART["mrn"],
    CHART["name"],
    CHART["date_of_birth"],
    CHART["ssn"],
    CHART["diagnosis"],
    CHART["notes"],
    *CHART["medications"],
]


def rule(title: str) -> None:
    print(f"\n--- {title} " + "-" * max(0, 62 - len(title)))


def subject_id(mrn: str) -> str:
    """A stable, non-reversible handle for one patient.

    HMAC rather than a bare hash: a plain ``sha256(mrn)`` is trivially reversed by
    enumerating a structured identifier space, and medical record numbers are exactly
    that. The key never enters the log, so the log cannot be brute-forced back to a
    patient list.
    """
    return "subj_" + hmac.new(PSEUDONYM_KEY, mrn.encode("utf-8"), hashlib.sha256).hexdigest()[:24]


def redact(
    actor: str, role: str, action: str, purpose: str, fields: List[str], outcome: str
) -> Dict[str, Any]:
    """Build the audit record. This is the only place PHI is allowed to be dropped.

    Field *names* are recorded, values are not. "who looked at the diagnosis field" is an
    auditable fact; the diagnosis is not an auditable fact, it is the thing being
    protected.
    """
    return {
        "event": "phi.access",
        "subject": subject_id(CHART["mrn"]),
        "actor": actor,
        "actor_role": role,
        "action": action,
        "purpose_of_use": purpose,
        "fields_accessed": sorted(fields),
        "outcome": outcome,
    }


def record_accesses(ledger: Ledger) -> None:
    rule("1. log PHI access, redacted at the call site")
    for actor, role, action, purpose, fields, outcome in ACCESSES:
        payload = redact(actor, role, action, purpose, fields, outcome)
        entry = ledger.append(
            payload,
            metadata={"system": "ehr", "control": "45-CFR-164.312(b)", "source": "api"},
        )
        print(
            f"  PHI access recorded  sequence {entry.sequence}  {actor:<12} "
            f"{action:<7} {outcome}"
        )


def show_both_halves() -> None:
    rule("2. what the ledger holds, and what it never saw")
    payload = redact(*ACCESSES[0])
    print("  captured in the ledger:")
    for key, value in sorted(payload.items()):
        print(f"    {key:<16} {value}")

    print("\n  NOT captured (redacted before the ledger ever saw it):")
    for key in ("mrn", "name", "date_of_birth", "ssn", "diagnosis", "medications", "notes"):
        shown = CHART[key]
        summary = shown if isinstance(shown, str) else ", ".join(shown)
        print(f"    {key:<16} <withheld: {len(summary)} characters>")
    print("\n  SignLedger redacts nothing on your behalf. It preserves exactly what you")
    print("  hand it, in a store built to make deletion detectable - which is the reason")
    print("  the filtering has to happen here, before append(), and not afterwards.")


def prove_no_phi_leaked(ledger: Ledger) -> None:
    rule("3. verify the redaction actually held")
    haystack = "\n".join(
        json.dumps(entry.data, sort_keys=True) + json.dumps(entry.metadata, sort_keys=True)
        for entry in ledger.iter_entries()
    )
    leaked = sorted({value for value in PHI_VALUES if value in haystack})
    if leaked:
        raise SystemExit(f"PHI leaked into the ledger: {leaked}")
    print(f"  scanned {ledger.count()} entries for {len(PHI_VALUES)} known PHI values")
    print("  no PHI found in the ledger payloads")
    print(f"  the subject id {subject_id(CHART['mrn'])} correlates them without naming anyone")


def access_history(ledger: Ledger) -> None:
    rule("4. one patient's access history")
    subject = subject_id(CHART["mrn"])
    for entry in ledger.iter_entries():
        if entry.data.get("subject") != subject:
            continue
        stamp = entry.timestamp.isoformat(timespec="seconds")
        print(
            f"  {stamp}  {entry.data['actor']:<12} {entry.data['action']:<7} "
            f"{entry.data['purpose_of_use']:<11} {entry.data['outcome']:<8} "
            f"fields={','.join(entry.data['fields_accessed'])}"
        )


def audit(ledger: Ledger) -> None:
    rule("5. audit")
    report = ledger.check_integrity()
    print(f"  entries verified: {report.entries_verified}")
    print(f"  integrity: {'OK' if report.ok else 'FAILED'}")
    if not report.ok:
        raise SystemExit(f"integrity check failed: {[f.message for f in report.failures]}")
    print(f"  assurance level: {report.assurance.value}")
    print("  signed means: every entry verifies against a resolved public key, so an")
    print("  insider with database write access cannot forge or alter one. They can")
    print("  still delete the newest entries, and a hash chain cannot see that - what")
    print("  survives a truncation is internally perfect. Add an external anchor")
    print("  (banking_sox.py) if that is in your threat model. For most PHI access logs")
    print("  it should be.")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="signledger-hipaa-") as directory:
        database = pathlib.Path(directory) / "phi_access.db"
        signer = Ed25519Signer.generate()
        resolver = InMemoryKeyResolver()
        resolver.add_pem(signer.public_key_pem())

        with Ledger(
            SQLiteBackend(str(database)),
            enable_signatures=True,
            signer=signer,
            key_resolver=resolver,
        ) as ledger:
            record_accesses(ledger)
            show_both_halves()
            prove_no_phi_leaked(ledger)
            access_history(ledger)
            audit(ledger)


if __name__ == "__main__":
    main()
