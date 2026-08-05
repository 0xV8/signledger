"""SignLedger - a tamper-evident, append-only audit log.

SignLedger answers exactly one question: **has this log been altered since it was
written?** It does not claim to answer it unconditionally. The strength of the answer
depends on how the log is deployed, and the library reports that strength at runtime
rather than asserting it in prose::

    from signledger import Ledger

    with Ledger() as ledger:
        ledger.append({"actor": "alice", "action": "deleted-record", "record": 42})
        report = ledger.check_integrity()

    report.ok            # did the chain verify?
    report.assurance     # what that verification is actually worth

``assurance`` is the important half. A bare hash chain (``AssuranceLevel.CHAIN_ONLY``)
detects accidental corruption and naive edits, and nothing more: SHA-256 is public, so
anyone who can write to the store can edit a row and re-derive every hash after it.
Detecting *that* needs a secret the attacker cannot forge (``enable_signatures``) or a
head published where the attacker cannot reach it (``anchor_sink``), which raise the
level to ``SIGNED``, ``ANCHORED`` and ``WORM``. See ``docs/ARCHITECTURE.md`` section 1.

Only the dependency-free surface is imported here. ``cryptography``, ``psycopg``,
``pymongo`` and the web frameworks are optional extras, so their modules are reachable by
path (``signledger.crypto.signatures``, ``signledger.backends.postgresql``, ...) but are
never pulled in by ``import signledger``.
"""

from signledger.anchor.checkpoint import Checkpoint
from signledger.core.batch import BatchItem, BatchProcessor, BatchResult
from signledger.core.entry import CURRENT_FORMAT_VERSION, Entry
from signledger.core.exceptions import (
    AnchorError,
    CanonicalizationError,
    ConfigurationError,
    IntegrityError,
    SignatureError,
    SignLedgerError,
    StorageError,
    ValidationError,
)
from signledger.core.ledger import Ledger
from signledger.core.verification import (
    AssuranceLevel,
    FailureKind,
    VerificationFailure,
    VerificationReport,
)

#: Python API version. Independent of the on-disk schema and of the hash
#: ``format_version`` (``CURRENT_FORMAT_VERSION``) - contract section 12 versions all
#: three surfaces separately.
#:
#: 2.0.0, not 0.2.0: 1.0.0 is published and withdrawn, and pip resolves to the highest
#: version. Numbering the rebuild below the release it replaces would leave
#: ``pip install signledger`` serving the broken one. It is also a full API break.
__version__ = "2.0b1"

__all__ = [
    "CURRENT_FORMAT_VERSION",
    "AnchorError",
    "AssuranceLevel",
    "BatchItem",
    "BatchProcessor",
    "BatchResult",
    "CanonicalizationError",
    "Checkpoint",
    "ConfigurationError",
    "Entry",
    "FailureKind",
    "IntegrityError",
    "Ledger",
    "SignLedgerError",
    "SignatureError",
    "StorageError",
    "ValidationError",
    "VerificationFailure",
    "VerificationReport",
    "__version__",
]
