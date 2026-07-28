"""Core data model, the ledger itself, and verification.

``Entry`` is the record, ``Ledger`` is the writer, and ``verify_chain`` is the reader that
trusts neither of them - it re-derives every digest from the bytes on disk and reports
what it finds. Verification never raises for an integrity reason: it returns a
``VerificationReport`` listing *every* failure, because an operator staring at a damaged
log needs the whole damage report, not the first line of it.

**Why half of this is lazy.** ``exceptions`` is a leaf that everything imports, including
``crypto.canonical`` and ``anchor.sinks``. Python cannot import ``signledger.core.exceptions``
without first executing *this* file, so anything heavy here is transitively imported by
every module in the library - and ``ledger`` imports ``anchor.sinks`` and ``backends``,
which import ``core.exceptions`` right back. Eagerly importing ``Ledger`` here therefore
closes a real cycle, and it fails loudly::

    ImportError: cannot import name 'checkpoint_signing_payload' from partially
    initialized module 'signledger.anchor.sinks'

So the light leaves (``Entry``, the exception hierarchy) are eager and everything that
reaches outside this package resolves on first attribute access instead. The public names
are identical either way: ``from signledger.core import Ledger`` works exactly as before.
"""

from typing import Any, List

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

#: Attribute name -> the module that defines it. Resolved on first access so that
#: importing ``signledger.core.exceptions`` stays free of the ledger's dependency tree.
_LAZY_EXPORTS = {
    "Ledger": "signledger.core.ledger",
    "AssuranceLevel": "signledger.core.verification",
    "FailureKind": "signledger.core.verification",
    "VerificationFailure": "signledger.core.verification",
    "VerificationReport": "signledger.core.verification",
    "verify_chain": "signledger.core.verification",
    "DEFAULT_MAX_BATCH_SIZE": "signledger.core.batch",
    "BatchItem": "signledger.core.batch",
    "BatchProcessor": "signledger.core.batch",
    "BatchResult": "signledger.core.batch",
    "DeadLetter": "signledger.core.batch",
    "PartialBatchError": "signledger.core.batch",
}


def __getattr__(name: str) -> Any:
    """Resolve the heavier exports on first access (PEP 562).

    Unknown names still raise ``AttributeError``: a ``__getattr__`` that answered every
    name would turn a typo into a failure somewhere far away from its cause.
    """
    try:
        module_name = _LAZY_EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None

    import importlib

    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value  # cache, so this costs one lookup per process
    return value


def __dir__() -> List[str]:
    """Make the lazy names discoverable by ``dir()`` and tab completion."""
    return sorted(set(globals()) | set(_LAZY_EXPORTS))


__all__ = [
    "CURRENT_FORMAT_VERSION",
    "DEFAULT_MAX_BATCH_SIZE",
    "AnchorError",
    "AssuranceLevel",
    "BatchItem",
    "BatchProcessor",
    "BatchResult",
    "CanonicalizationError",
    "ConfigurationError",
    "DeadLetter",
    "Entry",
    "FailureKind",
    "IntegrityError",
    "Ledger",
    "PartialBatchError",
    "SignLedgerError",
    "SignatureError",
    "StorageError",
    "ValidationError",
    "VerificationFailure",
    "VerificationReport",
    "verify_chain",
]
