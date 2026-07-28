"""SignLedger exception hierarchy (architecture contract, section 10).

Every error raised by SignLedger derives from :class:`SignLedgerError`, so a caller
can wrap the whole library in a single ``except``.

No exception message may contain entry payloads, connection strings or credentials.
Messages carry *structural* information only - a JSON path, a field name, a sequence
number, a type name - never a value taken from user data.
"""

__all__ = [
    "AnchorError",
    "CanonicalizationError",
    "ConfigurationError",
    "IntegrityError",
    "SignLedgerError",
    "SignatureError",
    "StorageError",
    "ValidationError",
]


class SignLedgerError(Exception):
    """Base class for every error raised by SignLedger."""


class IntegrityError(SignLedgerError):
    """The hash chain, a signature or an anchored checkpoint did not verify."""


class ValidationError(SignLedgerError):
    """An argument or entry field failed validation before it reached the store."""


class StorageError(SignLedgerError):
    """A storage backend failed. Appends fail closed rather than writing unlinked rows."""


class SignatureError(SignLedgerError):
    """A signing or signature-verification operation failed."""


class CanonicalizationError(SignLedgerError):
    """A value could not be encoded under SLCS/1.

    Raised for non-JSON-native types, non-``str`` object keys, ``NaN``/``Infinity``,
    integers outside +/-2**53, and violations of the size or nesting limits. The message
    carries the JSON path of the offending value, never the value itself.
    """


class ConfigurationError(SignLedgerError):
    """A ledger or backend was constructed with an unusable configuration."""


class AnchorError(SignLedgerError):
    """Publishing a checkpoint to, or fetching one from, an external anchor sink failed."""
