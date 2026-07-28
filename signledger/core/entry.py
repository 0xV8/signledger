"""``Entry`` - one immutable record in the append-only log (contract section 3).

``sequence`` is the ordering authority, not ``timestamp``. No query may order by wall
clock: equal timestamps come back in an arbitrary order, and an NTP step-back can make a
pristine ledger look reordered.

Error types raised from construction:

* :class:`~signledger.core.exceptions.ValidationError` - a field is structurally wrong
  (empty ``data``, naive ``timestamp``, negative ``sequence``), **or** a field has the
  wrong Python type outright.
* :class:`~signledger.core.exceptions.CanonicalizationError` - ``data`` or ``metadata``
  contains something SLCS/1 refuses to encode. The message carries the JSON path.

Both derive from ``SignLedgerError``, so one ``except`` covers the whole surface.

Pydantic's own ``ValidationError`` is deliberately **not** allowed to escape. Its message
renders the rejected value verbatim (``input_value='...'``), so ``ledger.append(secret)``
would print the secret into whatever log caught it - a direct violation of contract
section 10, on the library's single most-used method. :func:`_redacted` rewrites it into
a SignLedger ``ValidationError`` carrying the field location and the failure kind and
nothing else.
"""

import copy
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Optional

from pydantic import BaseModel, ConfigDict, field_validator
from pydantic import ValidationError as PydanticValidationError

from signledger.core.exceptions import ValidationError

__all__ = ["Entry"]

#: The format version written by this release.
CURRENT_FORMAT_VERSION = 2


class Entry(BaseModel):
    """An immutable, hash-chained ledger entry.

    Every field except ``hash`` and ``signature`` is covered by the hash preimage - see
    :func:`~signledger.crypto.canonical.entry_preimage`. ``key_id`` and
    ``signature_algorithm`` are inside it on purpose, so a signing identity cannot be
    swapped without breaking the digest.

    ``hash`` defaults to ``""`` so a draft can be built, hashed, and then sealed::

        draft = Entry(..., previous_hash=head)
        sealed = draft.model_copy(update={"hash": chain.calculate_hash(draft)})
    """

    model_config = ConfigDict(frozen=True)

    def __init__(self, **fields: Any) -> None:
        """Construct and validate, translating pydantic's error into a redacted one."""
        try:
            super().__init__(**fields)
        except PydanticValidationError as err:
            # `from None` is load-bearing, not cosmetic: chaining would attach the
            # original pydantic error, and printing the traceback would render the
            # rejected value after all.
            raise _redacted(err) from None

    # `model_validate` deliberately has no override: pydantic routes it through
    # `__init__` (main.py -> Entry.__init__), so it is already redacted. An override here
    # would be a second `except PydanticValidationError` that nothing can reach - dead
    # code guarding a door that is already locked. `test_model_validate_does_not_echo_
    # the_payload` pins the behaviour, so if pydantic ever stops delegating, a test says
    # so rather than an unreachable branch silently pretending to help.

    format_version: int = CURRENT_FORMAT_VERSION
    ledger_id: str
    sequence: int
    entry_id: str
    timestamp: datetime
    data: Dict[str, Any]
    metadata: Dict[str, Any] = {}
    previous_hash: str
    key_id: Optional[str] = None
    signature_algorithm: Optional[str] = None
    hash: str = ""
    signature: Optional[str] = None

    # -- payload ------------------------------------------------------------------

    @field_validator("data", mode="before")
    @classmethod
    def _validate_data(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value  # let pydantic report the type error
        if not value:
            raise ValidationError("data must not be empty")
        return _validated_payload(value, "data")

    @field_validator("metadata", mode="before")
    @classmethod
    def _validate_metadata(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value  # let pydantic report the type error
        return _validated_payload(value, "metadata")

    # -- identity and ordering ----------------------------------------------------

    @field_validator("format_version")
    @classmethod
    def _validate_format_version(cls, value: int) -> int:
        if value < 1:
            raise ValidationError("format_version must be 1 or greater")
        return value

    @field_validator("sequence")
    @classmethod
    def _validate_sequence(cls, value: int) -> int:
        if value < 0:
            raise ValidationError("sequence must be zero or greater")
        return value

    @field_validator("ledger_id", "entry_id", "previous_hash")
    @classmethod
    def _validate_required_text(cls, value: str, info: Any) -> str:
        if not value:
            raise ValidationError(f"{info.field_name} must not be empty")
        return value

    # -- timestamp ----------------------------------------------------------------

    @field_validator("timestamp", mode="before")
    @classmethod
    def _reject_epoch_numbers(cls, value: Any) -> Any:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            raise ValidationError(
                "timestamp must be a tz-aware datetime, not a bare epoch number "
                "whose unit cannot be inferred"
            )
        return value

    @field_validator("timestamp")
    @classmethod
    def _normalise_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
            raise ValidationError("timestamp must be timezone-aware")
        return value.astimezone(timezone.utc)

    # -- construction helpers ------------------------------------------------------

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Entry":
        """Build an ``Entry`` from a plain mapping without mutating it.

        The mapping is copied before use and ``data``/``metadata`` are deep-copied by
        the field validators, so the caller keeps sole ownership of what it passed in.
        """
        return cls(**dict(payload))


def _redacted(err: PydanticValidationError) -> ValidationError:
    """Rebuild a pydantic error as a SignLedger one, keeping only structure.

    Two fields of each pydantic error are used, and both are value-independent:

    * ``loc`` - the field path, so the caller can still find the bug;
    * ``type`` - a stable machine token such as ``dict_type`` or ``string_type``.

    ``input`` is dropped, and so is ``msg``: ``msg`` is human prose that pydantic is free
    to reword between releases, and some variants describe the offending input closely
    enough that a future version quoting it would silently reopen the leak. The type
    token says the same thing and cannot change shape underneath us.
    """
    parts = []
    for detail in err.errors():
        location = ".".join(str(item) for item in detail["loc"]) or "<entry>"
        parts.append(f"{location}: {detail['type']}")
    return ValidationError(
        f"{len(parts)} invalid field(s) on Entry: " + "; ".join(sorted(parts))
    )


def _validated_payload(value: Dict[Any, Any], field: str) -> Dict[Any, Any]:
    """Enforce SLCS/1 on a payload, then detach it from the caller.

    Canonicalizing first is what makes the deep copy safe: the size and nesting limits
    have already been checked, so ``deepcopy`` cannot be walked off a cliff by a
    hostile 20,000-deep payload.

    The import is deliberately function-local. ``signledger.crypto.canonical`` imports
    ``signledger.core.exceptions``, and importing any submodule of ``signledger.core``
    executes this package's ``__init__``, which imports this module - so a module-scope
    import here closes a genuine cycle. It is currently survivable only because
    ``signledger/__init__.py`` happens to import ``core`` before ``crypto``; reordering
    those two lines would break every ``import signledger.crypto.*`` at once. Deferring
    it to call time removes the cycle instead of relying on that ordering.
    """
    from signledger.crypto.canonical import canonicalize

    canonicalize(value, path=field, allow_datetime=False)
    return copy.deepcopy(value)
