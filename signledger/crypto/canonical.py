"""SLCS/1 - SignLedger Canonical Serialization, version 1.

The hash preimage is a *frozen byte encoding*. It is deliberately not
``json.dumps`` defaults, and it is not negotiable: a second-language verifier must
reproduce these bytes exactly. The frozen conformance vectors in
``tests/vectors/slcs1.json`` are the cross-language contract.

Rules (architecture contract, section 2)
----------------------------------------
1. UTF-8, no insignificant whitespace - separators are ``,`` and ``:``.
2. Object keys are sorted by **UTF-16 code unit** (RFC 8785 section 3.2.3), *not* by
   Unicode code point. The two orders disagree for non-BMP characters, whose surrogate
   pairs begin at U+D800 and therefore sort ahead of BMP characters U+E000..U+FFFF.
3. ``datetime`` encodes as an **integer number of microseconds since the Unix epoch,
   UTC**. Never an ISO string. Naive datetimes are rejected; aware datetimes are
   converted to UTC first, so the same instant expressed in any timezone yields
   identical bytes.
4. ``NaN``, ``Infinity`` and ``-Infinity`` are rejected.
5. Only JSON-native types. Non-``str`` object keys are rejected. ``tuple``, ``set``,
   ``bytes``, ``Decimal`` and every other non-JSON type are rejected, with the JSON
   path of the offender in the message. Strict rejection is deliberate: silent
   coercion is how two different payloads end up with one hash.
6. ``int`` outside +/-2**53 is rejected - beyond that, languages whose only numeric
   type is a double cannot round-trip the value.
7. ``MAX_ENTRY_BYTES`` and ``MAX_NESTING_DEPTH`` are enforced here, so every caller
   inherits them.

Numbers follow ECMAScript ``Number::toString`` (RFC 8785 section 3.2.2.3), which means
``1``, ``1.0`` and ``1e0`` all encode as ``1``. That is not a collision: they are the
same JSON number, and it is precisely what keeps ``store -> load -> re-hash``
byte-stable across a store that round-trips payloads through JSON.
"""

import json
import math
from datetime import datetime, timezone
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

from signledger.core.exceptions import CanonicalizationError

if TYPE_CHECKING:  # pragma: no cover
    from signledger.core.entry import Entry

__all__ = [
    "MAX_ENTRY_BYTES",
    "MAX_NESTING_DEPTH",
    "canonicalize",
    "entry_preimage",
]

#: Maximum size of a canonical encoding, in bytes.
MAX_ENTRY_BYTES = 1_048_576

#: Maximum number of nested containers in a canonical encoding.
MAX_NESTING_DEPTH = 64

#: Largest integer a double can represent exactly.
_MAX_SAFE_INT = 2**53

#: Fields of :class:`~signledger.core.entry.Entry` that enter the hash preimage.
#: ``hash`` and ``signature`` are excluded - they are derived from it.
_PREIMAGE_FIELDS: Tuple[str, ...] = (
    "format_version",
    "ledger_id",
    "sequence",
    "entry_id",
    "timestamp",
    "data",
    "metadata",
    "previous_hash",
    "key_id",
    "signature_algorithm",
)

_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)

_CONVERSION_HINTS = {
    "tuple": "convert to list",
    "set": "convert to list",
    "frozenset": "convert to list",
    "bytes": "convert to str",
    "bytearray": "convert to str",
    "memoryview": "convert to str",
    "Decimal": "convert to str",
    "complex": "convert to str",
    "date": "convert to str",
    "time": "convert to str",
    "UUID": "convert to str",
}
_DEFAULT_HINT = "use only dict, list, str, int, float, bool or None"


def canonicalize(obj: Any, *, path: str = "", allow_datetime: bool = True) -> bytes:
    """Encode ``obj`` as SLCS/1 bytes.

    ``path`` seeds the JSON path used in error messages, so a caller validating a
    sub-tree can report ``data.amount`` rather than a bare ``amount``. It does not
    affect the produced bytes.

    ``allow_datetime`` selects between rule 3 and rule 5, which deliberately disagree.
    Rule 3 gives ``datetime`` a canonical form (integer microseconds) because the
    ``timestamp`` field of a preimage needs one. Rule 5 forbids ``datetime`` *inside*
    ``data``/``metadata``, where a payload should carry an explicit, self-describing
    value rather than something whose encoding a reader has to infer. Pass
    ``allow_datetime=False`` when validating a user payload.

    Raises:
        CanonicalizationError: if ``obj`` is not encodable under SLCS/1, or the
            encoding breaches ``MAX_ENTRY_BYTES`` or ``MAX_NESTING_DEPTH``.
    """
    return _finish(_render(obj, path=path, depth=0, allow_datetime=allow_datetime))


def entry_preimage(entry: "Entry") -> bytes:
    """Return the SLCS/1 encoding of ``entry``'s preimage field set.

    The preimage covers every field of contract section 3 except ``hash`` and
    ``signature``. ``key_id`` and ``signature_algorithm`` are inside it deliberately,
    so a signing identity cannot be swapped without breaking the digest.

    The envelope object is free of nesting budget: ``data`` nested N levels deep costs
    exactly the same here as it does under a direct ``canonicalize(entry.data)`` call.
    """
    fields: Dict[str, Any] = {name: getattr(entry, name) for name in _PREIMAGE_FIELDS}
    return _finish(_render(fields, path="", depth=-1, allow_datetime=True))


# --------------------------------------------------------------------------------------
# encoder
# --------------------------------------------------------------------------------------


def _finish(chunks: List[str]) -> bytes:
    encoded = "".join(chunks).encode("utf-8")
    if len(encoded) > MAX_ENTRY_BYTES:
        raise CanonicalizationError(
            f"canonical form is {len(encoded)} bytes; the limit is {MAX_ENTRY_BYTES}"
        )
    return encoded


def _render(obj: Any, *, path: str, depth: int, allow_datetime: bool) -> List[str]:
    out: List[str] = []
    _encode(obj, out, path, depth, allow_datetime)
    return out


def _encode(obj: Any, out: List[str], path: str, depth: int, allow_datetime: bool) -> None:
    # bool must precede int: bool is an int subclass.
    if obj is None:
        out.append("null")
    elif isinstance(obj, bool):
        out.append("true" if obj else "false")
    elif isinstance(obj, str):
        out.append(_encode_string(obj, path))
    elif isinstance(obj, int):
        out.append(_encode_int(obj, path))
    elif isinstance(obj, float):
        out.append(_encode_float(obj, path))
    elif isinstance(obj, datetime):
        if not allow_datetime:
            raise CanonicalizationError(
                f"{_show(path)}: datetime is not allowed inside a payload; "
                f"convert to str or to integer microseconds"
            )
        out.append(_encode_datetime(obj, path))
    elif isinstance(obj, dict):
        _encode_object(obj, out, path, depth, allow_datetime)
    elif isinstance(obj, list):
        _encode_array(obj, out, path, depth, allow_datetime)
    else:
        _reject_type(obj, path)


def _enter_container(path: str, depth: int) -> int:
    """Return the depth for a container's children, refusing to descend past the limit.

    The check happens *before* recursing, so the recursion is bounded by
    ``MAX_NESTING_DEPTH`` frames no matter how deep the input actually is - a hostile
    20,000-deep payload gets a CanonicalizationError, never a RecursionError.
    """
    inner = depth + 1
    if inner > MAX_NESTING_DEPTH:
        raise CanonicalizationError(
            f"{_show(path)}: nesting depth exceeds the maximum of {MAX_NESTING_DEPTH}"
        )
    return inner


def _reject_type(obj: Any, path: str) -> None:
    name = type(obj).__name__
    hint = _CONVERSION_HINTS.get(name, _DEFAULT_HINT)
    raise CanonicalizationError(f"{_show(path)}: {name} is not JSON-native; {hint}")


def _encode_object(
    obj: Dict[Any, Any], out: List[str], path: str, depth: int, allow_datetime: bool
) -> None:
    inner = _enter_container(path, depth)
    for key in obj:
        if not isinstance(key, str):
            raise CanonicalizationError(
                f"{_show(path)}: object keys must be str, got {type(key).__name__}"
            )
    out.append("{")
    for position, key in enumerate(sorted(obj, key=lambda k: _utf16_sort_key(k, path))):
        if position:
            out.append(",")
        out.append(_encode_string(key, path))
        out.append(":")
        _encode(obj[key], out, _key_path(path, key), inner, allow_datetime)
    out.append("}")


def _encode_array(
    obj: List[Any], out: List[str], path: str, depth: int, allow_datetime: bool
) -> None:
    inner = _enter_container(path, depth)
    out.append("[")
    for index, item in enumerate(obj):
        if index:
            out.append(",")
        _encode(item, out, _index_path(path, index), inner, allow_datetime)
    out.append("]")


def _encode_string(value: str, path: str) -> str:
    _reject_lone_surrogates(value, path)
    return json.dumps(value, ensure_ascii=False)


def _reject_lone_surrogates(value: str, path: str) -> None:
    """Reject strings CPython accepts but UTF-8 cannot represent.

    A lone surrogate survives inside a Python ``str`` but has no UTF-8 encoding, so it
    would blow up at the very end with no idea which field caused it.
    """
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as err:
        raise CanonicalizationError(
            f"{_show(path)}: str contains an unpaired surrogate and is not UTF-8 encodable"
        ) from err


def _encode_int(value: int, path: str) -> str:
    if abs(value) > _MAX_SAFE_INT:
        raise CanonicalizationError(
            f"{_show(path)}: int exceeds +/-2**53 and cannot be represented exactly "
            f"by a double; convert to str"
        )
    return str(value)


def _encode_float(value: float, path: str) -> str:
    if not math.isfinite(value):
        raise CanonicalizationError(
            f"{_show(path)}: non-finite float ({_nonfinite_name(value)}) has no JSON form"
        )
    return _es6_number_to_string(value)


def _encode_datetime(value: datetime, path: str) -> str:
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise CanonicalizationError(f"{_show(path)}: naive datetime is ambiguous; attach a tzinfo")
    delta = value.astimezone(timezone.utc) - _EPOCH
    micros = (delta.days * 86_400 + delta.seconds) * 1_000_000 + delta.microseconds
    return _encode_int(micros, path)


def _nonfinite_name(value: float) -> str:
    if math.isnan(value):
        return "NaN"
    return "Infinity" if value > 0 else "-Infinity"


def _utf16_sort_key(key: str, path: str) -> bytes:
    """Sort key implementing RFC 8785 section 3.2.3.

    UTF-16 big-endian bytes compare lexicographically exactly as the underlying 16-bit
    code units do, because every unit is two bytes with the high byte first.
    """
    _reject_lone_surrogates(key, path)
    return key.encode("utf-16-be")


def _es6_number_to_string(value: float) -> str:
    """Format a finite double per ECMAScript ``Number::toString`` (RFC 8785 3.2.2.3).

    ``repr`` already yields the shortest round-tripping decimal; this reshapes it into
    the ECMAScript layout, which differs from Python's for integral values (``1`` not
    ``1.0``), for exponents (``1e-7`` not ``1e-07``) and for negative zero (``0``).
    """
    if value == 0.0:  # folds -0.0 to "0", exactly as ECMAScript String(-0) does
        return "0"

    sign = "-" if value < 0 else ""
    parsed = Decimal(repr(abs(value))).as_tuple()
    digits = list(parsed.digits)
    exponent = int(parsed.exponent)
    while len(digits) > 1 and digits[-1] == 0:
        digits.pop()
        exponent += 1

    k = len(digits)
    n = k + exponent
    s = "".join(str(d) for d in digits)

    if k <= n <= 21:
        return sign + s + "0" * (n - k)
    if 0 < n <= 21:
        return sign + s[:n] + "." + s[n:]
    if -6 < n <= 0:
        return sign + "0." + "0" * -n + s
    mantissa = s if k == 1 else s[0] + "." + s[1:]
    marker = "+" if n > 0 else "-"
    return sign + mantissa + "e" + marker + str(abs(n - 1))


# --------------------------------------------------------------------------------------
# JSON paths
# --------------------------------------------------------------------------------------


def _show(path: str) -> str:
    return path or "<root>"


def _key_path(path: str, key: str) -> str:
    return f"{path}.{key}" if path else key


def _index_path(path: str, index: int) -> str:
    return f"{path}[{index}]"
