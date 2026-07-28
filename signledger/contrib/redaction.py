"""What is allowed to enter the ledger.

An append-only log is the worst place in a system to put a secret. You cannot delete it,
and because every later entry's digest is chained to this one, you cannot rewrite it
either without destroying the very property the ledger exists to provide. A credential
that lands here is not "logged"; it is *notarised*, permanently.

v1 wrote ``Authorization`` headers, every cookie and the entire request body - passwords,
SSNs - straight into that store, in a library marketed for HIPAA and PCI work.

So redaction here is not a formatting nicety, and it is not opt-in:

* The header denylist is **hard-coded** and cannot be configured away.
* Body capture is **off** unless a deployment asks for it, and even then the declared
  ``Content-Length`` is checked *before* a single byte is read.
* Sensitive field names are scrubbed at **any** nesting depth, because every real API
  nests and a top-level-only scrub is worse than none - it reads as protection.

Everything emitted here is also guaranteed to be encodable under SLCS/1. That is a
security property, not a convenience: with ``fail_closed`` audit semantics, a payload the
canonicalizer refuses is a request that returns 500, so a client who can post ``NaN``
could otherwise take an audited endpoint offline.
"""

import json
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Iterable, Mapping, Optional

from signledger.core.exceptions import ConfigurationError

__all__ = [
    "DEFAULT_MAX_BODY_BYTES",
    "DENIED_HEADERS",
    "MAX_REDACTION_DEPTH",
    "OVERSIZED",
    "REDACTED",
    "SENSITIVE_FIELDS",
    "SENSITIVE_NAME_PATTERN",
    "TRUNCATED",
    "UNPARSEABLE",
    "BodyDecision",
    "RedactionPolicy",
    "is_sensitive_field",
    "is_sensitive_header",
    "redact_headers",
    "redact_payload",
]

#: Stable stand-in for a value that must never be stored. Never the original, and never
#: a hash of the original either - a hashed password header is still a password oracle.
REDACTED = "[REDACTED]"

#: Stand-in for structure that was too deep to walk.
TRUNCATED = "[TRUNCATED]"

#: Stand-in for a body that was not valid JSON text. Recorded rather than raised: a
#: client who can post broken JSON must not be able to 500 an audited endpoint.
UNPARSEABLE = "[UNPARSEABLE]"

#: Stand-in for a body whose real size exceeded the limit its ``Content-Length`` claimed.
OVERSIZED = "[OVERSIZED]"

#: Default ceiling on a captured body. Small on purpose - an audit trail records that a
#: request happened, and only a deployment that has thought about it should be storing
#: what was in it.
DEFAULT_MAX_BODY_BYTES = 8192

#: Headers whose values are never recorded. This set is not configurable: a deployment
#: that could switch off ``authorization`` would eventually do it by accident.
DENIED_HEADERS = frozenset(
    {
        "authorization",
        "cookie",
        "set-cookie",
        "proxy-authorization",
        "x-api-key",
        "x-csrf-token",
    }
)

#: Any *name* that reads as sensitive is treated as sensitive. No fixed list can
#: enumerate the headers a given deployment invents, so the fallback is a pattern.
SENSITIVE_NAME_PATTERN = re.compile(r"token|secret|password|key", re.IGNORECASE)

#: Payload keys whose values are never recorded, at any nesting depth.
SENSITIVE_FIELDS = frozenset({"password", "ssn", "card", "cvv", "secret", "token"})

#: Matched as a **substring**: real payloads say ``card_number`` and
#: ``password_confirmation``, never the bare word. This over-redacts (``discard`` matches
#: ``card``), which is the correct direction to be wrong in for a store nobody can edit.
_FIELD_PATTERN = re.compile("|".join(sorted(SENSITIVE_FIELDS)), re.IGNORECASE)

#: How deep redaction will walk before replacing the rest with :data:`TRUNCATED`.
#: Bounded *before* recursing, so a hostile 20,000-deep body raises nothing at all.
#: Comfortably under the canonicalizer's own limit of 64 so the audit envelope still fits.
MAX_REDACTION_DEPTH = 24

#: SLCS/1 rejects integers beyond this, because languages with only a float numeric type
#: cannot represent them exactly.
_MAX_SAFE_INT = 2**53


def is_sensitive_header(name: str) -> bool:
    """Whether a header of this name may never have its value recorded."""
    lowered = name.lower()
    return lowered in DENIED_HEADERS or SENSITIVE_NAME_PATTERN.search(lowered) is not None


def is_sensitive_field(name: str, extra: FrozenSet[str] = frozenset()) -> bool:
    """Whether a payload key of this name may never have its value recorded.

    ``extra`` only ever *adds*. A deployment can widen this - ``patient_name``,
    ``account_number`` - but nothing can narrow the built-in set.
    """
    if _FIELD_PATTERN.search(name) is not None:
        return True
    lowered = name.lower()
    return any(word in lowered for word in extra)


def redact_headers(headers: Mapping[str, str]) -> Dict[str, str]:
    """Return ``headers`` with every sensitive value replaced by :data:`REDACTED`.

    Names are lower-cased so that one ledger holds one spelling: WSGI hands over
    ``HTTP_AUTHORIZATION``, ASGI lower-case bytes and Flask a title-cased mapping, and an
    auditor should not have to know which framework wrote a given row.
    """
    result: Dict[str, str] = {}
    for name, value in headers.items():
        lowered = name.lower()
        result[lowered] = REDACTED if is_sensitive_header(lowered) else str(value)
    return result


def redact_payload(value: Any, extra_fields: FrozenSet[str] = frozenset()) -> Any:
    """Return a copy of ``value`` that is safe to hash into the ledger.

    Two jobs at once, both mandatory:

    1. every sensitive key is replaced by :data:`REDACTED`, at any depth;
    2. the result is guaranteed encodable under SLCS/1, so the audit write cannot be
       made to fail by a client who controls the request body.

    The argument is never mutated - the caller still owns its own request body.
    """
    return _redact(value, 0, extra_fields)


def _redact(value: Any, depth: int, extra: FrozenSet[str]) -> Any:
    """Walk ``value``, scrubbing and sanitising. Depth is checked before recursing."""
    if depth >= MAX_REDACTION_DEPTH:
        return TRUNCATED
    if isinstance(value, dict):
        return {
            str(key): (
                REDACTED
                if is_sensitive_field(str(key), extra)
                else _redact(item, depth + 1, extra)
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_redact(item, depth + 1, extra) for item in value]
    return _sanitise(value)


def _sanitise(value: Any) -> Any:
    """Coerce one scalar into something SLCS/1 will accept.

    Conversions are chosen to be information-preserving where they can be: an oversized
    integer becomes its exact decimal string, which is precisely what the canonicalizer's
    own error message tells callers to do. Only a genuinely unrepresentable type
    collapses to a marker, and then the marker names the type and never the value.
    """
    # bool first: it is a subclass of int, and ``True`` must stay ``True``.
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return value if abs(value) <= _MAX_SAFE_INT else str(value)
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        if math.isnan(value):
            return "NaN"
        return "Infinity" if value > 0 else "-Infinity"
    return f"[UNSUPPORTED:{type(value).__name__}]"


@dataclass(frozen=True)
class BodyDecision:
    """Whether a request body may be read, and - when it may not - why not.

    The reason is recorded in the ledger in place of the body. "We chose not to store
    this, and here is the rule that said so" is auditable; a silently absent field is
    indistinguishable from a middleware that was never installed.
    """

    capture: bool
    reason: str


class RedactionPolicy:
    """The configurable half of redaction. The hard-coded half is not configurable.

    Every knob here can only make the policy *stricter* or turn capture on for a
    deployment that has decided it needs it. There is deliberately no way to remove a
    header from :data:`DENIED_HEADERS` or a name from :data:`SENSITIVE_FIELDS`.
    """

    def __init__(
        self,
        *,
        capture_body: bool = False,
        max_body_bytes: int = DEFAULT_MAX_BODY_BYTES,
        trusted_proxy_count: int = 0,
        extra_sensitive_fields: Iterable[str] = (),
    ) -> None:
        if max_body_bytes < 1:
            raise ConfigurationError(
                f"max_body_bytes must be at least 1, got {max_body_bytes}"
            )
        if trusted_proxy_count < 0:
            raise ConfigurationError(
                f"trusted_proxy_count must be zero or greater, got {trusted_proxy_count}"
            )
        self.capture_body = capture_body
        self.max_body_bytes = max_body_bytes
        self.trusted_proxy_count = trusted_proxy_count
        self.extra_sensitive_fields: FrozenSet[str] = frozenset(
            name.lower() for name in extra_sensitive_fields
        )

    # -- payloads ----------------------------------------------------------------------

    def headers(self, headers: Mapping[str, str]) -> Dict[str, str]:
        """Redact a header mapping."""
        return redact_headers(headers)

    def payload(self, value: Any) -> Any:
        """Redact and sanitise an arbitrary payload."""
        return redact_payload(value, self.extra_sensitive_fields)

    def query(self, params: Mapping[str, Any]) -> Dict[str, Any]:
        """Redact a query-string mapping.

        Parameter names are checked against the *header* rule as well as the payload one.
        A query string carries credentials in transit exactly like a header does -
        ``?api_key=`` is the classic example, and the payload field list has no ``key``
        entry, so the payload rule alone would write it into the ledger verbatim.

        Names keep their case: unlike headers, query parameters are case-sensitive.
        """
        return {
            name: (
                REDACTED
                if is_sensitive_header(name)
                or is_sensitive_field(name, self.extra_sensitive_fields)
                else self.payload(value)
            )
            for name, value in params.items()
        }

    # -- bodies ------------------------------------------------------------------------

    def body_decision(
        self, *, content_length: Optional[int], content_type: Optional[str]
    ) -> BodyDecision:
        """Decide whether a body may be read - **before** any of it is read.

        Checking the declared length first is the whole point. Reading and then measuring
        means an attacker sets the size of your memory allocation, and a middleware that
        buffers an unbounded body is a denial of service wearing a compliance badge.
        """
        if not self.capture_body:
            return BodyDecision(capture=False, reason="disabled")
        if content_length is None:
            return BodyDecision(capture=False, reason="no-content-length")
        if content_length > self.max_body_bytes:
            return BodyDecision(capture=False, reason="too-large")
        if not _is_json(content_type):
            return BodyDecision(capture=False, reason="unsupported-content-type")
        return BodyDecision(capture=True, reason="captured")

    def decode_body(self, raw: bytes) -> Any:
        """Parse and redact a body that :meth:`body_decision` approved.

        ``Content-Length`` is a claim made by the client, not a fact, so the real length
        is checked again here against the same limit.
        """
        if len(raw) > self.max_body_bytes:
            return OVERSIZED
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            # ValueError covers JSONDecodeError. Narrow on purpose: a bug in this method
            # should surface, only a malformed *body* is turned into a marker.
            return UNPARSEABLE
        return self.payload(parsed)

    # -- client identity ---------------------------------------------------------------

    def client_ip(self, *, peer: Optional[str], forwarded_for: Optional[str]) -> Optional[str]:
        """The client address, honouring ``X-Forwarded-For`` only as far as configured.

        ``X-Forwarded-For`` is a request header, which means the client writes it. Trusting
        it without knowing how many proxies actually sit in front of you lets anyone forge
        the one field an auditor will use to attribute an action to a person.

        With ``trusted_proxy_count = n`` the last ``n`` entries were appended by
        infrastructure you control, so the client is the entry just before them. Anything
        further left was supplied by the caller and is ignored. If the chain is shorter
        than ``n`` the deployment is not the one that was described, so the header proves
        nothing and the peer address is used instead.
        """
        if self.trusted_proxy_count and forwarded_for:
            parts = [item.strip() for item in forwarded_for.split(",") if item.strip()]
            if len(parts) >= self.trusted_proxy_count:
                return parts[-self.trusted_proxy_count]
        return peer


def _is_json(content_type: Optional[str]) -> bool:
    """Whether this ``Content-Type`` names JSON, ignoring any parameters."""
    if not content_type:
        return False
    return content_type.split(";", 1)[0].strip().lower() == "application/json"
