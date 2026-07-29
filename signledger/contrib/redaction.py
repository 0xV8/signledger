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
* **One** name rule decides all three of body, header and query string. Two rules meant
  ``?api_key=`` was redacted while ``{"api_key": ...}`` was notarised in the clear.

Everything emitted here is also guaranteed to be encodable under SLCS/1. That is a
security property, not a convenience: with ``fail_closed`` audit semantics, a payload the
canonicalizer refuses is a request that returns 500, so a client who can post ``NaN`` -
or a lone UTF-16 surrogate, which ``json.loads`` accepts and UTF-8 cannot represent -
could otherwise take an audited endpoint offline, or under ``fail_open`` choose to
delete its own audit record.

What this does **not** do
-------------------------

Two limits, stated here because an undocumented limit reads as a guarantee to whoever is
deciding what their service should capture:

* This is a denylist of **field names**, and it never looks at a value. A field called
  ``notes`` holding ``"the customer said their password is hunter2"`` is notarised exactly
  as written - free text is never scanned, and a list of names can never be complete
  anyway, because the next deployment will invent a name nobody here thought of. If a
  field can hold prose that a human might paste a credential into, the safe answer is to
  not capture that field at all.
* Request bodies are **not captured by default**. ``RedactionPolicy.capture_body`` is
  ``False``, so unless a deployment turns it on, no part of a body - redacted or not -
  reaches the ledger, and the name list below only ever sees headers and query strings.
  Turning body capture on is the point at which that list becomes the only thing standing
  between a login form and an append-only store, and it is the point at which a deployment
  should add the field names its own APIs use, through
  ``RedactionPolicy(extra_sensitive_fields=...)``. That argument only ever widens the
  rule - on bodies, headers and query strings alike - and nothing can narrow it.
"""

import json
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Iterable, List, Mapping, Optional, Set

from signledger.core.exceptions import ConfigurationError

__all__ = [
    "DEFAULT_MAX_BODY_BYTES",
    "DENIED_HEADERS",
    "MAX_REDACTION_DEPTH",
    "OVERSIZED",
    "REDACTED",
    "SENSITIVE_COMPOUNDS",
    "SENSITIVE_FIELDS",
    "SENSITIVE_SUBSTRINGS",
    "SENSITIVE_WORDS",
    "TRUNCATED",
    "UNENCODABLE",
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

#: Stand-in for text CPython holds happily and UTF-8 cannot represent.
UNENCODABLE = "[UNENCODABLE]"

#: Default ceiling on a captured body. Small on purpose - an audit trail records that a
#: request happened, and only a deployment that has thought about it should be storing
#: what was in it.
DEFAULT_MAX_BODY_BYTES = 8192

#: Headers whose values are never recorded. This set is not configurable: a deployment
#: that could switch off ``authorization`` would eventually do it by accident.
#:
#: :data:`SENSITIVE_SUBSTRINGS` already covers every name here as a substring. The exact
#: list is kept anyway, as a floor: if a future edit narrows the pattern, these six go.
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

#: The stems this list has always had, matched as a plain **substring** of the lower-cased
#: name. The first of three sets; :data:`SENSITIVE_COMPOUNDS` and :data:`SENSITIVE_WORDS`
#: below hold the rest, each with the matching style its length can carry.
#:
#: A name any of the three matches is never recorded - in a body, a header or a query
#: string.
#:
#: **One list, one rule.** There used to be two: a regex guarding headers and query
#: strings and a shorter word list guarding bodies. The result was that ``?api_key=`` was
#: redacted and ``{"api_key": ...}`` was notarised in the clear - the same credential,
#: protected on one path and permanently recorded on the other. A rule that depends on
#: which part of a request a secret arrived in is not a rule. Widening the list has not
#: reintroduced that split: there are three *matching styles* here, and all three apply to
#: every name, whatever transport it arrived on.
#:
#: Substrings, because real payloads say ``card_number`` and
#: ``password_confirmation`` and never the bare word. That deliberately over-redacts, and
#: the trade is not close: an audit record missing one field costs an operator a lookup,
#: while an append-only log holding one credential cannot be repaired at all - deleting
#: the row breaks every later entry's hash chain.
#:
#: The two entries that pay for that choice, decided knowingly:
#:
#: ``key``
#:     also matches ``monkey``, ``donkey``, ``keyboard``, ``keyword``, ``key_count`` and
#:     ``public_key``. A public key genuinely is not a secret. It is still redacted,
#:     because no substring separates ``api_key`` and ``secret_key`` from ``monkey``
#:     without also letting through whatever a deployment names its credential next -
#:     and the failure that matters is ``private_key``, which is unrecoverable.
#: ``auth``
#:     also matches ``author`` and ``authority``. Same reasoning: ``auth`` is itself a
#:     common field name for a bearer credential and there is nothing narrower.
#:
#: And the boundary in the other direction: the password family is spelled ``passw``,
#: ``pwd`` and ``passphrase`` rather than a bare ``pass``. ``passw`` already covers
#: ``password``, ``passwd`` and ``password_confirmation``, so widening it to ``pass``
#: adds no real credential name while blanking ``passenger_count``, ``compass`` and
#: ``bypass``. Over-redaction is the right default, not a licence to redact everything.
#:
#: Matched against the raw lower-cased name and *not* against the separator-stripped
#: form, because these stems are short: ``metrics_snapshot`` holds no ``ssn`` and
#: ``metricssnapshot`` would.
SENSITIVE_SUBSTRINGS = frozenset(
    {
        "auth",
        "card",
        "cookie",
        "credential",
        "cvv",
        "key",
        "passphrase",
        "passw",
        "pwd",
        "secret",
        "session",
        "ssn",
        "token",
    }
)

#: Long stems, matched as a substring of the name with **every separator removed**.
#:
#: The list above missed five whole categories of credential, and the reason it missed
#: them is instructive: ``passcode`` shares no substring with ``passw``, ``pwd`` or
#: ``passphrase``, so a login form whose field is called ``passcode`` was stored in the
#: clear. Nothing about the old rule was going to catch that; only adding the name does.
#:
#: Separators are stripped first so that one entry covers every spelling a language and a
#: framework can produce: ``security_code``, ``SECURITY_CODE``, ``security-code``,
#: ``security.code``, ``securityCode``, ``SecurityCode`` and ``securitycode`` are one
#: credential, not seven, and a list that only knew the snake_case one would leak the
#: other six.
#:
#: Nothing short belongs here, because stripping separators lets a stem match across a
#: word boundary that was never in the name: ``taxid`` is the shortest entry at five
#: characters and it is also the only one that costs anything - ``syntax_id`` squashes to
#: ``syntaxid`` and is blanked. One contrived field name is the price of catching every
#: real spelling of ``tax_id``, and short stems go in :data:`SENSITIVE_WORDS` instead.
SENSITIVE_COMPOUNDS = frozenset(
    {
        "accountnumber",
        "backupcode",
        "bearer",
        "certificate",
        "confirmationcode",
        "entropy",
        "mnemonic",
        "onetimecode",
        "passcode",
        "passport",
        "pincode",
        "recoverycode",
        "routingnumber",
        "securitycode",
        "seedphrase",
        "signature",
        "socialinsurance",
        "socialsecurity",
        "taxid",
        "twofactor",
        "verificationcode",
    }
)

#: Short stems, matched as **whole words** only.
#:
#: This is where over-redaction stops being free. ``pin`` as a substring blanks
#: ``shipping``, ``spinner``, ``pinned`` and ``mapping``; ``sin`` blanks ``single``,
#: ``using`` and ``business``; ``seed`` blanks ``seeded``; ``cert`` blanks ``certain``.
#: Those are ordinary fields on ordinary endpoints, and a ledger whose every row reads
#: ``[REDACTED]`` has stopped being an audit trail - which is a security failure of its
#: own, just a quieter one.
#:
#: So a name is split into words first: on any non-alphanumeric separator, on camelCase
#: boundaries, and on letter/digit boundaries (so ``user2fa`` yields ``2fa`` and ``totp2``
#: yields ``totp``, while ``p12`` still yields itself). A trailing plural ``s`` is
#: stripped, so ``certs`` matches ``cert``. ``otp``, ``otp_code``, ``OtpCode``,
#: ``login-otp`` and ``otp.value`` all match; ``shipping`` does not.
#:
#: The price paid on this side, also knowingly: ``seed_data`` and ``random_seed`` are
#: blanked, and neither is crypto material. A wallet ``seed`` is, permanently.
SENSITIVE_WORDS = frozenset(
    {
        "2fa",
        "cert",
        "creds",
        "cvc",
        "hmac",
        "hotp",
        "iban",
        "jwt",
        "mfa",
        "nonce",
        "otp",
        "p12",
        "pem",
        "pfx",
        "pin",
        "salt",
        "seed",
        "sin",
        "totp",
    }
)

#: Every stem the built-in rule knows, in one frozenset, so a deployment can print the
#: policy it is running under. Which *style* each stem is matched with is the business of
#: the three sets above; this union is for reading, not for matching.
SENSITIVE_FIELDS = SENSITIVE_SUBSTRINGS | SENSITIVE_COMPOUNDS | SENSITIVE_WORDS

#: Compiled forms of the two substring styles. Sorted so the patterns are stable across
#: runs and reviewable in a diff, and applied to already-lower-cased text rather than with
#: ``re.IGNORECASE``, because the word rule has to lower-case the name anyway.
_SUBSTRING_PATTERN = re.compile("|".join(re.escape(stem) for stem in sorted(SENSITIVE_SUBSTRINGS)))
_COMPOUND_PATTERN = re.compile("|".join(re.escape(stem) for stem in sorted(SENSITIVE_COMPOUNDS)))

#: ``userPin`` and ``HTTPToken`` are one word to a splitter that only knows about ``_``.
#: A digit does *not* end a word here, deliberately: ``2FA_CODE`` would otherwise be cut
#: into ``2`` and ``FA`` and stop matching. Letter/digit boundaries are handled by
#: :data:`_RUNS` instead, which offers both halves *and* the whole.
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")

#: Anything that is not an ASCII alphanumeric separates two words - and is also what gets
#: stripped to compare against :data:`SENSITIVE_COMPOUNDS`.
_NOT_ALNUM = re.compile(r"[^0-9a-z]+")

#: A run of letters or a run of digits, so ``user2fa`` can be read as ``user`` + ``2fa``.
_RUNS = re.compile(r"[a-z]+|[0-9]+")

#: How many adjacent runs may be joined back into one candidate word. Rejoining *every*
#: span is quadratic in the length of the name, and a name is client-written text: a
#: single 4 KB key of ``a1a1a1...`` took 79 seconds, on a code path that runs on every
#: request. Three is more than enough - no stem here spans more than two runs
#: (``2fa`` is digits+letters, ``p12`` is letters+digits) - and it makes the walk linear.
_MAX_JOINED_RUNS = 3

#: How deep redaction will walk before replacing the rest with :data:`TRUNCATED`.
#: Bounded *before* recursing, so a hostile 20,000-deep body raises nothing at all.
#: Comfortably under the canonicalizer's own limit of 64 so the audit envelope still fits.
MAX_REDACTION_DEPTH = 24

#: SLCS/1 rejects integers beyond this, because languages with only a float numeric type
#: cannot represent them exactly.
_MAX_SAFE_INT = 2**53


def _squash(lowered: str) -> str:
    """``security-Code`` -> ``securitycode``. Every separator a spelling can invent, gone."""
    return _NOT_ALNUM.sub("", lowered)


def _words(name: str) -> FrozenSet[str]:
    """Split ``name`` into the words a human would read in it.

    Separators, camelCase boundaries and letter/digit boundaries all end a word. Up to
    :data:`_MAX_JOINED_RUNS` adjacent runs are also joined back up, so ``user2fa`` offers
    ``user``, ``2fa`` and ``user2fa`` and both ``2fa`` and ``p12`` can be recognised
    without one rule breaking the other. The bound is what keeps this linear on a name an
    attacker chose.
    """
    words: Set[str] = set()
    for part in _NOT_ALNUM.split(_CAMEL_BOUNDARY.sub(" ", name).lower()):
        runs = _RUNS.findall(part)
        for start in range(len(runs)):
            for end in range(start + 1, min(start + _MAX_JOINED_RUNS, len(runs)) + 1):
                words.add("".join(runs[start:end]))
    return frozenset(words)


def _is_sensitive_word(word: str) -> bool:
    """Whether one word of a name is a short credential stem, plural included."""
    if word in SENSITIVE_WORDS:
        return True
    return word.endswith("s") and word[:-1] in SENSITIVE_WORDS


def _is_sensitive_name(name: str) -> bool:
    """The one name rule. Headers, query parameters and payload keys all come here.

    Three styles, one answer: a short historical stem anywhere in the name, a long stem
    anywhere in the separator-stripped name, or a short stem as a whole word.
    """
    lowered = name.lower()
    if lowered in DENIED_HEADERS:
        return True
    if _SUBSTRING_PATTERN.search(lowered) is not None:
        return True
    if _COMPOUND_PATTERN.search(_squash(lowered)) is not None:
        return True
    return any(_is_sensitive_word(word) for word in _words(name))


def is_sensitive_header(name: str) -> bool:
    """Whether a header of this name may never have its value recorded."""
    return _is_sensitive_name(name)


def is_sensitive_field(name: str, extra: FrozenSet[str] = frozenset()) -> bool:
    """Whether a payload key of this name may never have its value recorded.

    The same rule as :func:`is_sensitive_header`, plus whatever the deployment added.
    ``extra`` only ever *adds*. A deployment can widen this - ``patient_name``,
    ``diagnosis_notes`` - but nothing can narrow the built-in set.

    An added name is matched with separators stripped from both sides, so an operator
    writes ``patient_name`` once and it covers ``patientName`` and ``Patient-Name`` too.
    Whichever API the deployment is auditing chose those separators, not the operator.
    """
    if _is_sensitive_name(name):
        return True
    squashed = _squash(name.lower())
    return any(_squash(word.lower()) in squashed for word in extra)


def _encodable(text: str) -> str:
    """Return ``text``, or :data:`UNENCODABLE` when UTF-8 cannot represent it.

    ``json.loads('{"u": "\\ud800"}')`` yields a Python ``str`` holding an unpaired
    surrogate. CPython stores it happily; UTF-8 has no encoding for it, so SLCS/1
    rejects it and the audit write fails. Under ``fail_closed`` that is a 500 on every
    audited endpoint, and under ``fail_open`` the attacker's own request is the one
    record that goes missing - so the check has to happen here, before the ledger sees
    it. Same test the canonicalizer applies, so the two can never disagree.
    """
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return UNENCODABLE
    return text


def redact_headers(
    headers: Mapping[str, str], extra_fields: FrozenSet[str] = frozenset()
) -> Dict[str, str]:
    """Return ``headers`` with every sensitive value replaced by :data:`REDACTED`.

    Names are lower-cased so that one ledger holds one spelling: WSGI hands over
    ``HTTP_AUTHORIZATION``, ASGI lower-case bytes and Flask a title-cased mapping, and an
    auditor should not have to know which framework wrote a given row.

    ``extra_fields`` is here for the same reason the built-in list is one list: a
    deployment that adds ``x-tenant-passcode`` means it everywhere, and a name that is
    scrubbed from a body and recorded from a header is the B1 defect wearing a hat.
    """
    result: Dict[str, str] = {}
    for name, value in headers.items():
        lowered = name.lower()
        # Sensitivity is decided on the real name; only what gets *stored* is marked.
        result[_encodable(lowered)] = (
            REDACTED if is_sensitive_field(lowered, extra_fields) else _encodable(str(value))
        )
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
        result: Dict[str, Any] = {}
        for key, item in value.items():
            # A key is canonicalized too - SLCS/1 sorts by UTF-16 code unit and rejects
            # an unencodable one - so the name is matched against the real text and only
            # the *stored* spelling is marked.
            name = str(key)
            result[_encodable(name)] = (
                REDACTED if is_sensitive_field(name, extra) else _redact(item, depth + 1, extra)
            )
        return result
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
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        return _encodable(value)
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
        """Redact a header mapping, including whatever names this deployment added."""
        return redact_headers(headers, self.extra_sensitive_fields)

    def payload(self, value: Any) -> Any:
        """Redact and sanitise an arbitrary payload."""
        return redact_payload(value, self.extra_sensitive_fields)

    def query(self, params: Mapping[str, Any]) -> Dict[str, List[Any]]:
        """Redact a query-string mapping.

        **Every value is a list**, redacted or not. HTTP query parameters are inherently
        multi-valued and the middlewares hand over a multi-dict, so ``?page=2`` arrives as
        ``["2"]``; redaction used to collapse a sensitive one to the bare string
        ``"[REDACTED]"``, and the ledger then held two types under one key. Anything
        reading it had to branch on the type of every value it looked at. Wrapping a flat
        value rather than unwrapping a single-element list is the direction that stays
        correct when ``?tag=a&tag=b`` shows up.

        Arity survives redaction: how many values were sent is auditable, what they were
        is not.

        Names keep their case: unlike headers, query parameters are case-sensitive.
        """
        result: Dict[str, List[Any]] = {}
        for name, value in params.items():
            values = list(value) if isinstance(value, (list, tuple)) else [value]
            if is_sensitive_field(name, self.extra_sensitive_fields):
                result[name] = [REDACTED] * len(values)
            else:
                result[name] = [self.payload(item) for item in values]
        return result

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
                # Client-written text on its way into the ledger, so it goes through the
                # same encodability check every other captured value does.
                return _encodable(parts[-self.trusted_proxy_count])
        return None if peer is None else _encodable(peer)


def _is_json(content_type: Optional[str]) -> bool:
    """Whether this ``Content-Type`` names JSON, ignoring any parameters."""
    if not content_type:
        return False
    return content_type.split(";", 1)[0].strip().lower() == "application/json"
