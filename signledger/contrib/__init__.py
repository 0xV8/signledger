"""Framework integrations for Flask, FastAPI and Django.

This module holds the half that is identical in all three: what an audit record looks
like, and what happens when writing one fails.

**Nothing here imports a web framework.** ``signledger.contrib.flask``,
``.fastapi`` and ``.django`` are reached by path, so ``import signledger.contrib`` keeps
working on a machine that has none of them installed.

The v1 versions of these integrations were 1,685 lines of code that could never run: each
one was shadowed by an empty package of the same name, so the module a user imported was
not the module anyone had tested. What did reach production wrote ``Authorization``
headers and whole request bodies into an immutable store. Both failures share one cause -
nothing was executed by a test - which is why every line below is reached by one.
"""

import uuid
from typing import Any, Dict, FrozenSet, Iterable, Mapping, Optional

from signledger.contrib.redaction import RedactionPolicy, redact_payload
from signledger.core.exceptions import ConfigurationError, SignLedgerError

#: What an entry describes. ``request`` is the record of one request; a settlement entry
#: written when a streamed body finishes carries ``response.completion``.
KIND_REQUEST = "request"
KIND_COMPLETION = "response.completion"

#: Whether ``status`` is a fact or a promise. A streamed response is audited at its status
#: line, before the body exists, so its first entry is provisional.
COMPLETION_FINAL = "final"
COMPLETION_PROVISIONAL = "provisional"

#: How a promised response actually ended. Carried only by settlement entries.
OUTCOME_COMPLETE = "complete"
OUTCOME_INCOMPLETE = "incomplete"

#: Statuses whose body no server sends, so nothing will ever settle a promise made for
#: one. Werkzeug's ``Response.get_app_iter`` and Starlette agree on this set.
_BODYLESS_STATUSES: FrozenSet[int] = frozenset({204, 304})

__all__ = [
    "AUDIT_FAILURE_POLICIES",
    "COMPLETION_FINAL",
    "COMPLETION_PROVISIONAL",
    "KIND_COMPLETION",
    "KIND_REQUEST",
    "OUTCOME_COMPLETE",
    "OUTCOME_INCOMPLETE",
    "FAIL_CLOSED",
    "FAIL_OPEN",
    "AuditRecorder",
    "AuditWriteError",
    "RedactionPolicy",
    "SettlingBody",
    "build_audit_record",
    "is_skipped",
    "will_send_body",
]


class SettlingBody:
    """A streamed response body that reports how it ended.

    Shared, because the moment a streamed response's outcome becomes knowable is the same
    moment in both server models: WSGI requires ``close()`` "upon completion of the current
    request, whether the request was completed normally or terminated early due to an
    application error", and Django registers the same call through ``_resource_closers``.
    That call happens after the framework has torn the request down, so everything the
    settlement needs is captured in the closure and nothing here touches request state.

    Wrapping the body rather than using a close callback is deliberate: a callback cannot
    tell a body that finished from one that died.
    """

    def __init__(
        self,
        body: Any,
        settle: Any,
    ) -> None:
        # Not ``Iterable[bytes]``: werkzeug allows a body to yield ``str`` and encodes it
        # later, and every chunk passes through untouched, so its type is genuinely none
        # of this class's business.
        self._body = body
        self._settle = settle
        self._complete = False
        self._error: Optional[str] = None
        self._settled = False

    def __iter__(self) -> Any:
        try:
            yield from self._body
        except Exception as err:
            # Named, not swallowed: it is re-raised into the server unchanged, and the
            # audit record is the only place it would otherwise leave no trace.
            self._error = type(err).__name__
            raise
        self._complete = True

    def close(self) -> None:
        if self._settled:
            # Servers and test clients both close responses more than once. A second
            # settlement entry for one response would be a second claim, not a repeat.
            return
        self._settled = True
        close = getattr(self._body, "close", None)
        if close is not None:
            # A generator holds a file handle open until it is closed. Wrapping a body
            # must not be what leaks it.
            close()
        self._settle(self._complete, self._error)


def will_send_body(method: str, status: int) -> bool:
    """Whether a server will consume this response's body at all.

    Written once because the two integrations that had it wrote it differently: Flask
    excluded ``HEAD`` and FastAPI tested the status alone, so a ``HEAD`` against a
    streaming route produced one final entry on Flask and two entries with a provisional
    first on FastAPI, for byte-identical traffic.

    It matters beyond tidiness. A response whose body is never iterated is never closed
    either, so a provisional entry written for one waits forever for a settlement that
    cannot come - a worse record than the overclaim it was meant to replace.
    """
    return method != "HEAD" and status >= 200 and status not in _BODYLESS_STATUSES


def is_skipped(path: str, prefixes: Iterable[str]) -> bool:
    """Whether ``path`` falls under one of the configured skip prefixes.

    All three integrations used a bare ``str.startswith``, which is not a path match: with
    ``skip_paths=("/health",)`` - the value the README recommends - a request to
    ``/healthz-admin-secret`` returned 200 with real data and wrote no audit entry at all.
    For a library whose product is a complete trail, that is a hole, not a nicety.

    A prefix matches the path itself or anything below it as a **path segment**, so
    ``/health`` covers ``/health`` and ``/health/live`` and nothing else.
    """
    for prefix in prefixes:
        trimmed = prefix.rstrip("/")
        if path == trimmed or path.startswith(trimmed + "/"):
            return True
    return False

#: The audit write must succeed or the request must not. The default.
FAIL_CLOSED = "fail_closed"

#: The request proceeds even when the audit write fails. Availability over auditability -
#: a real trade-off some deployments need, but never the one you get by accident.
FAIL_OPEN = "fail_open"

AUDIT_FAILURE_POLICIES = frozenset({FAIL_CLOSED, FAIL_OPEN})


class AuditWriteError(SignLedgerError):
    """An audit record could not be written and the policy is ``fail_closed``.

    Derives from :class:`~signledger.core.exceptions.SignLedgerError`, so an application
    that already wraps SignLedger in one ``except`` catches this too.
    """


class AuditRecorder:
    """The one place any integration turns a request into a ledger entry.

    Written once and shared by all three frameworks so the fail-open/fail-closed decision
    - the only part of an audit middleware that can quietly destroy the audit trail -
    exists in exactly one testable place.
    """

    def __init__(
        self,
        ledger: Any,
        *,
        redaction: Optional[RedactionPolicy] = None,
        audit_failure_policy: str = FAIL_CLOSED,
    ) -> None:
        if audit_failure_policy not in AUDIT_FAILURE_POLICIES:
            raise ConfigurationError(
                f"audit_failure_policy must be one of {sorted(AUDIT_FAILURE_POLICIES)}, "
                f"got {audit_failure_policy!r}"
            )
        self.ledger = ledger
        # A recorder with no policy still redacts. Redaction is not a feature that can be
        # left switched off by forgetting to pass an argument.
        self.redaction = RedactionPolicy() if redaction is None else redaction
        self.audit_failure_policy = audit_failure_policy
        self.audit_failures = 0
        self.last_audit_error: Optional[str] = None

    @property
    def fail_closed(self) -> bool:
        """Whether a failed audit write must fail the request."""
        return self.audit_failure_policy == FAIL_CLOSED

    def record(
        self, data: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Any]:
        """Append one audit record, honouring the configured failure policy.

        Returns the stored entry, or ``None`` when the write failed under ``fail_open``.
        """
        try:
            return self.ledger.append(data, metadata)
        except Exception as err:  # noqa: BLE001 - re-raised or counted, never swallowed
            # Deliberately broad, and deliberately not silent. ``fail_open`` promises the
            # request survives an audit failure; a version that only caught
            # SignLedgerError would break that promise on the first driver bug. Under
            # ``fail_closed`` the error is re-raised, and under ``fail_open`` it is
            # counted and named on the recorder - never swallowed, which is exactly what
            # v1 did.
            self.audit_failures += 1
            self.last_audit_error = type(err).__name__
            if self.fail_closed:
                # No payload in the message: contract section 10. The chained cause
                # carries the detail for whoever is holding the traceback.
                raise AuditWriteError(
                    "the audit record could not be written; refusing the request"
                ) from err
            return None


def build_audit_record(
    *,
    integration: str,
    method: str,
    path: str,
    status: Optional[int],
    client_ip: Optional[str],
    headers: Mapping[str, str],
    query: Optional[Any] = None,
    body: Any = None,
    body_reason: str = "disabled",
    user: Optional[str] = None,
    error: Optional[str] = None,
    kind: str = KIND_REQUEST,
    request_id: Optional[str] = None,
    completion: str = COMPLETION_FINAL,
    outcome: Optional[str] = None,
) -> Dict[str, Any]:
    """Assemble the audit payload every integration writes.

    One shape for all three frameworks, so a query an auditor writes against a Flask
    service still works against the Django one.

    ``headers``, ``query``, ``body`` and ``client_ip`` arrive already redacted. The
    remaining free-text fields do **not**: ``user`` is whatever a deployment's
    ``USER_RESOLVER`` returned, and real resolvers read it straight out of the request, so
    it is client-controlled text on its way into an append-only store. ``path`` reaches
    here the same way. That is the last open half of the unpaired-surrogate denial of
    service: ``json.loads`` accepts ``"\\ud800"``, CPython holds it, UTF-8 cannot encode
    it, and SLCS/1 refuses the entry - so under ``fail_closed`` a single anonymous request
    500s every audited endpoint, and under ``fail_open`` the attacker deletes its own
    audit record. Sanitising every path *into* the record is one choke point instead of
    three ``_user()`` implementations that must each remember.

    :func:`~signledger.contrib.redaction.redact_payload` is the same function the bodies
    go through, so the two can never disagree about what is encodable. On a string it only
    ever substitutes the ``[UNENCODABLE]`` marker; it never redacts a value for its own
    sake, because these fields are names of things, not values of them.
    """
    record: Dict[str, Any] = {
        "integration": redact_payload(integration),
        "method": redact_payload(method),
        "path": redact_payload(path),
        "status": status,
        "client_ip": client_ip,
        "headers": dict(headers),
        "query": query,
        # Either the captured body or a marker naming the rule that refused it. An
        # absent field would be indistinguishable from a middleware nobody installed.
        "body": body if body_reason == "captured" else f"[OMITTED:{body_reason}]",
        "user": None if user is None else redact_payload(user),
        # These three used to be stamped on by the Flask and FastAPI middlewares *after*
        # this function returned, and the Django one did not stamp them at all - so an
        # auditor querying on `kind` saw every Flask and FastAPI request and no Django
        # one, against the promise three paragraphs up. Two of three call sites adding
        # the same fields is exactly the divergence a shared builder exists to prevent,
        # so they live here now and no integration can forget them.
        "kind": kind,
        "request_id": uuid.uuid4().hex if request_id is None else request_id,
        "completion": completion,
    }
    if error is not None:
        # Present only when there was one. A field that is always there and always null
        # teaches people to stop reading it.
        record["error"] = redact_payload(error)
    if outcome is not None:
        # Only settlement entries have one, for the same reason.
        record["outcome"] = redact_payload(outcome)
    return record
