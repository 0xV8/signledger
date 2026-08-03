"""Flask integration.

Install the extension and every request becomes a hash-chained ledger entry::

    from flask import Flask
    from signledger.backends.sqlite import SQLiteBackend
    from signledger.contrib.flask import SignLedgerAudit
    from signledger.core.ledger import Ledger

    app = Flask(__name__)
    ledger = Ledger(backend=SQLiteBackend("audit.db"))
    SignLedgerAudit(app, ledger=ledger)

What v1 got wrong here, and what is different:

* v1's module raised ``NameError`` on import - a bare ``Tuple`` annotation sat at
  class-body scope - so nobody had ever executed a line of it. There is a subprocess test
  that imports this module in a fresh interpreter for exactly that reason.
* Headers, query parameters and bodies go through
  :class:`~signledger.contrib.redaction.RedactionPolicy` before they go anywhere near the
  ledger, and body capture is off unless asked for.
* A failed audit write fails the request by default. An unaudited request that returns
  ``200`` is a silent hole in the trail.

Build the ledger per worker, after the fork
-------------------------------------------

A ``Ledger`` owns live handles - a psycopg connection, a SQLite file descriptor. Those do
not survive ``fork()`` being shared: the kernel duplicates the descriptor, so every worker
ends up driving **one** connection. The snippet above builds the ledger at import time,
which is safe only because gunicorn imports the application separately in each worker.
Add ``--preload`` and it is built once in the master instead, before the fork, and every
worker inherits it. Measured on PostgreSQL under ``gunicorn --preload -w 4``: protocol
desync (``the last operation didn't produce records``), 60-second hangs, and a ledger
holding more entries than there were successful responses.

So: **construct the ledger inside an application factory**, which each worker runs for
itself::

    def create_app():
        app = Flask(__name__)
        SignLedgerAudit(app, ledger=Ledger(backend=SQLiteBackend("audit.db")))
        return app

    # gunicorn "create_app()"

If the application object has to exist before the fork - ``--preload``, or a module-level
``app`` some tooling insists on - replace the ledger in a post-fork hook, so the inherited
one is never used::

    # gunicorn.conf.py
    preload_app = True

    def post_fork(server, worker):
        from myapp import app
        audit = app.extensions["signledger"]
        audit.recorder.ledger = Ledger(backend=SQLiteBackend("audit.db"))

The same applies to any pre-fork work: ``uwsgi`` without ``--lazy-apps``, and
``os.fork()`` in a custom supervisor.

What an entry means
-------------------

Every entry carries a ``kind`` and a ``request_id``. ``kind="request"`` is the one this
extension writes per request; ``kind="response.completion"`` settles a streamed one.

``completion``: ``"final"`` or ``"provisional"``
-------------------------------------------------

``after_request`` runs when the response *object* is finished, which for a streamed body
is before a single byte of it exists: Flask hands the iterable to the WSGI server
afterwards, and the server consumes it long after this extension has stopped looking.
The status recorded for a streamed response is therefore a *promise*, and a generator
that raised on its second chunk was permanently in the ledger as a plain ``200``.

Entries are append-only and are not rewritten to fix that. A streamed response is instead
recorded as ``completion="provisional"``, and a second entry - ``kind`` of
``"response.completion"``, same ``request_id`` - records how it ended, when the WSGI
server closes the body:

* ``outcome="complete"`` - the iterable ran to the end;
* ``outcome="incomplete"`` - it did not, plus ``error`` naming the exception if one was
  raised. No ``error`` field means nobody raised: the client hung up.

A response whose body is already built - the ordinary case - is ``completion="final"``
and gets no second entry. So is a response whose body a WSGI server will never send at
all (``HEAD``, ``204``, ``304``, ``1xx``), because nothing would ever settle it.

A settlement that cannot be written is counted on ``recorder.audit_failures`` rather than
raised: the client has had the response since the first chunk, and there is no failing
closed after that.

A request that fails is still a request
---------------------------------------

``after_request`` does not run when an exception leaves the application, which is what
``TESTING``, ``DEBUG`` and ``PROPAGATE_EXCEPTIONS`` each arrange - and two of those are
what people develop and run their test suites against. The client received a real 500 over
TCP and the ledger gained nothing: ``fail_closed`` could not help, because the failure is
upstream of the audit write, and ``check_integrity()`` reported ``ok`` on a chain with no
memory of the request.

``teardown_request`` is the backstop. It fires only when there is an exception and nothing
has been recorded yet, so it cannot double-count the ordinary path, and it records
``status=500`` because that is what the server sends. ``got_request_exception`` is what
lets both paths name the cause: ``after_request`` is handed a response and no exception,
so a handled 500 used to be recorded with no ``error`` field at all, and
``WHERE error IS NOT NULL`` came back structurally empty however many times a view had
blown up.
"""

import contextlib
import functools
import uuid
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

from flask import Blueprint, Flask, Response, g, jsonify, request
from flask.signals import got_request_exception

from signledger import contrib as _shared
from signledger.contrib import (
    FAIL_CLOSED,
    OUTCOME_COMPLETE,
    OUTCOME_INCOMPLETE,
    AuditRecorder,
    AuditWriteError,
    SettlingBody,
    build_audit_record,
    is_skipped,
    will_send_body,
)
from signledger.contrib.redaction import RedactionPolicy
from signledger.core.exceptions import ConfigurationError

__all__ = [
    "COMPLETION_FINAL",
    "COMPLETION_PROVISIONAL",
    "INTEGRATION",
    "KIND_COMPLETION",
    "KIND_REQUEST",
    "SignLedgerAudit",
    "audit_blueprint",
]

#: Name recorded in every entry this integration writes.
INTEGRATION = "flask"

#: ``kind`` of the entry written for one request.
KIND_REQUEST = _shared.KIND_REQUEST

#: ``kind`` of the entry that settles a provisional one.
KIND_COMPLETION = _shared.KIND_COMPLETION

#: The response was complete when its entry was written.
COMPLETION_FINAL = _shared.COMPLETION_FINAL

#: The status was recorded before the body existed; a settlement entry follows.
COMPLETION_PROVISIONAL = _shared.COMPLETION_PROVISIONAL

#: Key under which the extension registers itself in ``app.extensions``.
EXTENSION_KEY = "signledger"

#: Default ceiling on how many entries one inspection request may return.
DEFAULT_MAX_PAGE_SIZE = 100

_BODY_KEY = "_signledger_body"
_REASON_KEY = "_signledger_body_reason"

#: Set once this request is one we audit, so the teardown backstop can tell a request it
#: must account for from one that was skipped or never reached ``before_request``.
_AUDITED_KEY = "_signledger_audited"

#: Set once an entry exists for this request, so the backstop never writes a second one.
_RECORDED_KEY = "_signledger_recorded"

#: The exception Flask is about to turn into a 500, captured while it is still knowable.
_ERROR_KEY = "_signledger_error"


class SignLedgerAudit:
    """Flask extension that writes one ledger entry per request.

    :param app: application to configure immediately, or ``None`` to use
        :meth:`init_app` later (the application-factory pattern).
    :param ledger: a :class:`~signledger.core.ledger.Ledger`.
    :param redaction: what may be recorded. Defaults to the strict policy - body capture
        off, no proxies trusted.
    :param audit_failure_policy: ``"fail_closed"`` (default) or ``"fail_open"``.
    :param skip_paths: path prefixes that are not audited, e.g. a health check.
    :param user_resolver: called with the Flask request, returns the acting identity.
    """

    def __init__(
        self,
        app: Optional[Flask] = None,
        *,
        ledger: Any = None,
        redaction: Optional[RedactionPolicy] = None,
        audit_failure_policy: str = FAIL_CLOSED,
        skip_paths: Iterable[str] = (),
        user_resolver: Optional[Callable[[Any], Any]] = None,
    ) -> None:
        self.recorder = AuditRecorder(
            ledger, redaction=redaction, audit_failure_policy=audit_failure_policy
        )
        self.skip_paths: Tuple[str, ...] = tuple(skip_paths)
        self.user_resolver = user_resolver
        if app is not None:
            self.init_app(app)

    def init_app(self, app: Flask) -> None:
        """Attach the request hooks to ``app``.

        ``teardown_request`` is the backstop and ``got_request_exception`` is what makes
        it able to say *what* failed. Neither is decoration: ``after_request`` does not run
        at all when an exception leaves the application, which every one of
        ``TESTING``, ``DEBUG`` and ``PROPAGATE_EXCEPTIONS`` arranges - so the client
        received a real 500 over TCP and the ledger gained nothing. ``fail_closed`` cannot
        help there, because the failure is upstream of the audit write.
        """
        app.before_request(self._before_request)
        app.after_request(self._after_request)
        app.teardown_request(self._teardown_request)
        # weak=False: the receiver is a bound method, and a weak reference to one dies
        # immediately, which would leave the signal connected to nothing.
        got_request_exception.connect(self._note_exception, app, weak=False)
        app.extensions[EXTENSION_KEY] = self

    # -- hooks -------------------------------------------------------------------------

    def _skip(self, path: str) -> bool:
        return is_skipped(path, self.skip_paths)

    def _before_request(self) -> None:
        """Decide about the body *before* the view has a chance to consume it.

        The declared ``Content-Length`` is checked first and the body is only read when
        the policy allows it. Flask caches what is read, so the view still gets its own
        ``request.get_json()`` - capturing for audit must not consume the request.
        """
        if self._skip(request.path):
            return
        setattr(g, _AUDITED_KEY, True)
        policy = self.recorder.redaction
        decision = policy.body_decision(
            content_length=request.content_length,
            content_type=request.content_type,
        )
        setattr(
            g,
            _BODY_KEY,
            policy.decode_body(request.get_data(cache=True)) if decision.capture else None,
        )
        setattr(g, _REASON_KEY, decision.reason)

    def _after_request(self, response: Response) -> Response:
        """Write the audit entry once the real status code is known.

        For a streamed response that is *all* that is known: the body has not been
        generated yet and will not be until the WSGI server pulls it, after this
        extension has stopped looking. Such an entry says so, and is settled by a second
        one when the server closes the body.
        """
        if self._skip(request.path):
            return response
        policy = self.recorder.redaction
        context: Dict[str, Any] = {
            "request_id": uuid.uuid4().hex,
            "method": request.method,
            "path": request.path,
            "status": response.status_code,
            # Everything the settlement entry needs, captured while the request context
            # still exists. It is torn down before the WSGI server closes the body, and
            # the settlement used to be assembled by hand from what little survived - so
            # it named no client, no user and no headers, on precisely the half of the
            # request that went wrong.
            "shape": {
                "client_ip": policy.client_ip(
                    peer=request.remote_addr,
                    forwarded_for=request.headers.get("X-Forwarded-For"),
                ),
                "headers": policy.headers(dict(request.headers)),
                "query": policy.query(request.args.to_dict(flat=False)),
                "body": getattr(g, _BODY_KEY, None),
                "body_reason": getattr(g, _REASON_KEY, "disabled"),
                "user": self._user(),
            },
        }
        streamed = _will_stream(response) and will_send_body(
            request.method, response.status_code
        )
        record = build_audit_record(
            integration=INTEGRATION,
            method=request.method,
            path=request.path,
            status=response.status_code,
            error=getattr(g, _ERROR_KEY, None),
            kind=KIND_REQUEST,
            **context["shape"],
            request_id=context["request_id"],
            # Says whether the status above is a fact or a promise. Never absent: an
            # auditor must not have to know which version of this extension - or which
            # framework - wrote the entry, which is why the builder owns the default.
            completion=COMPLETION_PROVISIONAL if streamed else COMPLETION_FINAL,
        )
        # Raises AuditWriteError under fail_closed, which Flask turns into a 500. An
        # unaudited request must not be able to answer 200.
        self.recorder.record(record, {"integration": INTEGRATION})
        setattr(g, _RECORDED_KEY, True)
        if streamed:
            response.response = SettlingBody(
                response.response, functools.partial(self._settle, context)
            )
        return response

    def _settle(
        self, context: Dict[str, Any], complete: bool, error: Optional[str]
    ) -> None:
        """Record how a promised response actually ended."""
        record = build_audit_record(
            integration=INTEGRATION,
            method=context["method"],
            path=context["path"],
            status=context["status"],
            error=error,
            kind=KIND_COMPLETION,
            request_id=context["request_id"],
            # The settlement's own status is a fact by construction: it is written once
            # the body has stopped moving.
            completion=COMPLETION_FINAL,
            outcome=OUTCOME_COMPLETE if complete else OUTCOME_INCOMPLETE,
            **context["shape"],
        )
        # No failing closed here: the client has had the response since the first chunk.
        # The recorder has counted the failure and named it, so this is not a swallowed
        # error - it is one with nowhere left to go.
        with contextlib.suppress(AuditWriteError):
            self.recorder.record(record, {"integration": INTEGRATION})

    def _note_exception(self, sender: Any, exception: BaseException, **extra: Any) -> None:
        """Remember what failed, while it is still knowable.

        ``after_request`` receives a response and no exception, so the entry it wrote for
        a handled 500 named no cause at all - ``WHERE error IS NOT NULL`` came back
        structurally empty on Flask however many times a view had blown up. Flask sends
        this signal for every unhandled exception, whether it goes on to render a 500 or
        to propagate, so one receiver covers both paths.
        """
        setattr(g, _ERROR_KEY, type(exception).__name__)

    def _teardown_request(self, exception: Optional[BaseException]) -> None:
        """Audit a request whose exception escaped before ``after_request`` could run.

        Only fires when there is an exception *and* nothing has been recorded yet, so it
        cannot double-count the ordinary path. The status is 500 because that is what the
        server sends once an exception reaches it - which is the point: the client got one,
        and until this existed the ledger had no memory of the request at all.

        Failing closed here would be theatre. The response is already being assembled by
        the server, so raising would replace one 500 with another while losing the record
        a second time; the recorder counts and names the failure instead.
        """
        if exception is None:
            return
        if not getattr(g, _AUDITED_KEY, False) or getattr(g, _RECORDED_KEY, False):
            return
        policy = self.recorder.redaction
        record = build_audit_record(
            integration=INTEGRATION,
            method=request.method,
            path=request.path,
            status=500,
            client_ip=policy.client_ip(
                peer=request.remote_addr,
                forwarded_for=request.headers.get("X-Forwarded-For"),
            ),
            headers=policy.headers(dict(request.headers)),
            query=policy.query(request.args.to_dict(flat=False)),
            body=getattr(g, _BODY_KEY, None),
            body_reason=getattr(g, _REASON_KEY, "disabled"),
            user=self._user(),
            error=type(exception).__name__,
            kind=KIND_REQUEST,
            completion=COMPLETION_FINAL,
        )
        setattr(g, _RECORDED_KEY, True)
        with contextlib.suppress(AuditWriteError):
            self.recorder.record(record, {"integration": INTEGRATION})

    def _user(self) -> Optional[str]:
        """The acting identity, as a string.

        Coerced because ``request.user.pk`` is an integer in most real applications and
        the ledger stores one type per field.
        """
        if self.user_resolver is None:
            return None
        resolved = self.user_resolver(request)
        return None if resolved is None else str(resolved)


def _will_stream(response: Response) -> bool:
    """Whether this response's body still has to be produced.

    ``Response.is_streamed`` is the obvious test and it is the wrong one. Werkzeug defines
    it as "``len(self.response)`` raises", and ``make_response`` runs every
    ``HTTPException`` through ``Response.force_type``, which replaces the body list with a
    ``werkzeug.wsgi.ClosingIterator``. That has no ``__len__``, so every 400, 404, 405 and
    500 answered by an error handler looked streamed: each was recorded ``provisional`` -
    untrue, the body went out with a ``Content-Length`` - and then earned a second
    settlement entry saying it completed. Two signed rows per error response, against a
    store that by design cannot be pruned, from any anonymous caller hitting bad URLs.

    A declared ``Content-Length`` means the body was fully built before the status went
    out. That is the same test :mod:`signledger.contrib.fastapi` applies to
    ``http.response.start``, which is the point: the two integrations answered this
    question differently, and an auditor's query cannot depend on which one wrote the row.
    """
    return response.content_length is None


def audit_blueprint(
    ledger: Any,
    *,
    authenticator: Optional[Callable[[Any], bool]],
    url_prefix: str = "/_audit",
    max_page_size: int = DEFAULT_MAX_PAGE_SIZE,
) -> Blueprint:
    """A blueprint exposing the audit trail over HTTP, authenticated by default.

    ``authenticator`` has no default. Leaving it out is a
    :class:`~signledger.core.exceptions.ConfigurationError` at registration time, because
    an inspection endpoint that is open by omission is both a denial-of-service surface
    (every request can walk the whole ledger) and an information-disclosure one.

    Even an authenticated caller cannot ask for the whole ledger: ``limit`` is capped at
    ``max_page_size``, and rows carry identifiers and digests only - never entry
    payloads, which are the part with the disclosure risk in them.
    """
    if authenticator is None:
        raise ConfigurationError(
            "audit_blueprint requires an authenticator; an unauthenticated audit "
            "endpoint exposes the whole ledger to anyone who can reach it"
        )
    if max_page_size < 1:
        raise ConfigurationError(f"max_page_size must be at least 1, got {max_page_size}")

    blueprint = Blueprint("signledger_audit", __name__, url_prefix=url_prefix)

    @blueprint.before_request
    def _authenticate() -> Optional[Response]:
        if authenticator(request):
            return None
        return _error("authentication required", 401)

    @blueprint.get("/status")
    def status() -> Response:
        """Cheap counters. Deliberately does *not* verify - see ``/verify``."""
        snapshot = ledger.get_stats_snapshot()
        snapshot.update(
            {
                "entry_count": ledger.count(),
                "capabilities": sorted(ledger.backend.capabilities()),
            }
        )
        return jsonify(snapshot)

    @blueprint.get("/verify")
    def verify() -> Response:
        """Run a real verification. O(n) in the ledger, which is why it is its own route.

        ``assurance`` is reported alongside ``ok`` on purpose: a chain-only deployment
        that answers ``ok`` has still only detected accidental corruption and naive edits.
        """
        report = ledger.check_integrity()
        return jsonify(
            {
                "ok": report.ok,
                "entries_verified": report.entries_verified,
                "assurance": report.assurance.value,
                # Failure messages carry sequences and digests only, never payloads.
                "failures": [failure.message for failure in report.failures],
            }
        )

    @blueprint.get("/entries")
    def entries() -> Response:
        try:
            limit = _positive_int(request.args.get("limit"), max_page_size)
            offset = _non_negative_int(request.args.get("offset"), 0)
        except ValueError as err:
            return _error(str(err), 400)
        rows = ledger.iter_entries(limit=min(limit, max_page_size), offset=offset)
        return jsonify({"entries": [_summarise(entry) for entry in rows]})

    return blueprint


def _summarise(entry: Any) -> Dict[str, Any]:
    """Identifiers and digests only.

    Enough to prove the chain and locate a record; not enough to be a data breach if the
    credential guarding the endpoint is ever weak.
    """
    return {
        "sequence": entry.sequence,
        "entry_id": entry.entry_id,
        "timestamp": entry.timestamp.isoformat(),
        "hash": entry.hash,
        "previous_hash": entry.previous_hash,
        "key_id": entry.key_id,
    }


def _error(message: str, code: int) -> Response:
    response = jsonify({"error": message})
    response.status_code = code
    return response


def _positive_int(raw: Optional[str], default: int) -> int:
    value = _as_int(raw, default)
    if value < 1:
        raise ValueError("limit must be at least 1")
    return value


def _non_negative_int(raw: Optional[str], default: int) -> int:
    value = _as_int(raw, default)
    if value < 0:
        raise ValueError("offset must be zero or greater")
    return value


def _as_int(raw: Optional[str], default: int) -> int:
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        raise ValueError("expected an integer") from None


#: The shared implementation, under the name this module used to define it with, for
#: anything that reached in and patched it.
_SettlingBody: Any = SettlingBody
