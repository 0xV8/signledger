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

The settlement does not depend on the server keeping its side of PEP 3333
-------------------------------------------------------------------------

PEP 3333 requires the server to call ``close()`` on the iterable it was handed, and the
case above is written against that promise. ``werkzeug.serving`` breaks it on the one
event the ``"the client hung up"`` sentence exists for: ``run_wsgi`` drains the request
socket and closes the application iterable inside a single ``finally``, drain first, and
a peer RST makes the drain raise before the close is reached. Measured against a real
server - a 400x64KB stream, the client reading 4KB and then resetting the connection -
the request entry stayed ``provisional`` with no settlement, permanently.

So a streamed body is settled when it is closed **or** when it is dropped, through one
guard that cannot fire twice. The second half is a finalizer and is described as one:
under CPython the last reference to the body dies with the server's request frame, so the
settlement is written there and then, in that thread - measured over a real socket, not
reasoned about. It is a backstop, not a guarantee. An implementation without reference
counting settles whenever it next collects, and a body still alive at process exit
settles never. The two alternatives were measured and neither works:
``Response.call_on_close`` is driven by the same ``close()`` the server skipped, and
``teardown_request`` fires before the first byte of a streamed body exists, which is
before its outcome is knowable.

The inspection endpoints are not part of the trail
--------------------------------------------------

:func:`audit_blueprint` records the path it was mounted at as one this extension does not
audit, so reading the ledger is not a write to it. Measured before that: every inspection
request appended an entry, including ``/verify`` - which reported ``entries_verified=5``
while appending the sixth, making each verification cost more than the last - and
including the ``401`` returned to a caller with no credential, which let an anonymous
prober append a signed row to an append-only store on every rejected attempt.

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
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from flask import Blueprint, Flask, Response, current_app, g, jsonify, request
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

#: Key under which the paths :func:`audit_blueprint` was mounted at are kept, so the
#: extension can leave them out of the trail. They live on the application rather than on
#: the extension because either half may be installed first - an application factory is
#: free to register the blueprint before it constructs the extension - and a deployer who
#: happened to pick the order that audits itself would have no way of telling.
INSPECTION_PREFIXES_KEY = "signledger_inspection_prefixes"

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
        """Whether this request is outside the trail.

        The configured prefixes, plus wherever the inspection blueprint was mounted on
        the application handling this request. The extension can be shared between
        applications - one instance, several ``init_app`` calls - so the mount points are
        read from the application in play rather than held on ``self``.
        """
        return is_skipped(path, self.skip_paths) or is_skipped(
            path, _inspection_prefixes(current_app)
        )

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
            response.response = _AbandonableBody(
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


class _AbandonableBody(SettlingBody):
    """A streamed body that settles when it is closed *or* when it is let go of.

    The shared :class:`~signledger.contrib.SettlingBody` settles in ``close()``, which
    PEP 3333 obliges a server to call "upon completion of the current request, whether the
    request was completed normally or terminated early". ``werkzeug.serving`` does not,
    on precisely the case a settlement matters most for: ``WSGIRequestHandler.run_wsgi``
    puts the request-socket drain (``self.rfile.read(10_000_000)``) and the
    ``application_iter.close()`` in one ``finally`` block with the drain first, and a peer
    RST makes the drain raise before the close is reached. Measured against a real server
    over a real socket - a 400x64KB body, the client reading 4KB and then resetting with
    ``SO_LINGER(1, 0)`` - the request entry was left ``provisional`` with no settlement
    entry, for as long as the ledger exists.

    ``close()`` already refuses to settle twice, so routing the finalizer through it is
    the whole implementation: whichever of the two happens first writes the entry and the
    other becomes a no-op. A body dropped mid-iteration has ``_complete`` false and no
    error, which is the documented shape of "the client hung up"; one dropped after the
    last chunk settles ``complete``, which is what happened.

    This is a backstop and not a guarantee, and it is the only one available: measured on
    the same RST, ``Response.call_on_close`` never fires - it hangs off the same
    ``close()`` the server skipped - and ``teardown_request`` fires with no exception
    *before* the body has produced a byte, which is before there is an outcome to record.
    Finalization is what is left. Under CPython the last reference dies with the server's
    request frame and the entry is written there, in that thread, before the next request
    is accepted; an implementation without reference counting settles whenever it next
    collects, and a body still alive when the process exits settles never.
    """

    def __del__(self) -> None:
        self.close()


def _inspection_prefixes(app: Flask) -> List[str]:
    """Paths on ``app`` that :func:`audit_blueprint` answers on, and this extension does
    not audit."""
    prefixes: List[str] = app.extensions.setdefault(INSPECTION_PREFIXES_KEY, [])
    return prefixes


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

    Registering this blueprint tells :class:`SignLedgerAudit` on the same application not
    to audit the path it was mounted at, so that reading the trail is not a write to it.
    That is not something a deployer has to remember, and it is not something they could
    get right by hand either: ``skip_paths=("/_audit",)`` would be wrong the moment
    ``url_prefix`` or ``register_blueprint(..., url_prefix=...)`` moved the endpoints.
    """
    if authenticator is None:
        raise ConfigurationError(
            "audit_blueprint requires an authenticator; an unauthenticated audit "
            "endpoint exposes the whole ledger to anyone who can reach it"
        )
    if max_page_size < 1:
        raise ConfigurationError(f"max_page_size must be at least 1, got {max_page_size}")

    blueprint = Blueprint("signledger_audit", __name__, url_prefix=url_prefix)

    @blueprint.record
    def _keep_out_of_the_trail(state: Any) -> None:
        """Claim the mounted prefix as unaudited, at registration.

        ``state.url_prefix`` rather than the argument above: ``register_blueprint`` may
        override it, and the path the endpoints actually answer on is the only one worth
        excluding. ``record`` rather than ``record_once`` so that every mount is claimed -
        one blueprint may be registered on several applications, or twice on one under
        different names, and a mount that went unclaimed would audit itself again.

        Every inspection request used to append an entry - measured: ``/status`` 200 +1,
        ``/entries`` 200 +1, ``/entries`` with no credential 401 +1, a malformed ``limit``
        400 +1, and ``/verify`` 200 +1 reporting ``entries_verified=5`` while appending
        the sixth. So each read cost more than the last, and the ``401`` meant an
        unauthenticated caller could append a signed, unprunable row by being turned away
        - against an endpoint whose docstring justifies its authenticator on the grounds
        that an open one is a denial-of-service surface.
        """
        mount = (state.url_prefix or "").rstrip("/")
        if not mount:
            # Excluding "" would exclude every path on the application: `is_skipped`
            # matches a prefix against the start of the path, and every path starts with
            # "/". Silently auditing the inspection endpoints is bad; silently auditing
            # nothing at all is unrecoverable, and a mount at the root cannot be
            # distinguished from the rest of the site by prefix. Refuse instead.
            raise ConfigurationError(
                "audit_blueprint must be mounted under a url_prefix so that its own "
                "requests can be kept out of the trail it serves"
            )
        _inspection_prefixes(state.app).append(mount)

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


#: The body wrapper, under the name this module used to define it with, for anything that
#: reached in and patched it.
_SettlingBody: Any = _AbandonableBody
