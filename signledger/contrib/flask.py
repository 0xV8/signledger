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
"""

from typing import Any, Callable, Dict, Iterable, Optional, Tuple

from flask import Blueprint, Flask, Response, g, jsonify, request

from signledger.contrib import FAIL_CLOSED, AuditRecorder, build_audit_record
from signledger.contrib.redaction import RedactionPolicy
from signledger.core.exceptions import ConfigurationError

__all__ = ["INTEGRATION", "SignLedgerAudit", "audit_blueprint"]

#: Name recorded in every entry this integration writes.
INTEGRATION = "flask"

#: Key under which the extension registers itself in ``app.extensions``.
EXTENSION_KEY = "signledger"

#: Default ceiling on how many entries one inspection request may return.
DEFAULT_MAX_PAGE_SIZE = 100

_BODY_KEY = "_signledger_body"
_REASON_KEY = "_signledger_body_reason"


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
        """Attach the request hooks to ``app``."""
        app.before_request(self._before_request)
        app.after_request(self._after_request)
        app.extensions[EXTENSION_KEY] = self

    # -- hooks -------------------------------------------------------------------------

    def _skip(self, path: str) -> bool:
        return any(path.startswith(prefix) for prefix in self.skip_paths)

    def _before_request(self) -> None:
        """Decide about the body *before* the view has a chance to consume it.

        The declared ``Content-Length`` is checked first and the body is only read when
        the policy allows it. Flask caches what is read, so the view still gets its own
        ``request.get_json()`` - capturing for audit must not consume the request.
        """
        if self._skip(request.path):
            return
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
        """Write the audit entry once the real status code is known."""
        if self._skip(request.path):
            return response
        policy = self.recorder.redaction
        record = build_audit_record(
            integration=INTEGRATION,
            method=request.method,
            path=request.path,
            status=response.status_code,
            client_ip=policy.client_ip(
                peer=request.remote_addr,
                forwarded_for=request.headers.get("X-Forwarded-For"),
            ),
            headers=policy.headers(dict(request.headers)),
            query=policy.query(request.args.to_dict(flat=False)),
            body=getattr(g, _BODY_KEY, None),
            body_reason=getattr(g, _REASON_KEY, "disabled"),
            user=self._user(),
        )
        # Raises AuditWriteError under fail_closed, which Flask turns into a 500. An
        # unaudited request must not be able to answer 200.
        self.recorder.record(record, {"integration": INTEGRATION})
        return response

    def _user(self) -> Optional[str]:
        """The acting identity, as a string.

        Coerced because ``request.user.pk`` is an integer in most real applications and
        the ledger stores one type per field.
        """
        if self.user_resolver is None:
            return None
        resolved = self.user_resolver(request)
        return None if resolved is None else str(resolved)


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

