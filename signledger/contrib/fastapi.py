"""FastAPI / Starlette integration.

::

    from fastapi import FastAPI
    from signledger.backends.sqlite import SQLiteBackend
    from signledger.contrib.fastapi import SignLedgerAuditMiddleware
    from signledger.core.ledger import Ledger

    app = FastAPI()
    ledger = Ledger(backend=SQLiteBackend("audit.db"))
    app.add_middleware(SignLedgerAuditMiddleware, ledger=ledger)

Three things v1 got wrong, and how each is prevented here.

**The decorator returned ``None``.** Decorating an endpoint therefore *deleted* it - the
route was registered against ``None``. Everything in :func:`audited` returns a real
function, and an async endpoint gets an async wrapper, pinned by a test that asserts
``inspect.iscoroutinefunction`` on the result.

**Credentials went into the ledger.** Headers, query parameters and bodies now go through
:class:`~signledger.contrib.redaction.RedactionPolicy` first, and body capture is off
unless a deployment asks for it.

**The event loop was blocked.** ``Ledger.append`` is synchronous - it can fsync a disk.
Calling it from the middleware coroutine stalls *every* connection in the process for the
duration. Every append here is dispatched with ``run_in_threadpool``.

This is a **pure ASGI middleware**, not a ``BaseHTTPMiddleware`` subclass. That is what
lets it observe the request body as the endpoint reads it, instead of draining the stream
and leaving the endpoint with nothing - and what lets ``fail_closed`` actually work: the
audit write happens *before* the response start is forwarded, because once the client has
its status line nothing later can take it back.

Build the ledger per worker, after the fork
-------------------------------------------

A ``Ledger`` owns live handles - a psycopg connection, a SQLite file descriptor. Those do
not survive ``fork()`` being shared: the kernel duplicates the descriptor, so every worker
ends up driving **one** connection. The snippet above builds the ledger at import time,
which is safe only because ``uvicorn --workers`` imports the application separately in
each worker. Run it under ``gunicorn -k uvicorn.workers.UvicornWorker --preload`` and it
is built once in the master instead, before the fork, and every worker inherits it.
Measured on PostgreSQL under ``gunicorn --preload -w 4``: protocol desync (``the last
operation didn't produce records``), 60-second hangs, and a ledger holding more entries
than there were successful responses.

So: **construct the ledger inside an application factory**, which each worker runs for
itself::

    def create_app():
        app = FastAPI()
        app.add_middleware(
            SignLedgerAuditMiddleware, ledger=Ledger(backend=SQLiteBackend("audit.db"))
        )
        return app

    # uvicorn --factory myapp:create_app --workers 4

If the application object has to exist before the fork - ``--preload``, or a module-level
``app`` some tooling insists on - replace the ledger in a post-fork hook, so the inherited
one is never used. ``add_middleware`` hands back no instance to reach, so wrap the app
yourself; this middleware is plain ASGI, which is all ``--preload`` leaves you::

    # myapp.py
    api = FastAPI()
    audit = SignLedgerAuditMiddleware(api, ledger=None)   # keep the handle
    app = audit

    # gunicorn.conf.py
    preload_app = True

    def post_fork(server, worker):
        from myapp import audit
        audit.recorder.ledger = Ledger(backend=SQLiteBackend("audit.db"))

Until that hook runs the ledger is ``None`` and every request fails closed, which is the
correct way round: a worker that cannot audit must not answer ``200``.

The same applies to any pre-fork work: ``uwsgi`` without ``--lazy-apps``, and
``os.fork()`` in a custom supervisor.

:func:`audited` is the one place that rule cannot be obeyed by moving code: a decorator
runs at *import* time, before any factory has been called. So it takes either a ledger or
a **zero-argument callable returning one**, resolved on every call::

    @router.get("/payments/{payment_id}")
    @audited(lambda: app.state.ledger, action="payment.read")
    async def read_payment(payment_id: str) -> dict: ...

The callable is invoked per request, not memoised: a factory that hands out a per-thread
handle - which a SQLite connection must be - would otherwise leak one thread's connection
into another.

What an entry means
-------------------

Every entry carries a ``kind`` and a ``request_id``.

``kind="request"``
    Written by the middleware, one per audited request.

``kind="endpoint"``
    Written by :func:`audited`, one per decorated call. A route covered by both the
    middleware and the decorator therefore produces **two** entries per request, and they
    are not deduplicated - both are true, and each records something the other does not.
    They share a ``request_id``, and the decorator entry repeats ``method`` and ``path``,
    so the pair can be reassembled. A decorated function called outside a request carries
    no request context at all rather than an invented one.

``kind="response.completion"``
    Written when a **streamed** response finishes, and only then.

The inspection endpoints are not part of the trail
--------------------------------------------------

Requests answered by :func:`audit_router` are not audited, so reading the ledger is not a
write to it. Measured over a real uvicorn socket before that, on an application carrying
both halves: ``/_audit/status`` +1 entry, ``/_audit/entries`` +1, an unauthenticated
``401`` +1, a ``422`` from ``Query(ge=1)`` +1, and ``GET /_audit/bogus`` - a 404 needing
no credential at all - +1. ``/verify`` then answered ``entries_verified=5`` while
appending the sixth, so each verification cost more than the last, and the cheapest probe
on the application let an anonymous caller append a signed row to a store that by design
cannot be pruned - against a docstring justifying the mandatory ``authenticator`` as
protection from exactly that.

The decision is taken **after** the request has been routed, as the status is being sent,
because that is the first moment it can be answered correctly. ``route.path`` is
application-relative and ``scope["path"]`` is not: under ``app.mount("/api", subapp)`` or
``FastAPI(root_path="/gw")`` the two differ by exactly the mount, and an exclusion derived
from one up front and matched against the other silently does nothing - which is worse
than no exclusion at all, because the deployer has been told it is automatic and so does
not set ``skip_paths`` either. By response time Starlette has reconciled the two itself:
``root_path`` has travelled with the request, ``get_route_path`` undoes it, and
``scope["endpoint"]`` names whatever answered.

So two questions, in order. *Did one of the router's own endpoints answer?* That settles
the 200s, the ``401`` from the guard, the ``422`` from parameter validation and the ``405``
a matched route answers without ever being entered - under any mount, at any
``root_path``, however the middleware was installed, with no path arithmetic at all. It is
also the only question that survives a single-page-application catch-all, which claims
every path there is. *Failing that, did nothing route this request, at a path under the
mount?* That is the ``/_audit/bogus`` probe, and it is why the exclusion has to be
prefix-shaped rather than a list of the paths the router registered. A prefix match on its
own is **not** taken as evidence: an application is free to serve ``/_audit/dashboard``
itself, so Starlette's own matcher is asked whether it does. Quietly dropping part of the
trail is a worse defect than the one being prevented, and unlike it, invisible from
outside.

Nothing is cached and nothing has to be configured. The route table is read per request -
the Flask sibling reads its mounts from ``current_app`` the same way - so a router
included by a factory after the first request is excluded from the next one, and the
``--preload`` shape above works, where the middleware wraps another middleware and the
thing it holds is not the application and has no routes at all. The application comes from
the scope instead.

An empty ``prefix`` is a :class:`~signledger.core.exceptions.ConfigurationError` at
construction, as it is in Flask: a router at the root cannot be told apart from the site it
reports on, and excluding everything is unrecoverable in a way that auditing a handful of
inspection requests is not.

``completion``: ``"final"`` or ``"provisional"``
-------------------------------------------------

``fail_closed`` requires the audit write to happen before ``http.response.start`` is
forwarded - once the client has the status line, nothing later can take it back. For a
response whose body is already in hand (any response declaring a ``content-length``) that
status is settled, and the entry says ``completion="final"``.

For a streamed body it is not. The status is a *promise*: the endpoint may still raise on
its third chunk, or the client may vanish. Measured against v2.0.0: a route that emitted
one chunk and then raised was permanently in the ledger as ``status: 200`` with no error
field, and so was a mid-stream disconnect.

Entries are append-only and are not rewritten to fix this. Instead a streamed response is
recorded as ``completion="provisional"``, and a second entry - ``kind`` of
``"response.completion"``, same ``request_id`` - records how it actually ended:

* ``outcome="complete"`` - the final body chunk went out;
* ``outcome="incomplete"`` - it did not, plus ``error`` naming the exception if there was
  one. No ``error`` field means nobody raised: the client went away.

So ``status: 200`` on a ``provisional`` entry means *200 was sent*, not *200 was
delivered*. An auditor counting successful responses wants ``completion="final"`` plus
the settled provisional ones. A settlement that cannot be written is counted on
``middleware.recorder.audit_failures`` rather than raised: the response has already gone,
and there is no failing closed after that.
"""

import contextlib
import contextvars
import functools
import inspect
import uuid
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from urllib.parse import parse_qs

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from starlette.concurrency import run_in_threadpool

# ``get_route_path`` is defined in ``starlette._utils`` and re-exported from
# ``starlette.routing``, which ``--strict`` reads as an implicit re-export - hence the
# ignore. Importing the private module instead would bind to something Starlette is free
# to move, and reimplementing the subtraction of ``root_path`` from ``path`` is precisely
# how this exclusion goes quietly wrong under a mount.
from starlette.routing import (  # type: ignore[attr-defined]
    Match,
    get_route_path,
)

from signledger import contrib as _shared
from signledger.contrib import (
    FAIL_CLOSED,
    OUTCOME_COMPLETE,
    OUTCOME_INCOMPLETE,
    AuditRecorder,
    AuditWriteError,
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
    "KIND_ENDPOINT",
    "KIND_REQUEST",
    "SignLedgerAuditMiddleware",
    "audit_router",
    "audited",
]

#: Name recorded in every entry this integration writes.
INTEGRATION = "fastapi"

#: ``kind`` of the entry the middleware writes for one request.
KIND_REQUEST = _shared.KIND_REQUEST

#: ``kind`` of the entry :func:`audited` writes for one decorated call.
KIND_ENDPOINT = "endpoint"

#: ``kind`` of the entry that settles a provisional one.
KIND_COMPLETION = _shared.KIND_COMPLETION

#: The response was complete when its entry was written.
COMPLETION_FINAL = _shared.COMPLETION_FINAL

#: The status was promised before the body existed; a settlement entry follows.
COMPLETION_PROVISIONAL = _shared.COMPLETION_PROVISIONAL

#: Default ceiling on how many entries one inspection request may return.
DEFAULT_MAX_PAGE_SIZE = 100

_AUDIT_FAILED_BODY = b'{"detail":"audit write failed"}'

#: Stamped on every endpoint :func:`audit_router` registers, carrying the path that
#: endpoint answers on *relative to the router*. Taking it off the end of the path the
#: application ended up serving that route at leaves the mount point - whatever ``prefix``
#: the router was built with, and whatever ``include_router`` then did to it. The
#: endpoints are closures built per call, so two routers, or one router included twice,
#: stamp independently.
_AUDIT_ROUTE_PATH = "__signledger_audit_path__"

#: The request the current task is serving, for :func:`audited` to correlate against.
#: Set by the middleware; ``None`` in a decorated call that no request reached.
_REQUEST: "contextvars.ContextVar[Optional[Dict[str, Any]]]" = contextvars.ContextVar(
    "signledger_fastapi_request", default=None
)


class SignLedgerAuditMiddleware:
    """Pure ASGI middleware writing one ledger entry per HTTP request.

    :param app: the next ASGI application. Supplied by ``add_middleware``.
    :param ledger: a :class:`~signledger.core.ledger.Ledger`.
    :param redaction: what may be recorded. Defaults to the strict policy.
    :param audit_failure_policy: ``"fail_closed"`` (default) or ``"fail_open"``.
    :param skip_paths: path prefixes that are not audited.
    :param user_resolver: called with a Starlette ``Request``, returns the identity.
    """

    def __init__(
        self,
        app: Any,
        *,
        ledger: Any = None,
        redaction: Optional[RedactionPolicy] = None,
        audit_failure_policy: str = FAIL_CLOSED,
        skip_paths: Iterable[str] = (),
        user_resolver: Optional[Callable[[Any], Any]] = None,
    ) -> None:
        self.app = app
        self.recorder = AuditRecorder(
            ledger, redaction=redaction, audit_failure_policy=audit_failure_policy
        )
        self.skip_paths: Tuple[str, ...] = tuple(skip_paths)
        self.user_resolver = user_resolver

    async def __call__(self, scope: Dict[str, Any], receive: Any, send: Any) -> None:
        path = scope.get("path", "")
        if scope["type"] != "http" or self._skip(path):
            # Websockets and lifespan messages reach the application untouched.
            await self.app(scope, receive, send)
            return

        policy = self.recorder.redaction
        headers = _scope_headers(scope)
        decision = policy.body_decision(
            content_length=_content_length(headers),
            content_type=headers.get("content-type"),
        )

        chunks: List[bytes] = []
        state: Dict[str, Any] = {
            "recorded": False,
            "refused": False,
            "buffered": 0,
            "overflow": False,
            # Set once the status is promised for a body that does not exist yet.
            "provisional": False,
            "status": None,
            "finished": False,
        }
        context: Dict[str, Any] = {
            "request_id": uuid.uuid4().hex,
            "method": scope.get("method", ""),
            "path": path,
        }

        async def observing_receive() -> Dict[str, Any]:
            """Watch the body go past on its way to the endpoint.

            Never drains it: the message is returned to the caller exactly as received,
            so the endpoint still gets its own request body.

            ``Content-Length`` was already checked, but it is a claim the client makes,
            not a fact. If the real stream runs past the limit we stop buffering and
            throw away what we have: a truncated JSON fragment in an audit log is worse
            than an honest "too-large", and a middleware whose memory use is chosen by
            the caller is a denial of service.
            """
            message: Dict[str, Any] = await receive()
            if decision.capture and not state["overflow"] and message["type"] == "http.request":
                chunk = message.get("body", b"")
                if state["buffered"] + len(chunk) > policy.max_body_bytes:
                    state["overflow"] = True
                    chunks.clear()
                else:
                    chunks.append(chunk)
                    state["buffered"] += len(chunk)
            return message

        async def auditing_send(message: Dict[str, Any]) -> None:
            if message["type"] == "http.response.start" and not state["recorded"]:
                # "Recorded" in the sense of "decided about": an inspection request has
                # been dealt with by writing nothing, and the exception path below must
                # not reconsider it.
                state["recorded"] = True
                if not _is_inspection(scope):
                    state["status"] = message["status"]
                    state["provisional"] = _is_streamed(scope.get("method", ""), message)
                    try:
                        await self._record(
                            scope, headers, decision, chunks, message["status"], None,
                            overflow=state["overflow"], context=context,
                            provisional=state["provisional"],
                        )
                    except AuditWriteError:
                        # fail_closed, and the client has not been told anything yet.
                        # This is the only moment at which refusing is still possible.
                        # Nothing was promised, so there is nothing to settle either.
                        state["refused"] = True
                        state["provisional"] = False
                        await _send_audit_failure(send)
                        return
            if state["refused"]:
                # The endpoint's own body must not follow our 500 down the wire.
                return
            await send(message)
            if message["type"] == "http.response.body" and not message.get("more_body", False):
                # Sent, not merely produced: the last chunk is on the wire.
                state["finished"] = True

        token = _REQUEST.set(context)
        try:
            await self.app(scope, observing_receive, auditing_send)
        except Exception as err:
            if not state["recorded"]:
                state["recorded"] = True
                # The failed requests are the ones an audit trail exists for - unless
                # this was an inspection request, which is no more part of the trail for
                # having blown up than it would have been for succeeding.
                if not _is_inspection(scope):
                    await self._record(
                        scope, headers, decision, chunks, 500, type(err).__name__,
                        overflow=state["overflow"], context=context, provisional=False,
                    )
            else:
                # The status went out before this happened and cannot be withdrawn. It
                # can be settled, which is the whole point of the second entry.
                await self._settle(context, state, type(err).__name__)
            raise
        else:
            await self._settle(context, state, None)
        finally:
            _REQUEST.reset(token)

    # -- internals ---------------------------------------------------------------------

    def _skip(self, path: str) -> bool:
        return is_skipped(path, self.skip_paths)

    async def _record(
        self,
        scope: Dict[str, Any],
        headers: Dict[str, str],
        decision: Any,
        chunks: List[bytes],
        status: int,
        error: Optional[str],
        *,
        context: Dict[str, Any],
        provisional: bool,
        overflow: bool = False,
    ) -> None:
        policy = self.recorder.redaction
        body, reason = self._body(decision, chunks, overflow)
        # Kept for the settlement entry, which is written after the request scope is gone
        # and used to hand-assemble a record carrying none of this. A settlement that
        # cannot say who made the request is missing the field an auditor reaches for
        # first, on exactly the half of the request that failed.
        context["shape"] = {
            "client_ip": policy.client_ip(
                peer=_peer(scope), forwarded_for=headers.get("x-forwarded-for")
            ),
            "headers": policy.headers(headers),
            "query": policy.query(_query(scope)),
            "body": body,
            "body_reason": reason,
            "user": self._user(scope),
        }
        record = build_audit_record(
            integration=INTEGRATION,
            method=scope.get("method", ""),
            path=scope.get("path", ""),
            status=status,
            error=error,
            **context["shape"],
            kind=KIND_REQUEST,
            request_id=context["request_id"],
            # Says whether the status above is a fact or a promise. Never absent: an
            # auditor must not have to know which version of this middleware - or which
            # framework - wrote the entry, which is why the builder owns the default.
            completion=COMPLETION_PROVISIONAL if provisional else COMPLETION_FINAL,
        )
        # The append is synchronous and may fsync. Off the event loop it goes.
        await run_in_threadpool(self.recorder.record, record, {"integration": INTEGRATION})

    async def _settle(
        self, context: Dict[str, Any], state: Dict[str, Any], error: Optional[str]
    ) -> None:
        """Record how a promised response actually ended.

        Only for provisional entries: settling a response that was already complete when
        it was recorded would double the size of every ledger and say nothing.
        """
        if not state["provisional"]:
            return
        record = build_audit_record(
            integration=INTEGRATION,
            method=context["method"],
            path=context["path"],
            status=state["status"],
            error=error,
            kind=KIND_COMPLETION,
            request_id=context["request_id"],
            # The settlement's own status is a fact by construction: it is written once
            # the body has stopped moving.
            completion=COMPLETION_FINAL,
            outcome=OUTCOME_COMPLETE if state["finished"] else OUTCOME_INCOMPLETE,
            **context["shape"],
        )
        # There is no failing closed here: the client has had the response since the
        # first chunk. The recorder has already counted the failure and named it, so this
        # is not a swallowed error - it is one that has nowhere left to go.
        with contextlib.suppress(AuditWriteError):
            await run_in_threadpool(
                self.recorder.record, record, {"integration": INTEGRATION}
            )

    def _body(self, decision: Any, chunks: List[bytes], overflow: bool) -> Tuple[Any, str]:
        if not decision.capture:
            return None, decision.reason
        if overflow:
            # The same answer the pre-read check would have given had the client been
            # honest about the length.
            return None, "too-large"
        if not chunks:
            # Capture was allowed but the endpoint never read the stream, so there is
            # nothing to report. Saying so beats inventing an empty body.
            return None, "not-read"
        return self.recorder.redaction.decode_body(b"".join(chunks)), "captured"

    def _user(self, scope: Dict[str, Any]) -> Optional[str]:
        if self.user_resolver is None:
            return None
        resolved = self.user_resolver(Request(scope))
        return None if resolved is None else str(resolved)


def audited(
    ledger: Any,
    *,
    action: Optional[str] = None,
    redaction: Optional[RedactionPolicy] = None,
    audit_failure_policy: str = FAIL_CLOSED,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorate one endpoint so that calling it writes a ledger entry.

    ``ledger`` is either a :class:`~signledger.core.ledger.Ledger` or a **zero-argument
    callable returning one**, called on each audited call. The callable form is what
    makes this decorator usable at all from an application factory: decorators run at
    import time, and the ledger must not be constructed until after the fork. Binding an
    instance here and building it in the factory are mutually exclusive instructions.

    An **async** endpoint gets an async wrapper, so the decorated object is still a
    coroutine function and FastAPI still awaits it. v1's version returned ``None``, which
    registered the route against nothing and destroyed the endpoint outright.

    A **sync** endpoint keeps a sync wrapper. Promoting it to a coroutine would move a
    blocking call onto the event loop - FastAPI already runs sync endpoints in a
    threadpool, and this must not undo that.

    ``functools.wraps`` is what keeps FastAPI's dependency injection working: it sets
    ``__wrapped__``, so ``inspect.signature`` still sees the endpoint's real parameters.
    """
    policy = RedactionPolicy() if redaction is None else redaction
    # Built and thrown away purely to validate ``audit_failure_policy`` now. Deferring
    # that to the first request means a typo is found by a user, in production.
    AuditRecorder(None, redaction=policy, audit_failure_policy=audit_failure_policy)

    def resolve() -> AuditRecorder:
        """The recorder for one call, against the ledger that exists at this moment.

        Deliberately not memoised. A factory may legitimately hand out a per-thread
        handle - a SQLite connection must be one - and caching the first would use it
        from every thread afterwards.
        """
        resolved = ledger() if callable(ledger) else ledger
        return AuditRecorder(
            resolved, redaction=policy, audit_failure_policy=audit_failure_policy
        )

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        label = func.__name__ if action is None else action

        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                try:
                    result = await func(*args, **kwargs)
                except Exception as err:
                    await run_in_threadpool(
                        resolve().record, _call_record(label, func, type(err).__name__)
                    )
                    raise
                await run_in_threadpool(resolve().record, _call_record(label, func, None))
                return result

            # Returned, not discarded. This one line is the whole v1 defect.
            return async_wrapper

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                result = func(*args, **kwargs)
            except Exception as err:
                resolve().record(_call_record(label, func, type(err).__name__))
                raise
            resolve().record(_call_record(label, func, None))
            return result

        return sync_wrapper

    return decorator


def _call_record(label: str, func: Callable[..., Any], error: Optional[str]) -> Dict[str, Any]:
    """The audit payload for a decorated call.

    Deliberately records *no* arguments. A decorator sees the endpoint's real parameters,
    which is exactly where the passwords are, and there is no request context here to
    redact against.

    It does carry the identity of the request it happened inside, when there is one. A
    route covered by both the middleware and this decorator writes two entries, and
    without a shared ``request_id`` they had exactly one field in common - ``integration``
    - so no auditor could tell they described the same request. Outside a request the
    fields are absent rather than null: there is nothing to correlate with, and a field
    that is always there and always empty teaches people to stop reading it.
    """
    record: Dict[str, Any] = {
        "integration": INTEGRATION,
        "kind": KIND_ENDPOINT,
        "action": label,
        "endpoint": func.__name__,
    }
    context = _REQUEST.get()
    if context is not None:
        record["request_id"] = context["request_id"]
        record["method"] = context["method"]
        record["path"] = context["path"]
    if error is not None:
        record["error"] = error
    return record


def _is_inspection(scope: Dict[str, Any]) -> bool:
    """Whether this request belongs to an :func:`audit_router` on this application.

    Asked *after* the request has been routed - the scope is the same dict the router has
    been writing into, so by the time a status is on its way out it says which endpoint
    answered, and ``root_path`` has been advanced by every mount the request passed
    through. That timing is the whole design. ``route.path`` is application-relative and
    ``scope["path"]`` is not; under ``app.mount("/api", subapp)`` or
    ``FastAPI(root_path="/gw")`` the two differ by exactly the mount, and an exclusion
    derived from one up front and compared against the other silently does nothing at
    all. Deciding late means Starlette has already reconciled the two itself, and
    :func:`get_route_path` - Starlette's own comparator, not a reimplementation of the
    subtraction - undoes the rest.

    Two questions, in this order.

    *Did one of the router's own endpoints answer?* ``scope["endpoint"]`` says so, with no
    path arithmetic at all, which settles the 200s, the ``401`` the guard raises, the
    ``422`` from ``Query(ge=1)`` and the ``405`` a matched route answers without ever
    being entered. It is also the only question that survives a single-page-application
    catch-all: with ``@app.get("/{path:path}")`` registered, "does the application have a
    route here?" is always yes, and asking that alone would audit every inspection request
    again.

    *Failing that, did nothing route this request, at a path under the mount?* That is
    ``GET /_audit/bogus``: no credential, no route, no endpoint, and so the cheapest probe
    on the application - which is why the exclusion has to be prefix-shaped rather than a
    list of the paths the router registered. A prefix match is not on its own taken as
    evidence, because an application is free to serve ``/_audit/dashboard`` itself and
    quietly dropping part of the trail is a worse defect than the one being prevented.
    Starlette's own matcher is asked whether the application claims the path first.

    That second question is not merely the 404 case dressed up. Starlette names a
    *mounted application* as the endpoint of a request its own router then failed to
    route, so under ``app.mount("/api", subapp)`` an unrouted probe arrives here with an
    endpoint set that is nobody's route - "something answered" and "one of ours answered"
    are different questions and only the second one is asked above.
    """
    if _audit_relative_path(scope.get("endpoint")) is not None:
        return True
    app = scope.get("app")
    mounts = _inspection_mounts(app)
    if not mounts or not is_skipped(get_route_path(scope), mounts):
        return False
    return not _claimed_by_application(app, scope)


def _inspection_mounts(app: Any) -> List[str]:
    """Paths on ``app`` that :func:`audit_router` answers on.

    Read from the live route table on every request rather than worked out once and kept.
    A cached answer is wrong for exactly as long as the process runs whenever a factory
    includes the router after the first request has been served, and the Flask sibling
    reads its own mounts from ``current_app`` per request for the same reason. The cost
    is a pass of ``getattr`` over the route table, against an audit write that may fsync
    a disk.

    ``app`` comes from the scope rather than from ``self.app``, because the two are not
    the same object in the ``--preload`` shape this module documents: there the
    middleware holds another middleware, which has no routes and is not the application.
    """
    mounts: List[str] = []
    for route in getattr(app, "routes", ()):
        relative = _audit_relative_path(getattr(route, "endpoint", None))
        path: str = getattr(route, "path", "")
        if relative is None or not path.endswith(relative):
            continue
        mounts.append(path[: len(path) - len(relative)])
    return mounts


def _claimed_by_application(app: Any, scope: Dict[str, Any]) -> bool:
    """Whether one of ``app``'s own routes - not the audit router's - answers this.

    Starlette's own matcher, so a path that merely shares the mount's prefix is treated
    as a probe only when the application really has nothing there. ``Match.PARTIAL``
    counts: a ``405`` from an application route is the application answering, and belongs
    in the trail.
    """
    for route in getattr(app, "routes", ()):
        if _audit_relative_path(getattr(route, "endpoint", None)) is not None:
            continue
        if route.matches(scope)[0] != Match.NONE:
            return True
    return False


def _audit_relative_path(endpoint: Any) -> Optional[str]:
    """The router-relative path stamped on ``endpoint``, if it is an audit one."""
    relative = getattr(endpoint, _AUDIT_ROUTE_PATH, None)
    return relative if isinstance(relative, str) else None


def _is_streamed(method: str, message: Dict[str, Any]) -> bool:
    """Whether a body is still to come that the status has already been promised for.

    A response declaring a ``content-length`` was fully built before its status went out;
    one that does not is being generated as it is sent, and may yet fail. Responses that
    will not send a body at all are not waiting on anything - and that test now comes from
    the shared helper, because testing the status alone left ``HEAD`` on a streaming route
    provisional here and final on the other two integrations, waiting for a settlement no
    server would ever trigger.
    """
    if not will_send_body(method, message["status"]):
        return False
    return not any(name.lower() == b"content-length" for name, _ in message.get("headers", []))


def audit_router(
    ledger: Any,
    *,
    authenticator: Optional[Callable[[Any], bool]],
    prefix: str = "/_audit",
    max_page_size: int = DEFAULT_MAX_PAGE_SIZE,
) -> APIRouter:
    """A router exposing the audit trail over HTTP, authenticated by default.

    ``authenticator`` has no default value. An inspection endpoint that is open because
    somebody forgot an argument is both a denial-of-service surface - every request can
    walk the entire ledger - and an information-disclosure one.

    Even authenticated, ``limit`` is capped at ``max_page_size`` and rows carry
    identifiers and digests only, never entry payloads.

    Including this router tells :class:`SignLedgerAuditMiddleware` on the same application
    not to audit the requests it answers, so that reading the trail is not a write to it.
    That is not something a deployer has to remember, and not something they could get
    right by hand either: ``skip_paths=("/_audit",)`` is wrong the moment ``prefix`` or
    ``include_router(..., prefix=...)`` moves the endpoints, and wrong again under a mount
    or a ``root_path``, where the path the client asked for is not the path the router was
    registered at.

    ``prefix`` may not be empty, and that is a
    :class:`~signledger.core.exceptions.ConfigurationError` here rather than a surprise
    later. Excluding the mount is what stops a read being a write, and there is no prefix
    meaning "the root" that does not also mean every other path on the application - a
    router at the root could not be told apart from the site it reports on. The Flask
    sibling refuses the same thing for the same reason.
    """
    if authenticator is None:
        raise ConfigurationError(
            "audit_router requires an authenticator; an unauthenticated audit endpoint "
            "exposes the whole ledger to anyone who can reach it"
        )
    if not prefix.strip("/"):
        raise ConfigurationError(
            "audit_router must be mounted under a non-empty prefix so that its own "
            "requests can be kept out of the trail it serves"
        )
    if max_page_size < 1:
        raise ConfigurationError(f"max_page_size must be at least 1, got {max_page_size}")

    def guard(request: Request) -> None:
        if not authenticator(request):
            raise HTTPException(status_code=401, detail="authentication required")

    router = APIRouter(prefix=prefix, dependencies=[Depends(guard)])

    @router.get("/status")
    def status() -> Dict[str, Any]:
        """Cheap counters. Deliberately does not verify - see ``/verify``."""
        snapshot: Dict[str, Any] = ledger.get_stats_snapshot()
        snapshot.update(
            {
                "entry_count": ledger.count(),
                "capabilities": sorted(ledger.backend.capabilities()),
            }
        )
        return snapshot

    @router.get("/verify")
    def verify() -> Dict[str, Any]:
        """Run a real verification. O(n) in the ledger, hence its own route.

        ``assurance`` travels with ``ok`` on purpose: a chain-only deployment answering
        ``ok`` has detected accidental corruption and naive edits, and nothing else.
        """
        report = ledger.check_integrity()
        return {
            "ok": report.ok,
            "entries_verified": report.entries_verified,
            "assurance": report.assurance.value,
            "failures": [failure.message for failure in report.failures],
        }

    @router.get("/entries")
    def entries(
        limit: int = Query(default=max_page_size, ge=1),
        offset: int = Query(default=0, ge=0),
    ) -> Dict[str, Any]:
        rows = ledger.iter_entries(limit=min(limit, max_page_size), offset=offset)
        return {"entries": [_summarise(entry) for entry in rows]}

    # Claim every route on the way out, rather than each endpoint at its definition: a
    # route added to this router later and left unstamped would audit itself again, and
    # nothing would say so. The stamp is the path *within* ``prefix``, because that is
    # what stays true when ``include_router`` moves the whole router somewhere else.
    for route in router.routes:
        endpoint = getattr(route, "endpoint", None)
        registered: str = getattr(route, "path", "")
        if endpoint is not None:
            setattr(endpoint, _AUDIT_ROUTE_PATH, registered[len(prefix) :])

    return router


def _summarise(entry: Any) -> Dict[str, Any]:
    """Identifiers and digests only - enough to audit, not enough to be a breach."""
    return {
        "sequence": entry.sequence,
        "entry_id": entry.entry_id,
        "timestamp": entry.timestamp.isoformat(),
        "hash": entry.hash,
        "previous_hash": entry.previous_hash,
        "key_id": entry.key_id,
    }


async def _send_audit_failure(send: Any) -> None:
    """Answer 500 in place of a response we were unable to audit."""
    await send(
        {
            "type": "http.response.start",
            "status": 500,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(_AUDIT_FAILED_BODY)).encode("ascii")),
            ],
        }
    )
    await send({"type": "http.response.body", "body": _AUDIT_FAILED_BODY})


def _scope_headers(scope: Dict[str, Any]) -> Dict[str, str]:
    """ASGI headers as a plain lower-cased mapping."""
    return {
        name.decode("latin-1").lower(): value.decode("latin-1")
        for name, value in scope.get("headers", [])
    }


def _content_length(headers: Dict[str, str]) -> Optional[int]:
    raw = headers.get("content-length")
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        # An unparseable length is no length at all, and body capture will refuse it.
        return None


def _query(scope: Dict[str, Any]) -> Dict[str, List[str]]:
    """Query parameters, keeping repeated names rather than silently dropping them."""
    return parse_qs(scope.get("query_string", b"").decode("latin-1"))


def _peer(scope: Dict[str, Any]) -> Optional[str]:
    client = scope.get("client")
    return None if client is None else str(client[0])

