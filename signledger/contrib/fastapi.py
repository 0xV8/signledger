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
"""

import functools
import inspect
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from urllib.parse import parse_qs

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from starlette.concurrency import run_in_threadpool

from signledger.contrib import (
    FAIL_CLOSED,
    AuditRecorder,
    AuditWriteError,
    build_audit_record,
)
from signledger.contrib.redaction import RedactionPolicy
from signledger.core.exceptions import ConfigurationError

__all__ = ["INTEGRATION", "SignLedgerAuditMiddleware", "audit_router", "audited"]

#: Name recorded in every entry this integration writes.
INTEGRATION = "fastapi"

#: Default ceiling on how many entries one inspection request may return.
DEFAULT_MAX_PAGE_SIZE = 100

_AUDIT_FAILED_BODY = b'{"detail":"audit write failed"}'


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
                state["recorded"] = True
                try:
                    await self._record(
                        scope, headers, decision, chunks, message["status"], None,
                        overflow=state["overflow"],
                    )
                except AuditWriteError:
                    # fail_closed, and the client has not been told anything yet. This is
                    # the only moment at which refusing is still possible.
                    state["refused"] = True
                    await _send_audit_failure(send)
                    return
            if state["refused"]:
                # The endpoint's own body must not follow our 500 down the wire.
                return
            await send(message)

        try:
            await self.app(scope, observing_receive, auditing_send)
        except Exception as err:
            if not state["recorded"]:
                state["recorded"] = True
                # The failed requests are the ones an audit trail exists for.
                await self._record(
                    scope, headers, decision, chunks, 500, type(err).__name__,
                    overflow=state["overflow"],
                )
            raise

    # -- internals ---------------------------------------------------------------------

    def _skip(self, path: str) -> bool:
        return any(path.startswith(prefix) for prefix in self.skip_paths)

    async def _record(
        self,
        scope: Dict[str, Any],
        headers: Dict[str, str],
        decision: Any,
        chunks: List[bytes],
        status: int,
        error: Optional[str],
        *,
        overflow: bool = False,
    ) -> None:
        policy = self.recorder.redaction
        body, reason = self._body(decision, chunks, overflow)
        record = build_audit_record(
            integration=INTEGRATION,
            method=scope.get("method", ""),
            path=scope.get("path", ""),
            status=status,
            client_ip=policy.client_ip(
                peer=_peer(scope), forwarded_for=headers.get("x-forwarded-for")
            ),
            headers=policy.headers(headers),
            query=policy.query(_query(scope)),
            body=body,
            body_reason=reason,
            user=self._user(scope),
            error=error,
        )
        # The append is synchronous and may fsync. Off the event loop it goes.
        await run_in_threadpool(self.recorder.record, record, {"integration": INTEGRATION})

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

    An **async** endpoint gets an async wrapper, so the decorated object is still a
    coroutine function and FastAPI still awaits it. v1's version returned ``None``, which
    registered the route against nothing and destroyed the endpoint outright.

    A **sync** endpoint keeps a sync wrapper. Promoting it to a coroutine would move a
    blocking call onto the event loop - FastAPI already runs sync endpoints in a
    threadpool, and this must not undo that.

    ``functools.wraps`` is what keeps FastAPI's dependency injection working: it sets
    ``__wrapped__``, so ``inspect.signature`` still sees the endpoint's real parameters.
    """
    recorder = AuditRecorder(
        ledger, redaction=redaction, audit_failure_policy=audit_failure_policy
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
                        recorder.record, _call_record(label, func, type(err).__name__)
                    )
                    raise
                await run_in_threadpool(recorder.record, _call_record(label, func, None))
                return result

            # Returned, not discarded. This one line is the whole v1 defect.
            return async_wrapper

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                result = func(*args, **kwargs)
            except Exception as err:
                recorder.record(_call_record(label, func, type(err).__name__))
                raise
            recorder.record(_call_record(label, func, None))
            return result

        return sync_wrapper

    return decorator


def _call_record(label: str, func: Callable[..., Any], error: Optional[str]) -> Dict[str, Any]:
    """The audit payload for a decorated call.

    Deliberately records *no* arguments. A decorator sees the endpoint's real parameters,
    which is exactly where the passwords are, and there is no request context here to
    redact against.
    """
    record: Dict[str, Any] = {
        "integration": INTEGRATION,
        "kind": "endpoint",
        "action": label,
        "endpoint": func.__name__,
    }
    if error is not None:
        record["error"] = error
    return record


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
    """
    if authenticator is None:
        raise ConfigurationError(
            "audit_router requires an authenticator; an unauthenticated audit endpoint "
            "exposes the whole ledger to anyone who can reach it"
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

