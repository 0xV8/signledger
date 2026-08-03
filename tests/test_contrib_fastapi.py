"""Tests for :mod:`signledger.contrib.fastapi`, driven through a real ASGI stack.

Three v1 defects are pinned here:

* the decorator returned ``None``, so decorating an endpoint *deleted* it;
* the middleware wrote ``Authorization`` headers, cookies and whole request bodies into
  an immutable store;
* the ledger append is synchronous and was called straight from the event loop, so every
  audited request blocked every other request in the process.

Nothing here mocks FastAPI, Starlette or the ledger.
"""

import inspect
import json
import subprocess
import sys
import threading
from typing import Any, AsyncIterator, Dict, Iterator, List

import pytest
from fastapi import FastAPI, WebSocket
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import Response

from signledger.backends.memory import MemoryBackend
from signledger.contrib import FAIL_OPEN
from signledger.contrib.fastapi import SignLedgerAuditMiddleware, audit_router, audited
from signledger.contrib.redaction import REDACTED, RedactionPolicy
from signledger.core.exceptions import ConfigurationError
from signledger.core.ledger import Ledger

REPO_ROOT = str(__import__("pathlib").Path(__file__).resolve().parent.parent)


class ThreadRecordingBackend(MemoryBackend):
    """A real backend that also notes which thread each append ran on.

    Not a mock: every method still executes the real ``MemoryBackend`` implementation.
    It adds one observation, which is the only way to assert *where* the synchronous
    append happened.
    """

    def __init__(self) -> None:
        super().__init__()
        self.append_threads: List[int] = []

    def append_atomic(self, build: Any) -> Any:
        self.append_threads.append(threading.get_ident())
        return super().append_atomic(build)


@pytest.fixture
def ledger() -> Iterator[Ledger]:
    with Ledger(backend=MemoryBackend(), ledger_id="fastapi-test") as led:
        yield led


def make_app(ledger: Ledger, **kwargs: Any) -> FastAPI:
    app = FastAPI()
    app.add_middleware(SignLedgerAuditMiddleware, ledger=ledger, **kwargs)

    @app.post("/login")
    async def login(payload: Dict[str, Any]) -> Dict[str, Any]:
        return {"ok": True}

    @app.get("/items")
    async def items() -> Dict[str, Any]:
        return {"items": []}

    @app.get("/boom")
    async def boom() -> Dict[str, Any]:
        raise RuntimeError("endpoint exploded")

    @app.get("/health")
    async def health() -> Dict[str, Any]:
        return {"ok": True}

    return app


def entries(ledger: Ledger) -> List[Any]:
    return list(ledger.iter_entries())


def kinds(ledger: Ledger) -> List[Any]:
    return [entry.data.get("kind") for entry in entries(ledger)]


def streaming_app(ledger: Ledger, **kwargs: Any) -> FastAPI:
    """An application whose response body is produced *after* the status is sent."""
    app = FastAPI()
    app.add_middleware(SignLedgerAuditMiddleware, ledger=ledger, **kwargs)

    @app.get("/stream")
    async def stream() -> StreamingResponse:
        async def body() -> AsyncIterator[bytes]:
            yield b'{"row": 0}\n'
            yield b'{"row": 1}\n'

        return StreamingResponse(body(), media_type="application/x-ndjson")

    @app.get("/broken-stream")
    async def broken_stream() -> StreamingResponse:
        async def body() -> AsyncIterator[bytes]:
            yield b'{"row": 0}\n'
            raise RuntimeError("settlement feed died mid-stream")

        return StreamingResponse(body(), media_type="application/x-ndjson")

    @app.get("/nothing")
    async def nothing() -> Response:
        return Response(status_code=204)

    return app


# ======================================================================================
# The headline requirement: credentials must not reach the ledger
# ======================================================================================


def test_an_authorization_header_and_a_password_body_leak_neither_into_the_ledger(
    ledger: Ledger,
) -> None:
    """The exact v1 defect, asserted against the whole serialized entry."""
    app = make_app(ledger, redaction=RedactionPolicy(capture_body=True))
    with TestClient(app) as client:
        client.post(
            "/login",
            headers={"Authorization": "Bearer sk-live-51H8xQsecret"},
            json={"user": "ada", "password": "hunter2"},
        )

    stored = json.dumps(entries(ledger)[0].model_dump(), default=str)
    assert "sk-live-51H8xQsecret" not in stored
    assert "hunter2" not in stored


def test_the_authorization_header_is_recorded_as_the_marker(ledger: Ledger) -> None:
    with TestClient(make_app(ledger)) as client:
        client.get("/items", headers={"Authorization": "Bearer secret-value"})
    assert entries(ledger)[0].data["headers"]["authorization"] == REDACTED


def test_a_cookie_is_never_recorded(ledger: Ledger) -> None:
    """Cookies are set on the client, not per request: httpx deprecated the latter."""
    with TestClient(make_app(ledger), cookies={"session": "abc123"}) as client:
        client.get("/items")
    assert entries(ledger)[0].data["headers"]["cookie"] == REDACTED


def test_a_query_parameter_carrying_an_api_key_is_redacted(ledger: Ledger) -> None:
    with TestClient(make_app(ledger)) as client:
        client.get("/items?page=2&api_key=sk-live-1")
    assert entries(ledger)[0].data["query"]["api_key"] == [REDACTED]


# ======================================================================================
# The decorator must not destroy the endpoint it decorates
# ======================================================================================


def test_the_decorator_returns_a_coroutine_function_for_an_async_endpoint(
    ledger: Ledger,
) -> None:
    """v1 returned ``None`` here, so decorating an endpoint deleted it."""

    async def endpoint() -> Dict[str, Any]:
        return {"ok": True}

    decorated = audited(ledger)(endpoint)
    assert inspect.iscoroutinefunction(decorated)


def test_the_decorator_never_returns_none(ledger: Ledger) -> None:
    async def endpoint() -> Dict[str, Any]:
        return {"ok": True}

    assert audited(ledger)(endpoint) is not None


def test_a_decorated_async_endpoint_still_returns_its_own_value(
    ledger: Ledger,
) -> None:
    app = FastAPI()

    @app.get("/x")
    @audited(ledger, action="read-x")
    async def endpoint() -> Dict[str, Any]:
        return {"value": 7}

    with TestClient(app) as client:
        assert client.get("/x").json() == {"value": 7}


def test_a_decorated_async_endpoint_writes_an_entry(ledger: Ledger) -> None:
    app = FastAPI()

    @app.get("/x")
    @audited(ledger, action="read-x")
    async def endpoint() -> Dict[str, Any]:
        return {"value": 7}

    with TestClient(app) as client:
        client.get("/x")
    assert entries(ledger)[0].data["action"] == "read-x"


def test_a_decorated_endpoint_keeps_its_path_parameters(ledger: Ledger) -> None:
    """FastAPI reads the signature to build dependencies; a wrapper that hides it
    breaks every injected argument."""
    app = FastAPI()

    @app.get("/item/{item_id}")
    @audited(ledger)
    async def endpoint(item_id: int) -> Dict[str, Any]:
        return {"item_id": item_id}

    with TestClient(app) as client:
        assert client.get("/item/42").json() == {"item_id": 42}


def test_the_decorator_also_works_on_a_synchronous_endpoint(ledger: Ledger) -> None:
    app = FastAPI()

    @app.get("/sync")
    @audited(ledger)
    def endpoint() -> Dict[str, Any]:
        return {"value": 9}

    with TestClient(app) as client:
        assert client.get("/sync").json() == {"value": 9}


def test_a_decorated_sync_endpoint_is_not_turned_into_a_coroutine(
    ledger: Ledger,
) -> None:
    """Converting it would move the call onto the event loop and block it."""

    def endpoint() -> Dict[str, Any]:
        return {"value": 9}

    assert not inspect.iscoroutinefunction(audited(ledger)(endpoint))


def test_a_decorated_sync_endpoint_writes_an_entry(ledger: Ledger) -> None:
    def endpoint() -> Dict[str, Any]:
        return {"value": 9}

    audited(ledger, action="sync-call")(endpoint)()
    assert entries(ledger)[0].data["action"] == "sync-call"


def test_the_decorator_keeps_the_endpoint_name(ledger: Ledger) -> None:
    async def my_endpoint() -> Dict[str, Any]:
        return {}

    assert audited(ledger)(my_endpoint).__name__ == "my_endpoint"


def test_a_decorated_endpoint_that_raises_still_writes_an_entry(
    ledger: Ledger,
) -> None:
    async def endpoint() -> Dict[str, Any]:
        raise RuntimeError("no")

    import anyio

    with pytest.raises(RuntimeError):
        anyio.run(audited(ledger)(endpoint))
    assert entries(ledger)[0].data["error"] == "RuntimeError"


# ======================================================================================
# The ledger may be supplied as a factory, resolved at call time
#
# The module docstring tells users to construct the ``Ledger`` inside an application
# factory, so that nothing holding a socket exists before a pre-forking server forks.
# Decorators run at *import* time, before any factory has run. A decorator that can only
# capture a live instance cannot be used from such an application at all.
# ======================================================================================


def test_the_decorator_accepts_a_ledger_that_does_not_exist_yet(ledger: Ledger) -> None:
    """The ledger is built after the module is imported, as an app factory would."""
    holder: Dict[str, Any] = {}
    app = FastAPI()

    @app.get("/x")
    @audited(lambda: holder["ledger"], action="read-x")
    async def endpoint() -> Dict[str, Any]:
        return {"value": 7}

    holder["ledger"] = ledger
    with TestClient(app) as client:
        assert client.get("/x").status_code == 200


def test_a_factory_supplied_ledger_receives_the_entry(ledger: Ledger) -> None:
    app = FastAPI()

    @app.get("/x")
    @audited(lambda: ledger, action="read-x")
    async def endpoint() -> Dict[str, Any]:
        return {"value": 7}

    with TestClient(app) as client:
        client.get("/x")
    assert entries(ledger)[0].data["action"] == "read-x"


def test_the_ledger_factory_runs_per_call_and_not_at_decoration_time(
    ledger: Ledger,
) -> None:
    """Resolving it eagerly would put the connection back on the wrong side of the fork.

    Resolving it once and caching it would be just as wrong for a factory that hands out
    a per-thread handle - a SQLite connection may not cross threads at all.
    """
    calls: List[int] = []

    def factory() -> Ledger:
        calls.append(1)
        return ledger

    app = FastAPI()

    @app.get("/x")
    @audited(factory)
    async def endpoint() -> Dict[str, Any]:
        return {"ok": True}

    at_decoration = len(calls)
    with TestClient(app) as client:
        client.get("/x")
        client.get("/x")
    assert (at_decoration, len(calls)) == (0, 2)


def test_the_factory_form_works_on_a_synchronous_endpoint(ledger: Ledger) -> None:
    def endpoint() -> Dict[str, Any]:
        return {"value": 9}

    audited(lambda: ledger, action="sync-call")(endpoint)()
    assert entries(ledger)[0].data["action"] == "sync-call"


def test_the_factory_form_still_records_an_endpoint_that_raised(ledger: Ledger) -> None:
    async def endpoint() -> Dict[str, Any]:
        raise RuntimeError("no")

    import anyio

    with pytest.raises(RuntimeError):
        anyio.run(audited(lambda: ledger)(endpoint))
    assert entries(ledger)[0].data["error"] == "RuntimeError"


def test_a_synchronous_endpoint_that_raises_is_audited_by_the_server_that_ran_it(
    ledger: Ledger,
) -> None:
    """Driven through a real application, not by calling the wrapper.

    FastAPI runs a ``def`` endpoint in a threadpool and an ``async def`` one on the
    loop, and the decorator deliberately keeps those apart - promoting a sync endpoint
    to a coroutine would move a blocking call onto the event loop. Every raising
    endpoint in this suite was ``async def``, so the sync wrapper's failure path had
    never been executed by a server at all. Calling ``audited(...)(fn)()`` by hand would
    execute the line without ever asking whether FastAPI still gets its exception.
    """
    app = FastAPI()

    @app.get("/sync-boom")
    @audited(ledger, action="sync-boom")
    def sync_boom() -> Dict[str, Any]:
        raise ValueError("sync endpoint exploded")

    with TestClient(app) as client:
        with pytest.raises(ValueError):
            client.get("/sync-boom")

    assert [
        (entry.data["action"], entry.data.get("error")) for entry in entries(ledger)
    ] == [("sync-boom", "ValueError")]


def test_a_bad_failure_policy_is_rejected_at_decoration_time_with_a_factory(
    ledger: Ledger,
) -> None:
    """Deferring the check to the first request means the typo is found in production."""
    with pytest.raises(ConfigurationError):
        audited(lambda: ledger, audit_failure_policy="maybe")


# ======================================================================================
# The event loop must not be blocked
# ======================================================================================


def test_the_synchronous_ledger_append_does_not_run_on_the_event_loop_thread() -> None:
    """``Ledger.append`` is synchronous. Calling it from ``dispatch`` stalls the whole
    process for the duration of a disk write, on every audited request."""
    backend = ThreadRecordingBackend()
    with Ledger(backend=backend, ledger_id="threading-test") as led:
        app = FastAPI()
        app.add_middleware(SignLedgerAuditMiddleware, ledger=led)
        loop_threads: List[int] = []

        @app.get("/x")
        async def endpoint() -> Dict[str, Any]:
            loop_threads.append(threading.get_ident())
            return {"ok": True}

        with TestClient(app) as client:
            client.get("/x")

        assert backend.append_threads[0] != loop_threads[0]


# ======================================================================================
# What gets recorded
# ======================================================================================


def test_one_entry_is_appended_per_request(ledger: Ledger) -> None:
    with TestClient(make_app(ledger)) as client:
        client.get("/items")
        client.get("/items")
    assert ledger.count() == 2


def test_the_recorded_status_is_the_one_the_client_received(ledger: Ledger) -> None:
    with TestClient(make_app(ledger)) as client:
        response = client.get("/nope")
    assert (response.status_code, entries(ledger)[0].data["status"]) == (404, 404)


def test_the_recorded_method_and_path_match_the_request(ledger: Ledger) -> None:
    with TestClient(make_app(ledger)) as client:
        client.post("/login", json={"user": "ada"})
    data = entries(ledger)[0].data
    assert (data["method"], data["path"]) == ("POST", "/login")


def test_an_endpoint_that_raises_is_still_audited(ledger: Ledger) -> None:
    """The requests you most need audited are the ones that went wrong."""
    with TestClient(make_app(ledger), raise_server_exceptions=False) as client:
        client.get("/boom")
    assert entries(ledger)[0].data["status"] == 500


def test_an_endpoint_that_raises_records_the_exception_type(ledger: Ledger) -> None:
    with TestClient(make_app(ledger), raise_server_exceptions=False) as client:
        client.get("/boom")
    assert entries(ledger)[0].data["error"] == "RuntimeError"


def test_the_integration_names_itself_in_the_record(ledger: Ledger) -> None:
    with TestClient(make_app(ledger)) as client:
        client.get("/items")
    assert entries(ledger)[0].data["integration"] == "fastapi"


def test_the_chain_still_verifies_after_a_series_of_audited_requests(
    ledger: Ledger,
) -> None:
    with TestClient(make_app(ledger)) as client:
        for _ in range(5):
            client.get("/items")
    assert ledger.check_integrity().ok


def test_the_resolved_user_is_recorded(ledger: Ledger) -> None:
    app = make_app(ledger, user_resolver=lambda request: "user-42")
    with TestClient(app) as client:
        client.get("/items")
    assert entries(ledger)[0].data["user"] == "user-42"


def test_no_user_is_recorded_without_a_resolver(ledger: Ledger) -> None:
    with TestClient(make_app(ledger)) as client:
        client.get("/items")
    assert entries(ledger)[0].data["user"] is None


def test_a_websocket_connection_is_passed_through_untouched(ledger: Ledger) -> None:
    """Only HTTP scopes are audited; anything else must reach the app unchanged."""
    app = make_app(ledger)

    @app.websocket("/ws")
    async def socket(websocket: WebSocket) -> None:
        await websocket.accept()
        await websocket.send_text("hello")
        await websocket.close()

    with TestClient(app) as client, client.websocket_connect("/ws") as connection:
        assert connection.receive_text() == "hello"


# ======================================================================================
# A streamed response is not a settled one
#
# ``fail_closed`` forces the audit write to happen before ``http.response.start`` is
# forwarded - that is the only moment at which a request can still be refused. For a
# streamed body the outcome is not known then, and the entry is append-only, so the
# status it carries is a *promise*. Every one of these tests exists because a stream that
# died was, measurably, in the ledger as a plain ``200``.
# ======================================================================================


def test_a_streamed_response_is_recorded_as_provisional(ledger: Ledger) -> None:
    with TestClient(streaming_app(ledger)) as client:
        client.get("/stream")
    assert entries(ledger)[0].data["completion"] == "provisional"


def test_a_buffered_response_is_recorded_as_final(ledger: Ledger) -> None:
    """The whole body was in hand when the entry was written, so the status is settled."""
    with TestClient(make_app(ledger)) as client:
        client.get("/items")
    assert entries(ledger)[0].data["completion"] == "final"


def test_a_buffered_response_writes_exactly_one_entry(ledger: Ledger) -> None:
    """Settlement entries are for streams. Doubling every ledger would be a defect."""
    with TestClient(make_app(ledger)) as client:
        client.get("/items")
    assert ledger.count() == 1


def test_a_bodyless_status_is_not_mistaken_for_a_stream(ledger: Ledger) -> None:
    """A 204 declares no ``content-length`` either, and has nothing left to send."""
    with TestClient(streaming_app(ledger)) as client:
        assert client.get("/nothing").status_code == 204
    assert (entries(ledger)[0].data["completion"], ledger.count()) == ("final", 1)


def test_a_completed_stream_is_settled_by_a_second_entry(ledger: Ledger) -> None:
    with TestClient(streaming_app(ledger)) as client:
        client.get("/stream")
    assert kinds(ledger) == ["request", "response.completion"]


def test_a_completed_stream_settles_as_complete(ledger: Ledger) -> None:
    with TestClient(streaming_app(ledger)) as client:
        client.get("/stream")
    assert entries(ledger)[1].data["outcome"] == "complete"


def test_a_stream_that_dies_mid_body_is_not_left_recorded_as_a_success(
    ledger: Ledger,
) -> None:
    """The measured defect: one chunk out, then ``RuntimeError``, and the client saw a
    ``RemoteProtocolError`` - while the ledger said ``200`` and nothing else, forever."""
    with TestClient(streaming_app(ledger), raise_server_exceptions=False) as client:
        client.get("/broken-stream")
    assert entries(ledger)[1].data["outcome"] == "incomplete"


def test_a_stream_that_dies_mid_body_records_the_exception_type(ledger: Ledger) -> None:
    with TestClient(streaming_app(ledger), raise_server_exceptions=False) as client:
        client.get("/broken-stream")
    assert entries(ledger)[1].data["error"] == "RuntimeError"


def test_the_settlement_entry_carries_the_status_that_was_promised(
    ledger: Ledger,
) -> None:
    """So an auditor reading the settlement alone knows which claim it settles."""
    with TestClient(streaming_app(ledger), raise_server_exceptions=False) as client:
        client.get("/broken-stream")
    assert entries(ledger)[1].data["status"] == 200


def test_the_settlement_entry_carries_the_request_id_of_the_entry_it_settles(
    ledger: Ledger,
) -> None:
    with TestClient(streaming_app(ledger)) as client:
        client.get("/stream")
    request_entry, settlement = entries(ledger)
    assert settlement.data["request_id"] == request_entry.data["request_id"]


def test_the_settlement_entry_names_the_request_it_settles(ledger: Ledger) -> None:
    with TestClient(streaming_app(ledger)) as client:
        client.get("/stream")
    data = entries(ledger)[1].data
    assert (data["method"], data["path"]) == ("GET", "/stream")


def test_the_promised_entry_is_never_rewritten_by_the_settlement(ledger: Ledger) -> None:
    """Append-only means append-only: the first entry still says what it said."""
    with TestClient(streaming_app(ledger), raise_server_exceptions=False) as client:
        client.get("/broken-stream")
    data = entries(ledger)[0].data
    assert (data["status"], "error" not in data) == (200, True)


def test_a_client_that_disappears_mid_stream_is_recorded_as_incomplete(
    ledger: Ledger,
) -> None:
    """Starlette cancels the body task on disconnect and returns without a final chunk.

    Driven as raw ASGI because that is the only way to produce a disconnect faithfully:
    a test client always reads to the end.
    """

    async def abandoned_stream(scope: Dict[str, Any], receive: Any, send: Any) -> None:
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"row-0", "more_body": True})

    middleware = SignLedgerAuditMiddleware(abandoned_stream, ledger=ledger)
    call_asgi(middleware, http_scope(), [])
    assert entries(ledger)[1].data["outcome"] == "incomplete"


def test_a_client_that_disappears_mid_stream_records_no_exception_type(
    ledger: Ledger,
) -> None:
    """Nothing raised. Naming an error that did not happen is a different lie."""

    async def abandoned_stream(scope: Dict[str, Any], receive: Any, send: Any) -> None:
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"row-0", "more_body": True})

    middleware = SignLedgerAuditMiddleware(abandoned_stream, ledger=ledger)
    call_asgi(middleware, http_scope(), [])
    assert "error" not in entries(ledger)[1].data


def test_a_settlement_that_cannot_be_written_is_counted_not_raised(
    ledger: Ledger,
) -> None:
    """The client already has the response; there is no closing that door again.

    So the failure is counted on the recorder - where ``fail_open`` failures are already
    counted - rather than thrown into a server that can no longer act on it.
    """

    async def stream_then_break(scope: Dict[str, Any], receive: Any, send: Any) -> None:
        await send({"type": "http.response.start", "status": 200, "headers": []})
        ledger.backend.close()
        await send({"type": "http.response.body", "body": b"x", "more_body": False})

    middleware = SignLedgerAuditMiddleware(stream_then_break, ledger=ledger)
    call_asgi(middleware, http_scope(), [])
    assert (middleware.recorder.audit_failures, middleware.recorder.last_audit_error) == (
        1,
        "StorageError",
    )


def test_a_stream_that_could_not_be_audited_at_all_is_not_settled(
    ledger: Ledger,
) -> None:
    """Refused at ``http.response.start``: nothing was promised, so nothing is settled.

    A settlement here would only count the same broken ledger a second time.
    """

    async def streaming_asgi(scope: Dict[str, Any], receive: Any, send: Any) -> None:
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"x", "more_body": False})

    middleware = SignLedgerAuditMiddleware(streaming_asgi, ledger=ledger)
    ledger.backend.close()
    sent = call_asgi(middleware, http_scope(), [])
    assert (sent[0]["status"], middleware.recorder.audit_failures) == (500, 1)


# ======================================================================================
# The middleware entry and the decorator entry are two halves of one request
# ======================================================================================


def doubly_audited_app(ledger: Ledger) -> FastAPI:
    """A route covered by both the middleware and the decorator - a real deployment."""
    app = FastAPI()
    app.add_middleware(SignLedgerAuditMiddleware, ledger=ledger)

    @app.get("/items/{item_id}")
    @audited(ledger, action="item.read")
    async def read_item(item_id: int) -> Dict[str, Any]:
        return {"item_id": item_id}

    return app


def test_the_middleware_entry_says_what_kind_of_record_it_is(ledger: Ledger) -> None:
    with TestClient(make_app(ledger)) as client:
        client.get("/items")
    assert entries(ledger)[0].data["kind"] == "request"


def test_a_doubly_audited_request_writes_two_entries_that_name_themselves(
    ledger: Ledger,
) -> None:
    """Not deduplicated: both are real records. Labelled, so neither is mistaken for the
    other by an auditor counting requests."""
    with TestClient(doubly_audited_app(ledger)) as client:
        client.get("/items/42")
    assert kinds(ledger) == ["endpoint", "request"]


def test_the_decorator_entry_carries_the_path_and_method_of_the_request(
    ledger: Ledger,
) -> None:
    """Measured: the two entries shared only ``integration``, so an auditor querying on
    ``path`` saw half the requests and one querying on ``action`` saw the other half."""
    with TestClient(doubly_audited_app(ledger)) as client:
        client.get("/items/42")
    data = entries(ledger)[0].data
    assert (data["method"], data["path"]) == ("GET", "/items/42")


def test_the_two_entries_for_one_request_share_a_request_id(ledger: Ledger) -> None:
    """Path and method do not identify a request; two concurrent GETs share both."""
    with TestClient(doubly_audited_app(ledger)) as client:
        client.get("/items/42")
    endpoint_entry, request_entry = entries(ledger)
    assert endpoint_entry.data["request_id"] == request_entry.data["request_id"]


def test_two_requests_do_not_share_a_request_id(ledger: Ledger) -> None:
    with TestClient(make_app(ledger)) as client:
        client.get("/items")
        client.get("/items")
    first, second = entries(ledger)
    assert first.data["request_id"] != second.data["request_id"]


def test_a_decorated_call_outside_a_request_claims_no_request_context(
    ledger: Ledger,
) -> None:
    """Invented correlation is worse than none: the decorator is also used on plain
    calls, and there is no request there to belong to."""

    async def endpoint() -> Dict[str, Any]:
        return {"ok": True}

    import anyio

    anyio.run(audited(ledger)(endpoint))
    data = entries(ledger)[0].data
    assert [key for key in ("request_id", "path", "method") if key in data] == []


# ======================================================================================
# Body capture
# ======================================================================================


def test_the_body_is_not_recorded_by_default(ledger: Ledger) -> None:
    with TestClient(make_app(ledger)) as client:
        client.post("/login", json={"user": "ada"})
    assert entries(ledger)[0].data["body"] == "[OMITTED:disabled]"


def test_an_enabled_body_capture_records_the_redacted_body(ledger: Ledger) -> None:
    app = make_app(ledger, redaction=RedactionPolicy(capture_body=True))
    with TestClient(app) as client:
        client.post("/login", json={"user": "ada", "password": "hunter2"})
    assert entries(ledger)[0].data["body"] == {"user": "ada", "password": REDACTED}


def test_a_body_over_the_limit_is_recorded_as_omitted(ledger: Ledger) -> None:
    app = make_app(ledger, redaction=RedactionPolicy(capture_body=True, max_body_bytes=16))
    with TestClient(app) as client:
        client.post("/login", json={"user": "a" * 200})
    assert entries(ledger)[0].data["body"] == "[OMITTED:too-large]"


def test_the_endpoint_still_receives_a_body_the_middleware_captured(
    ledger: Ledger,
) -> None:
    """Auditing the body must not consume the request stream out from under the app."""
    app = FastAPI()
    app.add_middleware(
        SignLedgerAuditMiddleware, ledger=ledger, redaction=RedactionPolicy(capture_body=True)
    )
    seen: Dict[str, Any] = {}

    @app.post("/echo")
    async def echo(payload: Dict[str, Any]) -> Dict[str, Any]:
        seen.update(payload)
        return {"ok": True}

    with TestClient(app) as client:
        client.post("/echo", json={"user": "ada"})
    assert seen == {"user": "ada"}


def test_a_body_the_endpoint_never_read_is_marked_rather_than_guessed(
    ledger: Ledger,
) -> None:
    """The middleware observes the stream; it does not drain it on the app's behalf."""
    app = FastAPI()
    app.add_middleware(
        SignLedgerAuditMiddleware, ledger=ledger, redaction=RedactionPolicy(capture_body=True)
    )

    @app.post("/ignore")
    async def ignore() -> Dict[str, Any]:
        return {"ok": True}

    with TestClient(app) as client:
        client.post("/ignore", json={"user": "ada"})
    assert entries(ledger)[0].data["body"] == "[OMITTED:not-read]"


# ======================================================================================
# audit_failure_policy
# ======================================================================================


def test_fail_closed_returns_500_to_the_client(ledger: Ledger) -> None:
    """The audit record is what makes the response defensible; without it, no 200.

    In ASGI this has to happen before the response start is forwarded, or the client
    already has its answer and nothing later can change it.
    """
    app = make_app(ledger)
    with TestClient(app) as client:
        ledger.backend.close()
        assert client.get("/items").status_code == 500


def test_fail_closed_does_not_deliver_the_endpoint_response_body(
    ledger: Ledger,
) -> None:
    app = make_app(ledger)
    with TestClient(app) as client:
        ledger.backend.close()
        assert b"items" not in client.get("/items").content


def test_fail_open_still_serves_the_real_response(ledger: Ledger) -> None:
    app = make_app(ledger, audit_failure_policy=FAIL_OPEN)
    with TestClient(app) as client:
        ledger.backend.close()
        response = client.get("/items")
    assert (response.status_code, response.json()) == (200, {"items": []})


def test_a_fail_open_audit_failure_is_counted_not_swallowed(ledger: Ledger) -> None:
    """Wrapped by hand rather than through ``add_middleware`` so the test can reach the
    recorder and read the counter. It is the same ASGI object either way."""
    inner = FastAPI()

    @inner.get("/items")
    async def items() -> Dict[str, Any]:
        return {"items": []}

    middleware = SignLedgerAuditMiddleware(
        inner, ledger=ledger, audit_failure_policy=FAIL_OPEN
    )
    with TestClient(middleware) as client:
        ledger.backend.close()
        client.get("/items")
    assert (middleware.recorder.audit_failures, middleware.recorder.last_audit_error) == (
        1,
        "StorageError",
    )


def test_an_unknown_audit_failure_policy_is_rejected_at_construction(
    ledger: Ledger,
) -> None:
    with pytest.raises(ConfigurationError):
        SignLedgerAuditMiddleware(None, ledger=ledger, audit_failure_policy="maybe")


# ======================================================================================
# Trusted proxies
# ======================================================================================


def test_x_forwarded_for_is_ignored_when_no_proxies_are_trusted(ledger: Ledger) -> None:
    with TestClient(make_app(ledger)) as client:
        client.get("/items", headers={"X-Forwarded-For": "1.2.3.4"})
    assert entries(ledger)[0].data["client_ip"] != "1.2.3.4"


def test_the_client_ip_is_taken_from_the_trusted_position_in_the_chain(
    ledger: Ledger,
) -> None:
    app = make_app(ledger, redaction=RedactionPolicy(trusted_proxy_count=1))
    with TestClient(app) as client:
        client.get("/items", headers={"X-Forwarded-For": "evil, 203.0.113.7"})
    assert entries(ledger)[0].data["client_ip"] == "203.0.113.7"


# ======================================================================================
# skip_paths
# ======================================================================================


def test_a_skipped_path_writes_no_entry(ledger: Ledger) -> None:
    app = make_app(ledger, skip_paths=("/health",))
    with TestClient(app) as client:
        client.get("/health")
    assert ledger.count() == 0


def test_a_path_that_is_not_skipped_still_writes(ledger: Ledger) -> None:
    app = make_app(ledger, skip_paths=("/health",))
    with TestClient(app) as client:
        client.get("/items")
    assert ledger.count() == 1


# ======================================================================================
# The audit-inspection endpoints
# ======================================================================================


def token_auth(request: Any) -> bool:
    return bool(request.headers.get("x-audit-token") == "let-me-in")


def inspect_app(ledger: Ledger, **kwargs: Any) -> FastAPI:
    app = FastAPI()
    app.include_router(audit_router(ledger, **kwargs))
    return app


def test_the_audit_router_cannot_be_built_without_an_authenticator(
    ledger: Ledger,
) -> None:
    with pytest.raises(ConfigurationError):
        audit_router(ledger, authenticator=None)


def test_an_unauthenticated_status_request_is_rejected(ledger: Ledger) -> None:
    with TestClient(inspect_app(ledger, authenticator=token_auth)) as client:
        assert client.get("/_audit/status").status_code == 401


def test_an_unauthenticated_entries_request_is_rejected(ledger: Ledger) -> None:
    """The unauthenticated full-ledger scan: DoS and disclosure in one request."""
    with TestClient(inspect_app(ledger, authenticator=token_auth)) as client:
        assert client.get("/_audit/entries").status_code == 401


def test_an_unauthenticated_verify_request_is_rejected(ledger: Ledger) -> None:
    with TestClient(inspect_app(ledger, authenticator=token_auth)) as client:
        assert client.get("/_audit/verify").status_code == 401


def test_a_rejected_request_returns_no_ledger_content(ledger: Ledger) -> None:
    ledger.append({"secret_business_fact": "acquisition"})
    with TestClient(inspect_app(ledger, authenticator=token_auth)) as client:
        assert b"acquisition" not in client.get("/_audit/entries").content


def test_an_authenticated_status_request_reports_the_real_entry_count(
    ledger: Ledger,
) -> None:
    ledger.append({"a": 1})
    with TestClient(inspect_app(ledger, authenticator=token_auth)) as client:
        response = client.get("/_audit/status", headers={"x-audit-token": "let-me-in"})
    assert response.json()["entry_count"] == 1


def test_the_entries_endpoint_caps_the_page_size(ledger: Ledger) -> None:
    for index in range(10):
        ledger.append({"n": index})
    app = inspect_app(ledger, authenticator=token_auth, max_page_size=3)
    with TestClient(app) as client:
        response = client.get(
            "/_audit/entries?limit=1000", headers={"x-audit-token": "let-me-in"}
        )
    assert len(response.json()["entries"]) == 3


def test_the_entries_endpoint_paginates_with_an_offset(ledger: Ledger) -> None:
    for index in range(5):
        ledger.append({"n": index})
    app = inspect_app(ledger, authenticator=token_auth)
    with TestClient(app) as client:
        response = client.get(
            "/_audit/entries?offset=3&limit=2", headers={"x-audit-token": "let-me-in"}
        )
    assert [row["sequence"] for row in response.json()["entries"]] == [3, 4]


def test_the_entries_endpoint_rejects_a_zero_limit(ledger: Ledger) -> None:
    app = inspect_app(ledger, authenticator=token_auth)
    with TestClient(app) as client:
        response = client.get("/_audit/entries?limit=0", headers={"x-audit-token": "let-me-in"})
    assert response.status_code == 422


def test_the_entries_endpoint_never_returns_entry_payloads(ledger: Ledger) -> None:
    """Identifiers and digests are enough to audit; payloads are the disclosure risk.

    The sentinel deliberately contains non-hex letters: ``entry_id`` is a uuid4 hex
    string, so a value like ``"ada"`` can turn up inside one by chance and this
    assertion would fail at random.
    """
    ledger.append({"patient_name": "zsofia-quinn"})
    app = inspect_app(ledger, authenticator=token_auth)
    with TestClient(app) as client:
        response = client.get("/_audit/entries", headers={"x-audit-token": "let-me-in"})
    assert b"zsofia-quinn" not in response.content


def test_the_entries_endpoint_returns_only_the_safe_field_set(ledger: Ledger) -> None:
    """Structural, so it cannot pass by accident: no ``data``, no ``metadata``."""
    ledger.append({"patient_name": "zsofia-quinn"})
    app = inspect_app(ledger, authenticator=token_auth)
    with TestClient(app) as client:
        response = client.get("/_audit/entries", headers={"x-audit-token": "let-me-in"})
    assert set(response.json()["entries"][0]) == {
        "sequence",
        "entry_id",
        "timestamp",
        "hash",
        "previous_hash",
        "key_id",
    }


def test_the_entries_endpoint_returns_the_sequence_and_digest(ledger: Ledger) -> None:
    entry = ledger.append({"a": 1})
    app = inspect_app(ledger, authenticator=token_auth)
    with TestClient(app) as client:
        response = client.get("/_audit/entries", headers={"x-audit-token": "let-me-in"})
    row = response.json()["entries"][0]
    assert (row["sequence"], row["hash"]) == (0, entry.hash)


def test_the_verify_endpoint_reports_a_healthy_chain(ledger: Ledger) -> None:
    ledger.append({"a": 1})
    app = inspect_app(ledger, authenticator=token_auth)
    with TestClient(app) as client:
        response = client.get("/_audit/verify", headers={"x-audit-token": "let-me-in"})
    assert response.json()["ok"] is True


def test_the_verify_endpoint_reports_an_empty_ledger_as_not_ok(ledger: Ledger) -> None:
    app = inspect_app(ledger, authenticator=token_auth)
    with TestClient(app) as client:
        response = client.get("/_audit/verify", headers={"x-audit-token": "let-me-in"})
    assert response.json()["ok"] is False


def test_the_verify_endpoint_names_the_assurance_level(ledger: Ledger) -> None:
    ledger.append({"a": 1})
    app = inspect_app(ledger, authenticator=token_auth)
    with TestClient(app) as client:
        response = client.get("/_audit/verify", headers={"x-audit-token": "let-me-in"})
    assert response.json()["assurance"] == "chain_only"


def test_the_router_can_be_mounted_under_a_different_prefix(ledger: Ledger) -> None:
    app = inspect_app(ledger, authenticator=token_auth, prefix="/internal/audit")
    with TestClient(app) as client:
        response = client.get(
            "/internal/audit/status", headers={"x-audit-token": "let-me-in"}
        )
    assert response.status_code == 200


def test_a_zero_max_page_size_is_rejected(ledger: Ledger) -> None:
    with pytest.raises(ConfigurationError):
        audit_router(ledger, authenticator=token_auth, max_page_size=0)


# ======================================================================================
# Reading the trail is not a write to it
#
# Every application here carries *both* halves - the middleware and the router it must
# not audit - because that is the only arrangement in which the defect exists. Each test
# drives a real ASGI stack and then counts what the ledger gained.
# ======================================================================================


def audited_inspect_app(ledger: Ledger, **kwargs: Any) -> FastAPI:
    """An application whose audit trail is both written and served by itself."""
    app = FastAPI()
    app.add_middleware(SignLedgerAuditMiddleware, ledger=ledger)
    app.include_router(audit_router(ledger, authenticator=token_auth, **kwargs))

    @app.get("/items")
    async def items() -> Dict[str, Any]:
        return {"items": []}

    return app


AUTH = {"x-audit-token": "let-me-in"}


def test_reading_the_status_endpoint_appends_nothing(ledger: Ledger) -> None:
    app = audited_inspect_app(ledger)
    with TestClient(app) as client:
        assert client.get("/_audit/status", headers=AUTH).status_code == 200
    assert ledger.count() == 0


def test_reading_the_entries_endpoint_appends_nothing(ledger: Ledger) -> None:
    app = audited_inspect_app(ledger)
    with TestClient(app) as client:
        assert client.get("/_audit/entries", headers=AUTH).status_code == 200
    assert ledger.count() == 0


def test_an_unauthenticated_probe_appends_nothing(ledger: Ledger) -> None:
    """The 401 is the cheapest authenticated-endpoint probe there is, and letting it
    write means an anonymous caller can grow an unprunable store at will - against the
    docstring that justifies the mandatory authenticator as protection from exactly
    that."""
    app = audited_inspect_app(ledger)
    with TestClient(app) as client:
        assert client.get("/_audit/entries").status_code == 401
    assert ledger.count() == 0


def test_a_rejected_query_parameter_appends_nothing(ledger: Ledger) -> None:
    """A 422 from ``Query(ge=1)`` never enters the endpoint, and must still not be
    audited: the route answered, which is what settles it."""
    app = audited_inspect_app(ledger)
    with TestClient(app) as client:
        assert client.get("/_audit/entries?limit=0", headers=AUTH).status_code == 422
    assert ledger.count() == 0


def test_an_unrouted_path_under_the_mount_appends_nothing(ledger: Ledger) -> None:
    """The cheapest probe on the whole application: no credential, no route, no
    endpoint ever entered. It is why the exclusion cannot be a list of the paths the
    router registered."""
    app = audited_inspect_app(ledger)
    with TestClient(app) as client:
        assert client.get("/_audit/bogus").status_code == 404
    assert ledger.count() == 0


def test_a_method_the_inspection_route_does_not_serve_appends_nothing(
    ledger: Ledger,
) -> None:
    app = audited_inspect_app(ledger)
    with TestClient(app) as client:
        assert client.post("/_audit/status", headers=AUTH).status_code == 405
    assert ledger.count() == 0


def test_verify_reports_the_ledger_it_leaves_behind(ledger: Ledger) -> None:
    """It used to answer ``entries_verified=n`` while appending the n+1th, so every
    verification cost more than the one before it."""
    app = audited_inspect_app(ledger)
    with TestClient(app) as client:
        client.get("/items")
        client.get("/items")
        report = client.get("/_audit/verify", headers=AUTH).json()
    assert (report["entries_verified"], ledger.count()) == (2, 2)


def test_the_exclusion_survives_a_mounted_sub_application(ledger: Ledger) -> None:
    """``route.path`` is application-relative and ``scope["path"]`` is not. Under a
    mount the two differ by exactly the mount point, and an exclusion that compares them
    directly silently does nothing."""
    root = FastAPI()
    root.mount("/api", audited_inspect_app(ledger))
    with TestClient(root) as client:
        assert client.get("/api/_audit/status", headers=AUTH).status_code == 200
    assert ledger.count() == 0


def test_an_unrouted_probe_under_a_mounted_sub_application_appends_nothing(
    ledger: Ledger,
) -> None:
    """Starlette names the *mounted application* as the endpoint of a request its own
    router then failed to route, so "something answered" is not the same question as
    "one of ours answered"."""
    root = FastAPI()
    root.mount("/api", audited_inspect_app(ledger))
    with TestClient(root) as client:
        assert client.get("/api/_audit/bogus").status_code == 404
    assert ledger.count() == 0


def test_the_exclusion_survives_a_root_path(ledger: Ledger) -> None:
    """``root_path`` travels with the request instead of being stripped from it, so the
    path the client asked for is not the path the router was registered at."""
    app = FastAPI(root_path="/gw")
    app.add_middleware(SignLedgerAuditMiddleware, ledger=ledger)
    app.include_router(audit_router(ledger, authenticator=token_auth))
    with TestClient(app, root_path="/gw") as client:
        assert client.get("/gw/_audit/status", headers=AUTH).status_code == 200
    assert ledger.count() == 0


def test_an_unrouted_probe_under_a_root_path_appends_nothing(ledger: Ledger) -> None:
    app = FastAPI(root_path="/gw")
    app.add_middleware(SignLedgerAuditMiddleware, ledger=ledger)
    app.include_router(audit_router(ledger, authenticator=token_auth))
    with TestClient(app, root_path="/gw") as client:
        assert client.get("/gw/_audit/bogus").status_code == 404
    assert ledger.count() == 0


def test_an_application_route_sharing_the_prefix_is_still_audited(ledger: Ledger) -> None:
    """Quietly dropping part of the trail is a worse defect than the one being fixed,
    and unlike it, invisible from outside. The prefix belongs to the router's own
    endpoints, not to every path that starts with it."""
    app = audited_inspect_app(ledger)

    @app.get("/_audit/dashboard")
    async def dashboard() -> Dict[str, Any]:
        return {"ok": True}

    with TestClient(app) as client:
        assert client.get("/_audit/dashboard").status_code == 200
    assert [entry.data["path"] for entry in entries(ledger)] == ["/_audit/dashboard"]


def test_a_method_an_application_route_under_the_prefix_refuses_is_still_audited(
    ledger: Ledger,
) -> None:
    app = audited_inspect_app(ledger)

    @app.get("/_audit/dashboard")
    async def dashboard() -> Dict[str, Any]:
        return {"ok": True}

    with TestClient(app) as client:
        assert client.post("/_audit/dashboard").status_code == 405
    assert [entry.data["status"] for entry in entries(ledger)] == [405]


def test_an_inspection_route_wins_against_an_application_catch_all(
    ledger: Ledger,
) -> None:
    """The single-page-application fallback, which claims every path there is.

    "Does the application have a route here?" is the wrong question on its own: with a
    catch-all registered, the answer is always yes, and every inspection request would be
    audited again. What settles it is that one of the router's *own* endpoints answered.
    """
    app = audited_inspect_app(ledger)

    @app.get("/{spa_path:path}")
    async def spa(spa_path: str) -> Dict[str, Any]:
        return {"path": spa_path}

    with TestClient(app) as client:
        assert client.get("/_audit/status", headers=AUTH).status_code == 200
        assert client.get("/_audit/entries").status_code == 401
    assert ledger.count() == 0


def test_a_catch_all_answering_under_the_prefix_is_audited_like_any_other_route(
    ledger: Ledger,
) -> None:
    """The other half of the same question. A path under the mount that the *application*
    answers is the application answering, whatever it looks like from the URL."""
    app = audited_inspect_app(ledger)

    @app.get("/{spa_path:path}")
    async def spa(spa_path: str) -> Dict[str, Any]:
        return {"path": spa_path}

    with TestClient(app) as client:
        assert client.get("/_audit/bogus").status_code == 200
    assert [entry.data["path"] for entry in entries(ledger)] == ["/_audit/bogus"]


def test_an_empty_prefix_is_refused_rather_than_disabling_the_trail(
    ledger: Ledger,
) -> None:
    """A router at the root cannot be told apart from the site it reports on. Refusing
    is recoverable; silently auditing nothing at all is not. Flask refuses the same
    thing for the same reason."""
    with pytest.raises(ConfigurationError):
        audit_router(ledger, authenticator=token_auth, prefix="")


def test_a_prefix_of_only_slashes_is_refused_too(ledger: Ledger) -> None:
    with pytest.raises(ConfigurationError):
        audit_router(ledger, authenticator=token_auth, prefix="/")


def test_a_custom_prefix_is_the_one_excluded(ledger: Ledger) -> None:
    app = audited_inspect_app(ledger, prefix="/internal/audit")
    with TestClient(app) as client:
        assert client.get("/internal/audit/status", headers=AUTH).status_code == 200
        assert client.get("/internal/audit/bogus").status_code == 404
    assert ledger.count() == 0


def test_a_prefix_added_by_include_router_is_the_one_excluded(ledger: Ledger) -> None:
    """``include_router(..., prefix=...)`` moves the endpoints after the router was
    built, so the argument :func:`audit_router` was given is not where they answer."""
    app = FastAPI()
    app.add_middleware(SignLedgerAuditMiddleware, ledger=ledger)
    app.include_router(audit_router(ledger, authenticator=token_auth), prefix="/ops")
    with TestClient(app) as client:
        assert client.get("/ops/_audit/status", headers=AUTH).status_code == 200
        assert client.get("/ops/_audit/bogus").status_code == 404
    assert ledger.count() == 0


def test_a_router_included_after_the_first_request_is_still_excluded(
    ledger: Ledger,
) -> None:
    """A factory is free to include the router late, and an answer worked out once and
    kept would be wrong for as long as the process lives."""
    app = FastAPI()
    app.add_middleware(SignLedgerAuditMiddleware, ledger=ledger)

    @app.get("/items")
    async def items() -> Dict[str, Any]:
        return {"items": []}

    with TestClient(app) as client:
        client.get("/items")
        app.include_router(audit_router(ledger, authenticator=token_auth))
        assert client.get("/_audit/status", headers=AUTH).status_code == 200
        assert client.get("/_audit/bogus").status_code == 404
    assert ledger.count() == 1


def test_the_exclusion_holds_when_the_middleware_is_not_the_direct_wrapper(
    ledger: Ledger,
) -> None:
    """The ``--preload`` shape the module documents: the thing the middleware holds is
    another middleware, which has no routes and is not the application."""
    inner = FastAPI()
    inner.include_router(audit_router(ledger, authenticator=token_auth))

    @inner.get("/items")
    async def items() -> Dict[str, Any]:
        return {"items": []}

    app = SignLedgerAuditMiddleware(
        CORSMiddleware(inner, allow_origins=["*"]), ledger=ledger
    )
    with TestClient(app) as client:
        assert client.get("/_audit/status", headers=AUTH).status_code == 200
        assert client.get("/_audit/bogus").status_code == 404
        assert client.get("/items").status_code == 200
    assert [entry.data["path"] for entry in entries(ledger)] == ["/items"]


def test_ordinary_application_routes_are_still_audited(ledger: Ledger) -> None:
    """The exclusion is the router's own mount and nothing else."""
    app = audited_inspect_app(ledger)
    with TestClient(app) as client:
        client.get("/items")
        client.get("/nowhere")
    assert [entry.data["path"] for entry in entries(ledger)] == ["/items", "/nowhere"]


def test_an_inspection_request_that_raises_is_still_not_in_the_trail(
    ledger: Ledger,
) -> None:
    """An inspection request is no more part of the trail for having blown up than it
    would have been for succeeding."""

    class Broken:
        def count(self) -> int:
            raise RuntimeError("backend gone")

        def get_stats_snapshot(self) -> Dict[str, Any]:
            return {}

    app = FastAPI()
    app.add_middleware(SignLedgerAuditMiddleware, ledger=ledger)
    app.include_router(audit_router(Broken(), authenticator=token_auth))
    with TestClient(app) as client:
        with pytest.raises(RuntimeError):
            client.get("/_audit/status", headers=AUTH)
    assert ledger.count() == 0


def test_an_application_with_no_audit_router_audits_every_path(ledger: Ledger) -> None:
    """Nothing about this exclusion applies to an application that does not serve its
    own trail."""
    app = make_app(ledger)
    with TestClient(app) as client:
        client.get("/_audit/status")
        client.get("/items")
    assert [entry.data["path"] for entry in entries(ledger)] == [
        "/_audit/status",
        "/items",
    ]


# ======================================================================================
# The module must import at all
# ======================================================================================


def test_the_module_imports_cleanly_in_a_fresh_interpreter() -> None:
    result = subprocess.run(
        [sys.executable, "-c", "import signledger.contrib.fastapi"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )
    assert result.returncode == 0, result.stderr


# ======================================================================================
# Raw ASGI edges
#
# Driven by hand-built ASGI messages rather than a client. ASGI *is* the real interface
# here, and these are shapes a well-behaved HTTP client will not produce - a lying
# Content-Length, a missing peer, a response that starts and then fails.
# ======================================================================================


def call_asgi(app: Any, scope: Dict[str, Any], messages: List[Dict[str, Any]]) -> List[Any]:
    """Drive one ASGI request to completion and return everything it sent."""
    import anyio

    sent: List[Any] = []
    inbox = list(messages)

    async def receive() -> Dict[str, Any]:
        return inbox.pop(0) if inbox else {"type": "http.disconnect"}

    async def send(message: Dict[str, Any]) -> None:
        sent.append(message)

    async def main() -> None:
        await app(scope, receive, send)

    anyio.run(main)
    return sent


def http_scope(**overrides: Any) -> Dict[str, Any]:
    scope: Dict[str, Any] = {
        "type": "http",
        "method": "POST",
        "path": "/x",
        "query_string": b"",
        "headers": [(b"content-type", b"application/json"), (b"content-length", b"10")],
        "client": ("10.0.0.9", 51000),
    }
    scope.update(overrides)
    return scope


async def draining_app(scope: Dict[str, Any], receive: Any, send: Any) -> None:
    """An app that reads the whole request body, then answers 200."""
    more = True
    while more:
        message = await receive()
        more = message.get("more_body", False)
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.body", "body": b"{}"})


def test_a_body_that_exceeds_its_declared_content_length_is_not_buffered_without_bound(
    ledger: Ledger,
) -> None:
    """Content-Length is a claim. A client that lies must not choose our memory usage.

    The assertion is ``too-large`` rather than ``OVERSIZED`` on purpose. ``OVERSIZED``
    only says the *decoder* noticed afterwards, which it would even if the middleware had
    happily buffered a gigabyte first. ``too-large`` can only be produced by the receive
    loop giving up, so it is the one answer that proves buffering actually stopped.
    """
    middleware = SignLedgerAuditMiddleware(
        draining_app, ledger=ledger, redaction=RedactionPolicy(capture_body=True, max_body_bytes=64)
    )
    chunk = {"type": "http.request", "body": b"a" * 100, "more_body": True}
    call_asgi(
        middleware,
        http_scope(),
        [chunk, chunk, {"type": "http.request", "body": b"a" * 100, "more_body": False}],
    )
    assert entries(ledger)[0].data["body"] == "[OMITTED:too-large]"


def test_a_partially_buffered_body_is_discarded_rather_than_stored_truncated(
    ledger: Ledger,
) -> None:
    """A half-read JSON fragment in an audit log is worse than an honest omission."""
    middleware = SignLedgerAuditMiddleware(
        draining_app, ledger=ledger, redaction=RedactionPolicy(capture_body=True, max_body_bytes=64)
    )
    call_asgi(
        middleware,
        http_scope(),
        [
            {"type": "http.request", "body": b'{"user": "ada", ', "more_body": True},
            {"type": "http.request", "body": b'"n": ' + b"9" * 100, "more_body": False},
        ],
    )
    assert "ada" not in str(entries(ledger)[0].data["body"])


def test_an_unparseable_content_length_is_treated_as_absent(ledger: Ledger) -> None:
    middleware = SignLedgerAuditMiddleware(
        draining_app, ledger=ledger, redaction=RedactionPolicy(capture_body=True)
    )
    call_asgi(
        middleware,
        http_scope(headers=[(b"content-type", b"application/json"), (b"content-length", b"ten")]),
        [{"type": "http.request", "body": b"{}", "more_body": False}],
    )
    assert entries(ledger)[0].data["body"] == "[OMITTED:no-content-length]"


def test_a_scope_with_no_peer_records_no_client_ip(ledger: Ledger) -> None:
    """ASGI allows ``client`` to be ``None`` - a unix socket, for instance."""
    middleware = SignLedgerAuditMiddleware(draining_app, ledger=ledger)
    call_asgi(
        middleware,
        http_scope(client=None),
        [{"type": "http.request", "body": b"{}", "more_body": False}],
    )
    assert entries(ledger)[0].data["client_ip"] is None


def test_an_exception_raised_after_the_response_started_writes_no_second_request_entry(
    ledger: Ledger,
) -> None:
    """The request entry was written when the status was known and is not rewritten,
    contradicted, or repeated. What follows it is a settlement, which is a different
    kind of record and says so."""

    async def failing_app(scope: Dict[str, Any], receive: Any, send: Any) -> None:
        await send({"type": "http.response.start", "status": 200, "headers": []})
        raise RuntimeError("died mid-stream")

    middleware = SignLedgerAuditMiddleware(failing_app, ledger=ledger)
    with pytest.raises(RuntimeError):
        call_asgi(middleware, http_scope(), [])
    assert kinds(ledger) == ["request", "response.completion"]


def test_the_entry_written_before_a_mid_stream_failure_records_the_real_status(
    ledger: Ledger,
) -> None:
    async def failing_app(scope: Dict[str, Any], receive: Any, send: Any) -> None:
        await send({"type": "http.response.start", "status": 201, "headers": []})
        raise RuntimeError("died mid-stream")

    middleware = SignLedgerAuditMiddleware(failing_app, ledger=ledger)
    with pytest.raises(RuntimeError):
        call_asgi(middleware, http_scope(), [])
    assert entries(ledger)[0].data["status"] == 201
