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
from typing import Any, Dict, Iterator, List

import pytest
from fastapi import FastAPI, WebSocket
from fastapi.testclient import TestClient

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
    assert entries(ledger)[0].data["query"]["api_key"] == REDACTED


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


def test_an_exception_raised_after_the_response_started_is_not_audited_twice(
    ledger: Ledger,
) -> None:
    """The entry was already written when the status was known; a later failure must
    not append a second, contradictory record for the same request."""

    async def failing_app(scope: Dict[str, Any], receive: Any, send: Any) -> None:
        await send({"type": "http.response.start", "status": 200, "headers": []})
        raise RuntimeError("died mid-stream")

    middleware = SignLedgerAuditMiddleware(failing_app, ledger=ledger)
    with pytest.raises(RuntimeError):
        call_asgi(middleware, http_scope(), [])
    assert ledger.count() == 1


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
