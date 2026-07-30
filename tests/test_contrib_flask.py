"""Tests for :mod:`signledger.contrib.flask`, driven through a real Flask test client.

Two v1 failures are pinned here permanently:

* the module could not be imported at all - a bare ``Tuple`` sat at class-body scope, so
  even after the shadowing package was removed the first ``import`` raised ``NameError``;
* what did run wrote credentials into an append-only store.

Nothing below mocks Flask or the ledger. Every assertion is made about a real request
that went through a real WSGI stack and a real hash-chained entry that came back out of a
real backend.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List

import pytest
from flask import Flask, Response
from werkzeug.test import EnvironBuilder

from signledger.backends.memory import MemoryBackend
from signledger.contrib import FAIL_OPEN, AuditWriteError
from signledger.contrib.flask import SignLedgerAudit, audit_blueprint
from signledger.contrib.redaction import REDACTED, RedactionPolicy
from signledger.core.exceptions import ConfigurationError
from signledger.core.ledger import Ledger

REPO_ROOT = str(__import__("pathlib").Path(__file__).resolve().parent.parent)


@pytest.fixture
def ledger() -> Iterator[Ledger]:
    with Ledger(backend=MemoryBackend(), ledger_id="flask-test") as led:
        yield led


def make_app(ledger: Ledger, **kwargs: Any) -> Flask:
    """A minimal but genuine Flask application with the audit extension installed."""
    app = Flask(__name__)
    app.config["PROPAGATE_EXCEPTIONS"] = False
    SignLedgerAudit(app, ledger=ledger, **kwargs)

    @app.post("/login")
    def login() -> Dict[str, Any]:
        return {"ok": True}

    @app.get("/items")
    def items() -> Dict[str, Any]:
        return {"items": []}

    @app.get("/boom")
    def boom() -> Dict[str, Any]:
        raise RuntimeError("view exploded")

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return {"ok": True}

    return app


def entries(ledger: Ledger) -> List[Any]:
    return list(ledger.iter_entries())


def kinds(ledger: Ledger) -> List[Any]:
    return [entry.data.get("kind") for entry in entries(ledger)]


def streaming_app(ledger: Ledger, **kwargs: Any) -> Flask:
    """Views whose body is produced *after* ``after_request`` has already run."""
    app = Flask(__name__)
    app.config["PROPAGATE_EXCEPTIONS"] = False
    SignLedgerAudit(app, ledger=ledger, **kwargs)

    @app.get("/stream")
    def stream() -> Response:
        def body() -> Iterator[bytes]:
            yield b'{"row": 0}\n'
            yield b'{"row": 1}\n'

        return Response(body(), mimetype="application/x-ndjson")

    @app.get("/broken-stream")
    def broken_stream() -> Response:
        def body() -> Iterator[bytes]:
            yield b'{"row": 0}\n'
            raise RuntimeError("settlement feed died mid-stream")

        return Response(body(), mimetype="application/x-ndjson")

    @app.get("/plain-iterator")
    def plain_iterator() -> Response:
        """An iterable with no ``close`` at all - a list iterator is one."""
        return Response(iter([b'{"row": 0}\n']), mimetype="application/x-ndjson")

    return app


def run_wsgi(app: Flask, path: str) -> List[bytes]:
    """Drive one request the way a conforming WSGI server does.

    PEP 3333: "If the iterable returned by the application has a ``close()`` method, the
    server or gateway **must** call that method upon completion of the current request,
    whether the request was completed normally or terminated early due to an application
    error." gunicorn, waitress and ``werkzeug.serving`` all do exactly this.

    Werkzeug's *test client* does not: it pulls one chunk, hands back a response whose
    body is consumed lazily when the test touches ``.data``, and abandons the iterator
    outright if it raises. A streamed body that dies therefore cannot be observed through
    it at all - which is also why the defect this drives at was invisible until the
    consumer project ran under a real server.
    """
    iterable = wsgi_request(app, path)
    chunks: List[bytes] = []
    try:
        for chunk in iterable:
            chunks.append(chunk)
    finally:
        close = getattr(iterable, "close", None)
        if close is not None:
            close()
    return chunks


def wsgi_request(app: Flask, path: str) -> Any:
    """The raw WSGI iterable for one request, not yet consumed."""

    def start_response(status: str, headers: Any, exc_info: Any = None) -> Any:
        return lambda chunk: None

    return app(EnvironBuilder(path=path).get_environ(), start_response)


# ======================================================================================
# The headline requirement: credentials must not reach the ledger
# ======================================================================================


def test_an_authorization_header_and_a_password_body_leak_neither_into_the_ledger(
    ledger: Ledger,
) -> None:
    """The exact v1 defect, asserted against the entry's whole serialized form.

    Not "the header field is redacted" - the *serialized entry*, everything the store
    will hold forever, must not contain either secret anywhere.
    """
    app = make_app(ledger, redaction=RedactionPolicy(capture_body=True))
    client = app.test_client()

    client.post(
        "/login",
        headers={"Authorization": "Bearer sk-live-51H8xQsecret"},
        json={"user": "ada", "password": "hunter2"},
    )

    stored = json.dumps(entries(ledger)[0].model_dump(), default=str)
    assert "sk-live-51H8xQsecret" not in stored
    assert "hunter2" not in stored


def test_the_authorization_header_is_present_but_replaced_with_the_marker(
    ledger: Ledger,
) -> None:
    """Redacted, not dropped: an auditor must see that a credential was presented."""
    client = make_app(ledger).test_client()
    client.get("/items", headers={"Authorization": "Bearer secret-value"})
    assert entries(ledger)[0].data["headers"]["authorization"] == REDACTED


def test_a_cookie_header_is_never_recorded(ledger: Ledger) -> None:
    """A session cookie is a bearer credential in a different envelope.

    Set through the client's real cookie jar, not a ``headers=`` entry: Werkzeug's test
    client silently drops a hand-written ``Cookie`` header, so that version of this test
    would have passed against a middleware that recorded cookies verbatim.
    """
    client = make_app(ledger).test_client()
    client.set_cookie("session", "abc123", domain="localhost")
    client.get("/items")
    assert entries(ledger)[0].data["headers"]["cookie"] == REDACTED


def test_a_query_parameter_carrying_a_token_is_redacted(ledger: Ledger) -> None:
    client = make_app(ledger).test_client()
    client.get("/items?page=2&access_token=abc123")
    query = entries(ledger)[0].data["query"]
    assert (query["access_token"], query["page"]) == ([REDACTED], ["2"])


# ======================================================================================
# The module must be importable at all (v1 raised NameError on import)
# ======================================================================================


def test_the_module_imports_cleanly_in_a_fresh_interpreter() -> None:
    """v1 had a bare ``Tuple`` at class-body scope; the very first import raised.

    Run in a subprocess so a module already cached by another test cannot hide it.
    """
    result = subprocess.run(
        [sys.executable, "-c", "import signledger.contrib.flask"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_the_flask_module_is_not_shadowed_by_an_empty_package() -> None:
    """v1 shipped four empty directories that shadowed their own sibling modules.

    Compared as path components rather than as a string: ``__file__`` is separated with
    backslashes on Windows, so ``endswith("contrib/flask.py")`` was False there and this
    check - which exists to catch the defect that broke v1 - silently failed on Windows
    while claiming to guard it.
    """
    import signledger.contrib.flask as module

    assert Path(module.__file__).parts[-2:] == ("contrib", "flask.py")


# ======================================================================================
# What gets recorded
# ======================================================================================


def test_one_entry_is_appended_per_request(ledger: Ledger) -> None:
    client = make_app(ledger).test_client()
    client.get("/items")
    client.get("/items")
    assert ledger.count() == 2


def test_the_recorded_status_is_the_one_the_client_received(ledger: Ledger) -> None:
    client = make_app(ledger).test_client()
    response = client.get("/nope")
    assert (response.status_code, entries(ledger)[0].data["status"]) == (404, 404)


def test_the_recorded_method_and_path_match_the_request(ledger: Ledger) -> None:
    client = make_app(ledger).test_client()
    client.post("/login", json={"user": "ada"})
    data = entries(ledger)[0].data
    assert (data["method"], data["path"]) == ("POST", "/login")


def test_a_view_that_raises_is_still_audited(ledger: Ledger) -> None:
    """The requests you most need in an audit trail are the ones that went wrong."""
    client = make_app(ledger).test_client()
    assert client.get("/boom").status_code == 500
    assert entries(ledger)[0].data["status"] == 500


def test_the_integration_names_itself_in_the_record(ledger: Ledger) -> None:
    client = make_app(ledger).test_client()
    client.get("/items")
    assert entries(ledger)[0].data["integration"] == "flask"


def test_the_chain_still_verifies_after_a_series_of_audited_requests(
    ledger: Ledger,
) -> None:
    """An audit middleware that corrupts the chain has destroyed the thing it feeds."""
    client = make_app(ledger).test_client()
    for _ in range(5):
        client.get("/items")
    assert ledger.check_integrity().ok


def test_the_resolved_user_is_recorded(ledger: Ledger) -> None:
    app = make_app(ledger, user_resolver=lambda request: "user-42")
    app.test_client().get("/items")
    assert entries(ledger)[0].data["user"] == "user-42"


def test_no_user_is_recorded_when_no_resolver_is_configured(ledger: Ledger) -> None:
    make_app(ledger).test_client().get("/items")
    assert entries(ledger)[0].data["user"] is None


def test_a_user_resolver_returning_a_non_string_is_coerced(ledger: Ledger) -> None:
    """``request.user.pk`` is an integer in most real applications."""
    app = make_app(ledger, user_resolver=lambda request: 42)
    app.test_client().get("/items")
    assert entries(ledger)[0].data["user"] == "42"


# ======================================================================================
# A streamed response is not a settled one
#
# ``after_request`` runs when the response *object* is built, which for a streamed body
# is before a single byte of it has been generated: Flask hands the iterable to the WSGI
# server afterwards. So the status in the entry is a promise, exactly as it is under
# ASGI, and a generator that raised on its second chunk was measurably in the ledger as
# a plain 200.
# ======================================================================================


def test_a_streamed_response_is_recorded_as_provisional(ledger: Ledger) -> None:
    streaming_app(ledger).test_client().get("/stream")
    assert entries(ledger)[0].data["completion"] == "provisional"


def test_a_buffered_response_is_recorded_as_final(ledger: Ledger) -> None:
    """The whole body existed when the entry was written, so the status is settled."""
    make_app(ledger).test_client().get("/items")
    assert entries(ledger)[0].data["completion"] == "final"


def test_a_buffered_response_writes_exactly_one_entry(ledger: Ledger) -> None:
    make_app(ledger).test_client().get("/items")
    assert ledger.count() == 1


@pytest.mark.parametrize(
    ("method", "status"),
    [("HEAD", 200), ("GET", 204), ("GET", 304), ("GET", 100)],
)
def test_a_response_whose_body_is_never_sent_is_not_provisional(
    ledger: Ledger, method: str, status: int
) -> None:
    """Werkzeug drops the body entirely for these, so the iterable is never consumed.

    A provisional entry for one of them would wait forever for a settlement that cannot
    come - which is worse than the overclaim it was meant to fix.
    """
    app = Flask(__name__)
    SignLedgerAudit(app, ledger=ledger)

    @app.route("/x", methods=["GET", "HEAD"])
    def view() -> Response:
        return Response(iter([b"body"]), status=status)

    app.test_client().open("/x", method=method)
    assert (entries(ledger)[0].data["completion"], ledger.count()) == ("final", 1)


def test_a_completed_stream_is_settled_by_a_second_entry(ledger: Ledger) -> None:
    run_wsgi(streaming_app(ledger), "/stream")
    assert kinds(ledger) == ["request", "response.completion"]


def test_a_completed_stream_settles_as_complete(ledger: Ledger) -> None:
    run_wsgi(streaming_app(ledger), "/stream")
    assert entries(ledger)[1].data["outcome"] == "complete"


def test_a_stream_that_dies_mid_body_is_not_left_recorded_as_a_success(
    ledger: Ledger,
) -> None:
    """Measured: the generator raised on its second chunk, the client got a truncated
    body, and the ledger said ``200`` and nothing else, permanently."""
    with pytest.raises(RuntimeError):
        run_wsgi(streaming_app(ledger), "/broken-stream")
    assert entries(ledger)[1].data["outcome"] == "incomplete"


def test_a_stream_that_dies_mid_body_records_the_exception_type(ledger: Ledger) -> None:
    with pytest.raises(RuntimeError):
        run_wsgi(streaming_app(ledger), "/broken-stream")
    assert entries(ledger)[1].data["error"] == "RuntimeError"


def test_the_promised_entry_is_never_rewritten_by_the_settlement(ledger: Ledger) -> None:
    with pytest.raises(RuntimeError):
        run_wsgi(streaming_app(ledger), "/broken-stream")
    data = entries(ledger)[0].data
    assert (data["status"], "error" not in data) == (200, True)


def test_the_settlement_entry_carries_the_request_id_of_the_entry_it_settles(
    ledger: Ledger,
) -> None:
    run_wsgi(streaming_app(ledger), "/stream")
    request_entry, settlement = entries(ledger)
    assert settlement.data["request_id"] == request_entry.data["request_id"]


def test_the_settlement_entry_names_the_request_it_settles(ledger: Ledger) -> None:
    run_wsgi(streaming_app(ledger), "/stream")
    data = entries(ledger)[1].data
    assert (data["method"], data["path"], data["status"]) == ("GET", "/stream", 200)


def test_a_stream_the_server_abandons_early_is_recorded_as_incomplete(
    ledger: Ledger,
) -> None:
    """A client that hangs up: the server closes the iterable without draining it.

    Nothing raised, so no ``error`` is named - inventing one would be its own lie.
    """
    iterable = wsgi_request(streaming_app(ledger), "/stream")
    next(iter(iterable))
    iterable.close()
    data = entries(ledger)[1].data
    assert (data["outcome"], "error" not in data) == ("incomplete", True)


def test_two_requests_do_not_share_a_request_id(ledger: Ledger) -> None:
    client = make_app(ledger).test_client()
    client.get("/items")
    client.get("/items")
    first, second = entries(ledger)
    assert first.data["request_id"] != second.data["request_id"]


def test_the_request_entry_says_what_kind_of_record_it_is(ledger: Ledger) -> None:
    make_app(ledger).test_client().get("/items")
    assert entries(ledger)[0].data["kind"] == "request"


def test_the_underlying_iterator_is_still_closed(ledger: Ledger) -> None:
    """Wrapping the body must not leak the file handle behind a ``send_file``."""
    closed: List[bool] = []
    app = Flask(__name__)
    SignLedgerAudit(app, ledger=ledger)

    def body() -> Iterator[bytes]:
        try:
            yield b'{"row": 0}\n'
        finally:
            closed.append(True)

    @app.get("/x")
    def view() -> Response:
        return Response(body(), mimetype="application/x-ndjson")

    run_wsgi(app, "/x")
    assert closed == [True]


def test_an_iterable_with_no_close_method_is_still_settled(ledger: Ledger) -> None:
    """``Response(iter([...]))`` is legal WSGI and has no ``close``."""
    run_wsgi(streaming_app(ledger), "/plain-iterator")
    assert entries(ledger)[1].data["outcome"] == "complete"


def test_a_file_response_is_settled_too(ledger: Ledger, tmp_path: Any) -> None:
    """``send_file`` returns a streamed ``direct_passthrough`` response, and werkzeug
    skips a response's ``call_on_close`` callbacks entirely for those - it hands the
    server the body iterable directly. Settling from the body itself is what reaches
    this case at all."""
    from flask import send_file

    path = tmp_path / "statement.txt"
    path.write_text("hello file body")
    app = Flask(__name__)
    SignLedgerAudit(app, ledger=ledger)

    @app.get("/file")
    def view() -> Response:
        return send_file(str(path), mimetype="text/plain")

    assert run_wsgi(app, "/file") == [b"hello file body"]
    assert kinds(ledger) == ["request", "response.completion"]


def test_closing_a_response_twice_settles_it_once(ledger: Ledger) -> None:
    """Servers and test clients both close a response more than once - werkzeug's own
    ``ClosingIterator`` closes the body, and the ``Response`` object closes it again. A
    second settlement entry would be a second claim about the same response."""
    iterable = wsgi_request(streaming_app(ledger), "/stream")
    list(iterable)
    iterable.close()
    iterable.close()
    assert kinds(ledger) == ["request", "response.completion"]


def test_a_settlement_that_cannot_be_written_is_counted_not_raised(
    ledger: Ledger,
) -> None:
    """The client has had the response since the first chunk; there is no refusing it
    now. The failure is counted on the recorder rather than raised into a WSGI server
    that is already closing the response down."""
    app = streaming_app(ledger)
    audit = app.extensions["signledger"]

    def close_the_backend() -> Iterator[bytes]:
        yield b'{"row": 0}\n'
        ledger.backend.close()

    @app.get("/breaks-after-the-first-chunk")
    def view() -> Response:
        return Response(close_the_backend(), mimetype="application/x-ndjson")

    assert run_wsgi(app, "/breaks-after-the-first-chunk") == [b'{"row": 0}\n']
    assert (audit.recorder.audit_failures, audit.recorder.last_audit_error) == (
        1,
        "StorageError",
    )


# ======================================================================================
# Body capture
# ======================================================================================


def test_the_body_is_not_recorded_by_default(ledger: Ledger) -> None:
    client = make_app(ledger).test_client()
    client.post("/login", json={"user": "ada"})
    assert entries(ledger)[0].data["body"] == "[OMITTED:disabled]"


def test_an_enabled_body_capture_records_the_redacted_body(ledger: Ledger) -> None:
    app = make_app(ledger, redaction=RedactionPolicy(capture_body=True))
    app.test_client().post("/login", json={"user": "ada", "password": "hunter2"})
    assert entries(ledger)[0].data["body"] == {"user": "ada", "password": REDACTED}


def test_a_body_over_the_limit_is_recorded_as_omitted_not_stored(ledger: Ledger) -> None:
    app = make_app(ledger, redaction=RedactionPolicy(capture_body=True, max_body_bytes=16))
    app.test_client().post("/login", json={"user": "a" * 200})
    assert entries(ledger)[0].data["body"] == "[OMITTED:too-large]"


def test_the_view_can_still_read_a_body_the_middleware_captured(ledger: Ledger) -> None:
    """Reading the body for audit must not consume it out from under the application."""
    app = Flask(__name__)
    SignLedgerAudit(app, ledger=ledger, redaction=RedactionPolicy(capture_body=True))
    seen: Dict[str, Any] = {}

    @app.post("/echo")
    def echo() -> Dict[str, Any]:
        from flask import request

        seen.update(request.get_json())
        return {"ok": True}

    app.test_client().post("/echo", json={"user": "ada"})
    assert seen == {"user": "ada"}


# ======================================================================================
# audit_failure_policy
# ======================================================================================


def test_fail_closed_is_the_default_and_turns_a_write_failure_into_a_500(
    ledger: Ledger,
) -> None:
    """An unaudited request must not be able to look like a successful one."""
    app = make_app(ledger)
    client = app.test_client()
    ledger.backend.close()
    assert client.get("/items").status_code == 500


def test_fail_closed_raises_an_audit_write_error(ledger: Ledger) -> None:
    app = make_app(ledger)
    app.config["PROPAGATE_EXCEPTIONS"] = True
    client = app.test_client()
    ledger.backend.close()
    with pytest.raises(AuditWriteError):
        client.get("/items")


def test_fail_open_still_serves_the_real_response(ledger: Ledger) -> None:
    app = make_app(ledger, audit_failure_policy=FAIL_OPEN)
    client = app.test_client()
    ledger.backend.close()
    assert client.get("/items").status_code == 200


def test_a_fail_open_audit_failure_is_counted_not_swallowed(ledger: Ledger) -> None:
    app = make_app(ledger, audit_failure_policy=FAIL_OPEN)
    audit = app.extensions["signledger"]
    client = app.test_client()
    ledger.backend.close()
    client.get("/items")
    assert audit.recorder.audit_failures == 1


def test_an_unknown_audit_failure_policy_is_rejected_at_construction(
    ledger: Ledger,
) -> None:
    with pytest.raises(ConfigurationError):
        SignLedgerAudit(ledger=ledger, audit_failure_policy="maybe")


# ======================================================================================
# Trusted proxies
# ======================================================================================


def test_x_forwarded_for_is_ignored_when_no_proxies_are_trusted(ledger: Ledger) -> None:
    """Otherwise any client can dictate the IP an auditor will attribute the action to."""
    client = make_app(ledger).test_client()
    client.get("/items", headers={"X-Forwarded-For": "1.2.3.4"}, environ_base={
        "REMOTE_ADDR": "10.0.0.9"
    })
    assert entries(ledger)[0].data["client_ip"] == "10.0.0.9"


def test_the_client_ip_is_taken_from_the_trusted_position_in_the_chain(
    ledger: Ledger,
) -> None:
    app = make_app(ledger, redaction=RedactionPolicy(trusted_proxy_count=1))
    app.test_client().get(
        "/items",
        headers={"X-Forwarded-For": "evil, 203.0.113.7"},
        environ_base={"REMOTE_ADDR": "10.0.0.9"},
    )
    assert entries(ledger)[0].data["client_ip"] == "203.0.113.7"


# ======================================================================================
# skip_paths
# ======================================================================================


def test_a_skipped_path_writes_no_entry(ledger: Ledger) -> None:
    """A health check hit every second would otherwise bury the audit trail."""
    app = make_app(ledger, skip_paths=("/health",))
    app.test_client().get("/health")
    assert ledger.count() == 0


def test_a_path_that_is_not_skipped_still_writes(ledger: Ledger) -> None:
    app = make_app(ledger, skip_paths=("/health",))
    app.test_client().get("/items")
    assert ledger.count() == 1


# ======================================================================================
# The audit-inspection endpoints
# ======================================================================================


def inspect_app(ledger: Ledger, **kwargs: Any) -> Flask:
    app = Flask(__name__)
    app.config["PROPAGATE_EXCEPTIONS"] = False
    app.register_blueprint(audit_blueprint(ledger, **kwargs))
    return app


def token_auth(request: Any) -> bool:
    return bool(request.headers.get("X-Audit-Token") == "let-me-in")


def test_the_audit_endpoints_cannot_be_registered_without_an_authenticator(
    ledger: Ledger,
) -> None:
    """Unauthenticated must not be reachable by omission."""
    with pytest.raises(ConfigurationError):
        audit_blueprint(ledger, authenticator=None)


def test_an_unauthenticated_status_request_is_rejected(ledger: Ledger) -> None:
    client = inspect_app(ledger, authenticator=token_auth).test_client()
    assert client.get("/_audit/status").status_code == 401


def test_an_unauthenticated_entries_request_is_rejected(ledger: Ledger) -> None:
    """This is the full-ledger scan: DoS and information disclosure in one request."""
    client = inspect_app(ledger, authenticator=token_auth).test_client()
    assert client.get("/_audit/entries").status_code == 401


def test_an_unauthenticated_verify_request_is_rejected(ledger: Ledger) -> None:
    client = inspect_app(ledger, authenticator=token_auth).test_client()
    assert client.get("/_audit/verify").status_code == 401


def test_a_rejected_request_returns_no_ledger_content(ledger: Ledger) -> None:
    ledger.append({"secret_business_fact": "acquisition"})
    client = inspect_app(ledger, authenticator=token_auth).test_client()
    assert b"acquisition" not in client.get("/_audit/entries").data


def test_an_authenticated_status_request_reports_the_real_entry_count(
    ledger: Ledger,
) -> None:
    ledger.append({"a": 1})
    client = inspect_app(ledger, authenticator=token_auth).test_client()
    response = client.get("/_audit/status", headers={"X-Audit-Token": "let-me-in"})
    assert response.get_json()["entry_count"] == 1


def test_the_entries_endpoint_caps_the_page_size(ledger: Ledger) -> None:
    """An authenticated caller must not be able to ask for the whole ledger either."""
    for index in range(10):
        ledger.append({"n": index})
    client = inspect_app(ledger, authenticator=token_auth, max_page_size=3).test_client()
    response = client.get(
        "/_audit/entries?limit=1000", headers={"X-Audit-Token": "let-me-in"}
    )
    assert len(response.get_json()["entries"]) == 3


def test_the_entries_endpoint_defaults_to_the_capped_page_size(ledger: Ledger) -> None:
    for index in range(10):
        ledger.append({"n": index})
    client = inspect_app(ledger, authenticator=token_auth, max_page_size=2).test_client()
    response = client.get("/_audit/entries", headers={"X-Audit-Token": "let-me-in"})
    assert len(response.get_json()["entries"]) == 2


def test_the_entries_endpoint_paginates_with_an_offset(ledger: Ledger) -> None:
    for index in range(5):
        ledger.append({"n": index})
    client = inspect_app(ledger, authenticator=token_auth).test_client()
    response = client.get(
        "/_audit/entries?offset=3&limit=2", headers={"X-Audit-Token": "let-me-in"}
    )
    assert [row["sequence"] for row in response.get_json()["entries"]] == [3, 4]


def test_the_entries_endpoint_rejects_a_non_numeric_limit(ledger: Ledger) -> None:
    client = inspect_app(ledger, authenticator=token_auth).test_client()
    response = client.get("/_audit/entries?limit=all", headers={"X-Audit-Token": "let-me-in"})
    assert response.status_code == 400


def test_the_entries_endpoint_rejects_a_negative_offset(ledger: Ledger) -> None:
    client = inspect_app(ledger, authenticator=token_auth).test_client()
    response = client.get("/_audit/entries?offset=-1", headers={"X-Audit-Token": "let-me-in"})
    assert response.status_code == 400


def test_the_entries_endpoint_rejects_a_zero_limit(ledger: Ledger) -> None:
    client = inspect_app(ledger, authenticator=token_auth).test_client()
    response = client.get("/_audit/entries?limit=0", headers={"X-Audit-Token": "let-me-in"})
    assert response.status_code == 400


def test_the_entries_endpoint_never_returns_entry_payloads(ledger: Ledger) -> None:
    """Identifiers and digests are enough to audit; payloads are the disclosure risk.

    The sentinel deliberately contains non-hex letters: ``entry_id`` is a uuid4 hex
    string, so a value like ``"ada"`` can turn up inside one by chance and this
    assertion would fail at random.
    """
    ledger.append({"patient_name": "zsofia-quinn"})
    client = inspect_app(ledger, authenticator=token_auth).test_client()
    response = client.get("/_audit/entries", headers={"X-Audit-Token": "let-me-in"})
    assert b"zsofia-quinn" not in response.data


def test_the_entries_endpoint_returns_only_the_safe_field_set(ledger: Ledger) -> None:
    """Structural, so it cannot pass by accident: no ``data``, no ``metadata``."""
    ledger.append({"patient_name": "zsofia-quinn"})
    client = inspect_app(ledger, authenticator=token_auth).test_client()
    response = client.get("/_audit/entries", headers={"X-Audit-Token": "let-me-in"})
    assert set(response.get_json()["entries"][0]) == {
        "sequence",
        "entry_id",
        "timestamp",
        "hash",
        "previous_hash",
        "key_id",
    }


def test_the_entries_endpoint_returns_the_sequence_and_digest(ledger: Ledger) -> None:
    entry = ledger.append({"a": 1})
    client = inspect_app(ledger, authenticator=token_auth).test_client()
    response = client.get("/_audit/entries", headers={"X-Audit-Token": "let-me-in"})
    row = response.get_json()["entries"][0]
    assert (row["sequence"], row["hash"]) == (0, entry.hash)


def test_the_verify_endpoint_reports_a_healthy_chain(ledger: Ledger) -> None:
    ledger.append({"a": 1})
    client = inspect_app(ledger, authenticator=token_auth).test_client()
    response = client.get("/_audit/verify", headers={"X-Audit-Token": "let-me-in"})
    assert response.get_json()["ok"] is True


def test_the_verify_endpoint_reports_an_empty_ledger_as_not_ok(ledger: Ledger) -> None:
    """Contract section 5 rule 1. v1's CLI printed a success tick here."""
    client = inspect_app(ledger, authenticator=token_auth).test_client()
    response = client.get("/_audit/verify", headers={"X-Audit-Token": "let-me-in"})
    assert response.get_json()["ok"] is False


def test_the_verify_endpoint_names_the_assurance_level(ledger: Ledger) -> None:
    """CHAIN_ONLY must never be rendered to an operator as a green tick."""
    ledger.append({"a": 1})
    client = inspect_app(ledger, authenticator=token_auth).test_client()
    response = client.get("/_audit/verify", headers={"X-Audit-Token": "let-me-in"})
    assert response.get_json()["assurance"] == "chain_only"


def test_the_audit_endpoints_can_be_mounted_under_a_different_prefix(
    ledger: Ledger,
) -> None:
    client = inspect_app(
        ledger, authenticator=token_auth, url_prefix="/internal/audit"
    ).test_client()
    response = client.get("/internal/audit/status", headers={"X-Audit-Token": "let-me-in"})
    assert response.status_code == 200


def test_a_zero_max_page_size_is_rejected_at_registration(ledger: Ledger) -> None:
    with pytest.raises(ConfigurationError):
        audit_blueprint(ledger, authenticator=token_auth, max_page_size=0)


# ======================================================================================
# Extension wiring
# ======================================================================================


def test_the_extension_can_be_initialised_after_construction(ledger: Ledger) -> None:
    """The Flask ``init_app`` pattern, for applications built by a factory."""
    audit = SignLedgerAudit(ledger=ledger)
    app = Flask(__name__)

    @app.get("/x")
    def view() -> Dict[str, Any]:
        return {"ok": True}

    audit.init_app(app)
    app.test_client().get("/x")
    assert ledger.count() == 1


def test_the_extension_registers_itself_under_a_known_key(ledger: Ledger) -> None:
    app = make_app(ledger)
    assert isinstance(app.extensions["signledger"], SignLedgerAudit)


def test_a_recorder_without_an_explicit_redaction_policy_still_redacts(
    ledger: Ledger,
) -> None:
    app = SignLedgerAudit(ledger=ledger)
    assert isinstance(app.recorder.redaction, RedactionPolicy)

