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
from typing import Any, Dict, Iterator, List

import pytest
from flask import Flask

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
    assert (query["access_token"], query["page"]) == (REDACTED, ["2"])


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
    """v1 shipped four empty directories that shadowed their own sibling modules."""
    import signledger.contrib.flask as module

    assert module.__file__.endswith("contrib/flask.py")


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

