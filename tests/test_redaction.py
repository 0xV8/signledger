"""Tests for :mod:`signledger.contrib.redaction`.

Redaction is the reason this package exists. v1's FastAPI middleware wrote the
``Authorization`` header, every cookie and the full request body straight into an
*immutable* ledger, in a library sold for HIPAA and PCI work. An immutable store is the
worst possible place to put a secret: you cannot delete it, and every later entry's hash
depends on it, so you cannot even rewrite it.

So the tests here are not "does the helper work" tests. They are "can this value reach
the ledger" tests, and the answer has to be no by default.
"""

from typing import Iterator

import pytest

from signledger.backends.memory import MemoryBackend
from signledger.contrib import (
    FAIL_CLOSED,
    FAIL_OPEN,
    AuditRecorder,
    AuditWriteError,
    build_audit_record,
)
from signledger.contrib.redaction import (
    MAX_REDACTION_DEPTH,
    OVERSIZED,
    REDACTED,
    TRUNCATED,
    UNPARSEABLE,
    BodyDecision,
    RedactionPolicy,
    redact_headers,
    redact_payload,
)
from signledger.core.exceptions import ConfigurationError, SignLedgerError
from signledger.core.ledger import Ledger


def test_authorization_header_value_is_replaced_with_the_marker() -> None:
    assert redact_headers({"Authorization": "Bearer sk-live-51H8xQ"})["authorization"] == REDACTED


def test_a_header_whose_name_merely_contains_secret_is_redacted() -> None:
    """The denylist is a floor, not a ceiling.

    Nobody can enumerate every header a deployment invents, so a name that *reads* as
    sensitive is treated as sensitive.
    """
    assert redact_headers({"X-Client-Secret": "hunter2"})["x-client-secret"] == REDACTED


def test_a_password_nested_three_levels_deep_is_redacted() -> None:
    """Depth is not a hiding place.

    A top-level-only scrub is worse than none: it reads as protection while every real
    API, which nests its payloads, keeps leaking.
    """
    payload = {"a": {"b": {"user": {"name": "ada", "password": "hunter2"}}}}
    assert redact_payload(payload)["a"]["b"]["user"]["password"] == REDACTED


def test_a_key_that_merely_contains_card_is_redacted() -> None:
    """Real payloads never use the bare word. They use ``card_number``."""
    assert redact_payload({"card_number": "4111111111111111"})["card_number"] == REDACTED


def test_a_sensitive_field_inside_a_list_of_objects_is_redacted() -> None:
    """A bulk-create endpoint posts a list, and every element of it is a payload."""
    users = {"users": [{"name": "ada"}, {"name": "bob", "ssn": "078-05-1120"}]}
    assert redact_payload(users)["users"][1]["ssn"] == REDACTED


def test_redaction_does_not_mutate_the_payload_it_was_given() -> None:
    """The caller still owns its request body; scrubbing our copy must not scrub theirs."""
    original = {"password": "hunter2"}
    redact_payload(original)
    assert original["password"] == "hunter2"


def test_a_payload_nested_past_the_depth_limit_is_truncated_not_recursed() -> None:
    """Depth is bounded before recursing, so a hostile body cannot blow the stack."""
    payload: dict = {"leaf": 1}
    for _ in range(MAX_REDACTION_DEPTH + 5):
        payload = {"n": payload}

    result = redact_payload(payload)
    for _ in range(MAX_REDACTION_DEPTH):
        result = result["n"]
    assert result == TRUNCATED


def test_a_deeply_nested_hostile_payload_does_not_raise_recursionerror() -> None:
    payload: dict = {"leaf": 1}
    for _ in range(20_000):
        payload = {"n": payload}
    redact_payload(payload)


def test_a_non_finite_float_becomes_a_string_the_ledger_can_store() -> None:
    """``json.loads('{"x": NaN}')`` succeeds, and SLCS/1 refuses NaN.

    Without this, any client could kill an audited endpoint by posting ``NaN``: the
    canonicalizer rejects it, the audit write fails, and fail-closed turns that into a
    denial of service.
    """
    assert redact_payload({"x": float("nan")})["x"] == "NaN"


def test_an_integer_beyond_the_canonical_range_becomes_its_decimal_string() -> None:
    """Exactly the conversion the canonicalizer's own error message recommends."""
    assert redact_payload({"n": 2**53 + 1})["n"] == "9007199254740993"


def test_an_integer_on_the_canonical_boundary_is_left_as_an_integer() -> None:
    assert redact_payload({"n": 2**53})["n"] == 2**53


def test_a_value_of_a_type_json_cannot_represent_becomes_a_type_marker() -> None:
    """Bytes cannot be canonicalized and their content is exactly what must not leak."""
    assert redact_payload({"blob": b"secret-bytes"})["blob"] == "[UNSUPPORTED:bytes]"


def test_a_non_string_mapping_key_becomes_a_string_key() -> None:
    """SLCS/1 rejects non-str keys, and v1 collided ``{1: ...}`` with ``{"1": ...}``."""
    assert redact_payload({1: "a"}) == {"1": "a"}


def test_a_redacted_payload_survives_canonicalization() -> None:
    """The whole point: whatever redaction emits, the ledger can actually hash."""
    from signledger.crypto.canonical import canonicalize

    hostile = {
        "nan": float("inf"),
        "big": 2**80,
        "blob": b"x",
        2: {"password": "p"},
        "list": [float("nan"), b"y"],
    }
    canonicalize(redact_payload(hostile), allow_datetime=False)


def test_a_float_that_is_finite_is_left_alone() -> None:
    assert redact_payload({"x": 1.5})["x"] == 1.5



# ======================================================================================
# RedactionPolicy: body capture
# ======================================================================================


def test_body_capture_is_off_by_default() -> None:
    """The safe default is the one a hurried developer gets without reading the docs."""
    assert RedactionPolicy().body_decision(content_length=10, content_type="application/json") == (
        BodyDecision(capture=False, reason="disabled")
    )


def test_a_body_is_refused_when_no_content_length_was_declared() -> None:
    """Length is enforced *before* reading, so an undeclared length is an unbounded read."""
    policy = RedactionPolicy(capture_body=True)
    decision = policy.body_decision(content_length=None, content_type="application/json")
    assert decision == BodyDecision(capture=False, reason="no-content-length")


def test_a_body_larger_than_the_limit_is_refused_before_it_is_read() -> None:
    policy = RedactionPolicy(capture_body=True, max_body_bytes=100)
    decision = policy.body_decision(content_length=101, content_type="application/json")
    assert decision == BodyDecision(capture=False, reason="too-large")


def test_a_body_exactly_on_the_limit_is_captured() -> None:
    policy = RedactionPolicy(capture_body=True, max_body_bytes=100)
    assert policy.body_decision(content_length=100, content_type="application/json").capture


def test_a_non_json_body_is_not_captured() -> None:
    """A urlencoded login form is exactly the body nobody should be storing."""
    policy = RedactionPolicy(capture_body=True)
    decision = policy.body_decision(
        content_length=10, content_type="application/x-www-form-urlencoded"
    )
    assert decision == BodyDecision(capture=False, reason="unsupported-content-type")


def test_a_json_content_type_with_a_charset_parameter_is_still_json() -> None:
    policy = RedactionPolicy(capture_body=True)
    assert policy.body_decision(
        content_length=10, content_type="application/json; charset=utf-8"
    ).capture


def test_a_missing_content_type_is_not_captured() -> None:
    policy = RedactionPolicy(capture_body=True)
    decision = policy.body_decision(content_length=10, content_type=None)
    assert decision == BodyDecision(capture=False, reason="unsupported-content-type")


def test_a_captured_body_is_parsed_and_redacted() -> None:
    policy = RedactionPolicy(capture_body=True)
    assert policy.decode_body(b'{"password": "hunter2", "n": 1}') == {
        "password": REDACTED,
        "n": 1,
    }


def test_a_malformed_json_body_is_marked_rather_than_raised() -> None:
    """A client that posts broken JSON must not be able to 500 an audited endpoint."""
    assert RedactionPolicy(capture_body=True).decode_body(b"{not json") == UNPARSEABLE


def test_a_body_that_is_not_utf8_is_marked_rather_than_raised() -> None:
    assert RedactionPolicy(capture_body=True).decode_body(b"\xff\xfe") == UNPARSEABLE


def test_a_body_longer_than_the_limit_is_refused_even_if_content_length_lied() -> None:
    """Content-Length is a claim by the client, not a fact."""
    policy = RedactionPolicy(capture_body=True, max_body_bytes=10)
    assert policy.decode_body(b'{"a": "0123456789012345"}') == OVERSIZED


def test_a_body_capture_limit_below_one_byte_is_rejected() -> None:
    with pytest.raises(ConfigurationError):
        RedactionPolicy(capture_body=True, max_body_bytes=0)


# ======================================================================================
# RedactionPolicy: client identity
# ======================================================================================


def test_x_forwarded_for_is_ignored_when_no_proxies_are_trusted() -> None:
    """Without a declared proxy count the header is attacker-controlled text."""
    policy = RedactionPolicy()
    assert policy.client_ip(peer="10.0.0.9", forwarded_for="1.2.3.4") == "10.0.0.9"


def test_the_client_ip_is_read_from_the_configured_position_in_the_chain() -> None:
    """With one trusted proxy the client is the last entry that proxy appended."""
    policy = RedactionPolicy(trusted_proxy_count=1)
    forwarded = "9.9.9.9, 203.0.113.7"
    assert policy.client_ip(peer="10.0.0.9", forwarded_for=forwarded) == "203.0.113.7"


def test_entries_a_spoofer_prepended_are_never_selected() -> None:
    policy = RedactionPolicy(trusted_proxy_count=2)
    forwarded = "evil, 203.0.113.7, 10.0.0.1"
    assert policy.client_ip(peer="10.0.0.9", forwarded_for=forwarded) == "203.0.113.7"


def test_a_chain_shorter_than_the_trusted_proxy_count_falls_back_to_the_peer() -> None:
    """The deployment is not the one described, so the header proves nothing."""
    policy = RedactionPolicy(trusted_proxy_count=3)
    assert policy.client_ip(peer="10.0.0.9", forwarded_for="1.2.3.4") == "10.0.0.9"


def test_an_absent_forwarded_header_falls_back_to_the_peer() -> None:
    policy = RedactionPolicy(trusted_proxy_count=1)
    assert policy.client_ip(peer="10.0.0.9", forwarded_for=None) == "10.0.0.9"


def test_a_forwarded_header_of_only_separators_falls_back_to_the_peer() -> None:
    policy = RedactionPolicy(trusted_proxy_count=1)
    assert policy.client_ip(peer="10.0.0.9", forwarded_for=" , ") == "10.0.0.9"


def test_a_negative_trusted_proxy_count_is_rejected_at_construction() -> None:
    with pytest.raises(ConfigurationError):
        RedactionPolicy(trusted_proxy_count=-1)


def test_an_unknown_peer_with_no_usable_header_is_recorded_as_none() -> None:
    assert RedactionPolicy().client_ip(peer=None, forwarded_for=None) is None


# ======================================================================================
# RedactionPolicy: configuration
# ======================================================================================


def test_extra_sensitive_fields_are_scrubbed_alongside_the_built_in_ones() -> None:
    policy = RedactionPolicy(extra_sensitive_fields=["patient_name"])
    assert policy.payload({"patient_name": "ada"})["patient_name"] == REDACTED


def test_extra_sensitive_fields_cannot_switch_off_the_built_in_ones() -> None:
    policy = RedactionPolicy(extra_sensitive_fields=["patient_name"])
    assert policy.payload({"password": "hunter2"})["password"] == REDACTED


def test_the_header_denylist_is_applied_through_the_policy_too() -> None:
    assert RedactionPolicy().headers({"Cookie": "session=abc"})["cookie"] == REDACTED


def test_a_tuple_becomes_a_json_array() -> None:
    """SLCS/1 refuses a tuple outright; JSON has one sequence type and this is it."""
    assert redact_payload({"pair": (1, 2)})["pair"] == [1, 2]


def test_negative_infinity_keeps_its_sign() -> None:
    """A temperature of -inf and one of +inf are not the same audit record."""
    assert redact_payload({"x": float("-inf")})["x"] == "-Infinity"


def test_positive_infinity_is_recorded_as_infinity() -> None:
    assert redact_payload({"x": float("inf")})["x"] == "Infinity"


def test_an_integer_below_the_negative_canonical_bound_becomes_a_string() -> None:
    assert redact_payload({"n": -(2**53) - 1})["n"] == "-9007199254740993"


def test_booleans_are_preserved_as_booleans_not_coerced_to_integers() -> None:
    assert redact_payload({"ok": True})["ok"] is True


def test_none_is_preserved() -> None:
    assert redact_payload({"x": None})["x"] is None


def test_extra_sensitive_field_names_are_matched_case_insensitively() -> None:
    policy = RedactionPolicy(extra_sensitive_fields=["Patient_Name"])
    assert policy.payload({"PATIENT_NAME": "ada"})["PATIENT_NAME"] == REDACTED


# ======================================================================================
# signledger.contrib: the shared audit-failure policy
#
# These live here rather than in a file of their own because ``contrib/__init__.py`` is
# the shared half of all three integrations and my brief lists four test modules only.
# ======================================================================================


@pytest.fixture
def ledger() -> Iterator[Ledger]:
    with Ledger(backend=MemoryBackend(), ledger_id="test-ledger") as led:
        yield led


def test_the_default_audit_failure_policy_is_fail_closed() -> None:
    """An audit trail you are allowed to skip is not an audit trail."""
    assert AuditRecorder(ledger=None).audit_failure_policy == FAIL_CLOSED


def test_a_successful_record_returns_the_stored_entry(ledger: Ledger) -> None:
    entry = AuditRecorder(ledger=ledger).record({"method": "GET"})
    assert entry is not None and entry.sequence == 0


def test_fail_closed_turns_a_failed_ledger_write_into_an_error(ledger: Ledger) -> None:
    """The write failed, so the request must not be allowed to look successful."""
    recorder = AuditRecorder(ledger=ledger, audit_failure_policy=FAIL_CLOSED)
    with pytest.raises(AuditWriteError):
        recorder.record({})


def test_fail_open_returns_none_instead_of_raising(ledger: Ledger) -> None:
    recorder = AuditRecorder(ledger=ledger, audit_failure_policy=FAIL_OPEN)
    assert recorder.record({}) is None


def test_a_fail_open_failure_is_counted_rather_than_silently_swallowed(
    ledger: Ledger,
) -> None:
    """v1's whole failure mode was a swallowed exception. A dropped audit record must
    still be visible to whoever runs the service."""
    recorder = AuditRecorder(ledger=ledger, audit_failure_policy=FAIL_OPEN)
    recorder.record({})
    assert (recorder.audit_failures, recorder.last_audit_error) == (1, "ValidationError")


def test_fail_open_also_survives_an_error_the_library_did_not_raise(
    ledger: Ledger,
) -> None:
    """A pydantic error is not a SignLedgerError. Availability was the whole promise."""
    recorder = AuditRecorder(ledger=ledger, audit_failure_policy=FAIL_OPEN)
    assert recorder.record({"a": 1}, metadata="not-a-dict") is None


def test_the_underlying_failure_is_chained_onto_the_audit_write_error(
    ledger: Ledger,
) -> None:
    """An operator needs to know *why* the ledger refused, not just that it did."""
    recorder = AuditRecorder(ledger=ledger)
    with pytest.raises(AuditWriteError) as caught:
        recorder.record({})
    assert isinstance(caught.value.__cause__, SignLedgerError)


def test_the_audit_write_error_message_carries_no_payload(ledger: Ledger) -> None:
    """Contract section 10: no exception message may quote user data."""
    recorder = AuditRecorder(ledger=ledger)
    with pytest.raises(AuditWriteError) as caught:
        # An un-redacted payload that the canonicalizer refuses: the failure path has the
        # secret in scope, which is precisely when a message must not quote it.
        recorder.record({"password": "hunter2", "blob": object()})
    assert "hunter2" not in str(caught.value)


def test_audit_write_error_is_catchable_as_a_signledger_error() -> None:
    assert issubclass(AuditWriteError, SignLedgerError)


def test_an_unknown_audit_failure_policy_is_rejected_at_construction() -> None:
    """Not on the first failure, months later, in production."""
    with pytest.raises(ConfigurationError):
        AuditRecorder(ledger=None, audit_failure_policy="fail_sometimes")


def test_a_recorder_without_a_redaction_policy_still_redacts(ledger: Ledger) -> None:
    """Redaction is not something you can forget to switch on."""
    recorder = AuditRecorder(ledger=ledger)
    entry = recorder.record(recorder.redaction.payload({"password": "hunter2"}))
    assert entry is not None and entry.data["password"] == REDACTED


# ======================================================================================
# signledger.contrib: the shared audit record shape
# ======================================================================================


def test_an_uncaptured_body_is_recorded_as_a_marker_naming_the_reason() -> None:
    """An absent field cannot be told apart from a middleware nobody installed."""
    record = build_audit_record(
        integration="flask",
        method="POST",
        path="/pay",
        status=200,
        client_ip=None,
        headers={},
        body_reason="too-large",
    )
    assert record["body"] == "[OMITTED:too-large]"


def test_a_captured_body_is_recorded_under_the_same_key() -> None:
    record = build_audit_record(
        integration="flask",
        method="POST",
        path="/pay",
        status=200,
        client_ip=None,
        headers={},
        body={"amount": 10},
        body_reason="captured",
    )
    assert record["body"] == {"amount": 10}


def test_the_audit_record_is_never_empty_so_the_ledger_always_accepts_it() -> None:
    """``Entry`` rejects an empty ``data``; an audit record must never trip that."""
    record = build_audit_record(
        integration="flask",
        method="GET",
        path="/",
        status=200,
        client_ip=None,
        headers={},
    )
    assert record


def test_the_audit_record_can_be_canonicalized(ledger: Ledger) -> None:
    """Whatever the record holds, the ledger has to be able to hash it."""
    record = build_audit_record(
        integration="flask",
        method="GET",
        path="/",
        status=200,
        client_ip="1.2.3.4",
        headers={"authorization": REDACTED},
        query={"q": "x"},
        body={"a": 1},
        body_reason="captured",
        user="user-1",
        error="ValueError",
    )
    assert ledger.append(record).data["error"] == "ValueError"


def test_an_error_free_request_records_no_error_field() -> None:
    """A field that is always present and always null trains people to ignore it."""
    record = build_audit_record(
        integration="flask", method="GET", path="/", status=200, client_ip=None, headers={}
    )
    assert "error" not in record


def test_a_query_parameter_named_api_key_is_redacted() -> None:
    """``?api_key=`` is a credential in transit, exactly like a header.

    The payload field list alone would miss it - it has no ``key`` entry - so query
    parameter names are checked against the header rule as well.
    """
    assert RedactionPolicy().query({"api_key": ["sk-live-1"]})["api_key"] == REDACTED


def test_a_query_parameter_named_token_is_redacted() -> None:
    assert RedactionPolicy().query({"reset_token": ["abc"]})["reset_token"] == REDACTED


def test_an_ordinary_query_parameter_is_kept() -> None:
    assert RedactionPolicy().query({"page": ["2"]})["page"] == ["2"]


def test_query_parameter_names_keep_their_case() -> None:
    """Unlike headers, query parameters are case-sensitive and ``?Page`` is not ``?page``."""
    assert "Page" in RedactionPolicy().query({"Page": ["2"]})
