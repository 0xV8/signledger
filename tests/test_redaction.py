"""Tests for :mod:`signledger.contrib.redaction`.

Redaction is the reason this package exists. v1's FastAPI middleware wrote the
``Authorization`` header, every cookie and the full request body straight into an
*immutable* ledger, in a library sold for HIPAA and PCI work. An immutable store is the
worst possible place to put a secret: you cannot delete it, and every later entry's hash
depends on it, so you cannot even rewrite it.

So the tests here are not "does the helper work" tests. They are "can this value reach
the ledger" tests, and the answer has to be no by default.
"""

import json
import os
import socket
import time
import uuid
from typing import Any, Iterator

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
    SENSITIVE_COMPOUNDS,
    SENSITIVE_FIELDS,
    SENSITIVE_SUBSTRINGS,
    SENSITIVE_WORDS,
    TRUNCATED,
    UNENCODABLE,
    UNPARSEABLE,
    BodyDecision,
    RedactionPolicy,
    is_sensitive_field,
    is_sensitive_header,
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


def test_a_list_nested_past_the_depth_limit_is_truncated_not_recursed() -> None:
    """The list branch is bounded by the same counter the dict branch is.

    Nesting is not a dict-only shape: ``[[[[...]]]]`` is valid JSON and arrives from the
    same client-written body. A depth counter that only advances through mappings reads
    as a bound while imposing none, so the list branch gets its own proof.

    ``redact_payload`` is entered on the wrapping dict, which spends depth 0, so the list
    at the top of ``root`` is already at depth 1 and one fewer step is needed to reach the
    marker than in the dict case above.
    """
    payload: Any = {"leaf": 1}
    for _ in range(MAX_REDACTION_DEPTH + 5):
        payload = [payload]

    result = redact_payload({"root": payload})["root"]
    for _ in range(MAX_REDACTION_DEPTH - 1):
        result = result[0]
    assert result == TRUNCATED


def test_a_deeply_nested_hostile_list_does_not_raise_recursionerror() -> None:
    """20,000 nested lists cost a CPython stack, not an interpreter one.

    The dict version of this test is the one that has always been here; a body made of
    lists is the cheaper attack to write and has to be just as harmless.
    """
    payload: Any = {"leaf": 1}
    for _ in range(20_000):
        payload = [payload]
    redact_payload({"root": payload})


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


def test_a_body_exactly_on_the_limit_is_decoded_not_marked_oversized() -> None:
    """``decode_body`` re-checks the same limit ``body_decision`` applied, inclusively.

    ``body_decision`` admits a body of exactly ``max_body_bytes``; if the second check
    were the stricter one, every request that sat precisely on a deployment's configured
    ceiling would be admitted and then thrown away as ``[OVERSIZED]``. Two checks against
    one limit have to agree on the limit.
    """
    policy = RedactionPolicy(capture_body=True, max_body_bytes=13)
    raw = b'{"a": "0123"}'
    assert len(raw) == policy.max_body_bytes
    assert policy.body_decision(content_length=len(raw), content_type="application/json").capture
    assert policy.decode_body(raw) == {"a": "0123"}


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


def test_a_user_carrying_a_lone_surrogate_cannot_break_the_audit_write() -> None:
    """The other half of the unpaired-surrogate DoS.

    ``user`` is whatever a deployment's resolver returned, and real resolvers read it out
    of the request body, so it is attacker-controlled text that never passed through
    redaction on its way here. Before this, one anonymous POST 500ed every audited
    endpoint under ``fail_closed``.
    """
    record = build_audit_record(
        integration="flask",
        method="GET",
        path="/",
        status=401,
        client_ip=None,
        headers={},
        user="\ud800",
    )
    from signledger.crypto.canonical import canonicalize

    assert record["user"] == UNENCODABLE
    canonicalize(record, allow_datetime=False)


def test_a_path_or_error_carrying_a_lone_surrogate_is_sanitised_too() -> None:
    """``path``, ``method``, ``integration`` and ``error`` reach the record by the same
    unsanitised route ``user`` does."""
    record = build_audit_record(
        integration="\ud800",
        method="\ud800",
        path="/\ud800",
        status=500,
        client_ip=None,
        headers={},
        error="\ud800",
    )
    assert (record["integration"], record["method"], record["path"], record["error"]) == (
        UNENCODABLE,
        UNENCODABLE,
        UNENCODABLE,
        UNENCODABLE,
    )
    from signledger.crypto.canonical import canonicalize

    canonicalize(record, allow_datetime=False)


def test_an_absent_user_stays_absent_rather_than_becoming_a_marker() -> None:
    """``None`` means "nobody was resolved", which is not the same as unencodable text."""
    record = build_audit_record(
        integration="flask",
        method="GET",
        path="/",
        status=200,
        client_ip=None,
        headers={},
    )
    assert record["user"] is None


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
    """``?api_key=`` is a credential in transit, exactly like a header."""
    assert RedactionPolicy().query({"api_key": ["sk-live-1"]})["api_key"] == [REDACTED]


def test_a_query_parameter_named_token_is_redacted() -> None:
    assert RedactionPolicy().query({"reset_token": ["abc"]})["reset_token"] == [REDACTED]


def test_an_ordinary_query_parameter_is_kept() -> None:
    assert RedactionPolicy().query({"page": ["2"]})["page"] == ["2"]


def test_query_parameter_names_keep_their_case() -> None:
    """Unlike headers, query parameters are case-sensitive and ``?Page`` is not ``?page``."""
    assert "Page" in RedactionPolicy().query({"Page": ["2"]})


# ======================================================================================
# The published policy set
# ======================================================================================


def test_the_published_policy_set_is_the_union_of_the_three_styles() -> None:
    """``SENSITIVE_FIELDS`` is what a deployment prints to read the policy it is under.

    It is documentation of the three matching sets, so it has to be their *union*: any
    other combining operator still yields a frozenset, still imports, still type-checks,
    and quietly publishes a policy the matcher is not running. The intersection of the
    three is empty - a printed policy of nothing at all, next to a matcher that redacts
    53 stems.

    The count is asserted alongside the union because the union alone is satisfied by
    three empty sets, and because shrinking the published policy should be a deliberate,
    visible edit rather than a silent one.
    """
    assert SENSITIVE_FIELDS == SENSITIVE_SUBSTRINGS | SENSITIVE_COMPOUNDS | SENSITIVE_WORDS
    assert len(SENSITIVE_FIELDS) == 53
    assert SENSITIVE_SUBSTRINGS <= SENSITIVE_FIELDS
    assert SENSITIVE_COMPOUNDS <= SENSITIVE_FIELDS
    assert SENSITIVE_WORDS <= SENSITIVE_FIELDS


# ======================================================================================
# B1 - one name rule, not two
#
# Two rules shipped: a regex (``token|secret|password|key``) guarding headers and query
# strings, and a shorter word list guarding *bodies*. So ``?api_key=`` was redacted and
# ``{"api_key": ...}`` was notarised in the clear - the same credential, protected on one
# path and permanently recorded on the other. These tests pin the two paths together.
# ======================================================================================


#: Credential names a real client actually sends. Every one of these was returned
#: verbatim by ``policy.payload`` before the rules were unified.
CREDENTIAL_NAMES = [
    "api_key",
    "apikey",
    "access_key",
    "private_key",
    "secret_key",
    "signing_key",
    "pwd",
    "passwd",
    "user_pwd",
    "passphrase",
    "session_id",
    "sessionid",
    "session_token",
    "auth",
    "authorization",
    "basic_auth",
    "oauth_token",
    "credentials",
    "cookie",
    "set-cookie",
    "password",
    "password_confirmation",
    "ssn",
    "card_number",
    "cvv",
    "client_secret",
    "reset_token",
]


@pytest.mark.parametrize("name", CREDENTIAL_NAMES)
def test_a_credential_name_is_redacted_in_the_body(name: str) -> None:
    """The body is where the credentials actually are.

    A login POST carries the password in the body, never in the query string, and this
    library's whole reason to exist is that an append-only store cannot be cleaned up
    afterwards.
    """
    assert RedactionPolicy().payload({name: "SECRET"})[name] == REDACTED


@pytest.mark.parametrize("name", CREDENTIAL_NAMES)
def test_the_body_rule_and_the_header_rule_are_the_same_rule(name: str) -> None:
    """One rule, or an operator has to know which transport a value arrived on."""
    assert is_sensitive_field(name) is is_sensitive_header(name) is True


@pytest.mark.parametrize("name", CREDENTIAL_NAMES)
def test_a_credential_name_is_redacted_in_the_query_string_too(name: str) -> None:
    assert RedactionPolicy().query({name: ["SECRET"]})[name] == [REDACTED]


@pytest.mark.parametrize("name", CREDENTIAL_NAMES)
def test_a_credential_name_is_redacted_however_it_is_capitalised(name: str) -> None:
    """``X-Api-Key`` in a header, ``API_KEY`` in a body: one ledger, one answer."""
    assert RedactionPolicy().payload({name.upper(): "SECRET"})[name.upper()] == REDACTED


#: The price of substring matching on ``key``, ``auth`` and ``session``. Paid knowingly:
#: see the comment on :data:`SENSITIVE_FIELDS`.
ACCEPTED_FALSE_POSITIVES = [
    "monkey",
    "donkey",
    "keyboard",
    "keyword",
    "key_count",
    "public_key",
    "author",
    "authority",
    "session_count",
    "discard",
]


@pytest.mark.parametrize("name", ACCEPTED_FALSE_POSITIVES)
def test_a_name_that_merely_contains_a_credential_word_is_redacted_anyway(name: str) -> None:
    """Over-redaction is the chosen direction, and it is chosen on purpose.

    ``public_key`` is not a secret and ``monkey`` is not a key at all. Losing either
    from an audit record costs an operator one field. Notarising a ``private_key``
    costs the key, permanently, with no way to erase it. There is no substring that
    separates ``api_key`` from ``monkey`` without also letting through the next
    credential name somebody invents, so this stays wrong in the safe direction.
    """
    assert RedactionPolicy().payload({name: "value"})[name] == REDACTED


#: The other side of the choice: names that must survive, or an audit trail records
#: nothing but ``[REDACTED]``.
#:
#: ``passport`` used to be on this list, as evidence that ``passw`` had not been widened
#: to a bare ``pass``. It is a credential now - a field called ``passport`` holds a
#: passport number, which is the same PII class as ``passport_number`` - so the pass/passw
#: boundary is pinned by ``passenger_count``, ``compass`` and ``bypass`` instead.
NAMES_THAT_MUST_SURVIVE = [
    "page",
    "limit",
    "offset",
    "amount",
    "currency",
    "email",
    "username",
    "user_id",
    "account_id",
    "status",
    "passenger_count",
    "compass",
    "bypass",
    "message",
    # The price of *not* squashing separators away before applying the short legacy
    # stems: ``metrics_snapshot`` holds no ``ssn``, and ``metricssnapshot`` would.
    "metrics_snapshot",
]


@pytest.mark.parametrize("name", NAMES_THAT_MUST_SURVIVE)
def test_an_ordinary_field_name_is_not_redacted(name: str) -> None:
    """Pins the upper bound of the pattern.

    Widening ``passw`` to a bare ``pass`` - the obvious "be even safer" edit - silently
    blanks ``passenger_count``, ``compass`` and ``bypass``, and an audit trail of
    ``[REDACTED]`` is not an audit trail. ``passw``/``pwd``/``passphrase`` already cover
    every real spelling of a password field, so the wider rule buys nothing.
    """
    assert RedactionPolicy().payload({name: "value"})[name] == "value"


def test_a_nested_body_credential_is_redacted_at_depth() -> None:
    """The Flask consumer posted a nested body and the credential was stored."""
    payload = {"payment": {"meta": {"api_key": "SECRETBODYAPIKEY55"}}}
    assert redact_payload(payload)["payment"]["meta"]["api_key"] == REDACTED


def test_a_credential_in_a_body_never_reaches_the_stored_bytes() -> None:
    """End to end, the way the Flask consumer proved the leak.

    It ran ``grep -a SECRETBODYAPIKEY55 audit.db`` and got two hits. The equivalent
    assertion here is against the serialised entry the backend actually holds.
    """
    policy = RedactionPolicy(capture_body=True)
    body = policy.decode_body(b'{"api_key": "SECRETBODYAPIKEY55", "amount": 10}')
    record = build_audit_record(
        integration="flask",
        method="POST",
        path="/api/transfer",
        status=200,
        client_ip="127.0.0.1",
        headers={},
        body=body,
        body_reason="captured",
    )
    with Ledger(backend=MemoryBackend(), ledger_id="b1") as led:
        entry = led.append(record)
    assert "SECRETBODYAPIKEY55" not in json.dumps(entry.data)


# ======================================================================================
# B4 - the families the denylist never had
#
# Unifying the three rules fixed the ``api_key`` family and stopped there. Nobody had
# ever fuzzed the *list* itself, so five whole categories of credential were still being
# notarised in the clear: one-time codes, crypto material, bearer tokens, key material and
# PII identifiers.
#
# ``passcode`` is the shape of the entire defect. The password family is spelled
# ``passw``/``pwd``/``passphrase``; ``passcode`` shares no substring with any of them, so
# a login form that calls its field ``passcode`` was stored verbatim, permanently, in a
# log that cannot be edited.
#
# The tests below therefore run in three parts: every missed family redacts, every
# ordinary field name still survives, and a generated corpus of credential-shaped names
# keeps both true so the *next* missing family is caught here rather than by a reviewer.
# ======================================================================================


#: One-time codes. None of these shares a substring with ``passw``/``pwd``/``passphrase``.
ONE_TIME_CODE_NAMES = [
    "otp",
    "totp",
    "hotp",
    "otp_code",
    "mfa",
    "mfa_code",
    "2fa_code",
    "pin",
    "pincode",
    "pin_code",
    "security_code",
    "verification_code",
    "confirmation_code",
    "recovery_code",
    "backup_code",
    "passcode",
    "one_time_code",
    "two_factor_code",
    "sms_otp",
    # A leading separator is what a Mongo-shaped payload looks like, and it is also the
    # only way an empty token reaches the splitter.
    "_otp",
]

#: Crypto material. A salt or a nonce is not a secret on its own; a seed, a mnemonic and
#: an HMAC key are the whole credential, and none of them can be rotated out of a log.
CRYPTO_MATERIAL_NAMES = [
    "salt",
    "seed",
    "mnemonic",
    "seed_phrase",
    "entropy",
    "nonce",
    "hmac",
    "signature",
    "password_salt",
    "recovery_seed",
    "hmac_sha256",
    "request_signature",
]

#: Bearer credentials that never say "token".
BEARER_TOKEN_NAMES = [
    "bearer",
    "jwt",
    "creds",
    "bearer_value",
    "jwt_header",
    "user_creds",
]

#: Key material in its file-format spellings. A ``.pem`` or a ``.p12`` posted to an
#: enrolment endpoint carries the private half.
KEY_MATERIAL_NAMES = [
    "cert",
    "pem",
    "pfx",
    "p12",
    "certificate",
    "client_cert",
    "cert_chain",
    "certs",
]

#: PII identifiers. Not passwords, and exactly what an append-only ledger must not be
#: holding when someone asks for erasure.
PII_IDENTIFIER_NAMES = [
    "account_number",
    "routing_number",
    "iban",
    "tax_id",
    "passport_number",
    "sin",
    "passport",
    "social_security_number",
    "bank_account_number",
]

MISSED_CREDENTIAL_NAMES = [
    *ONE_TIME_CODE_NAMES,
    *CRYPTO_MATERIAL_NAMES,
    *BEARER_TOKEN_NAMES,
    *KEY_MATERIAL_NAMES,
    *PII_IDENTIFIER_NAMES,
]


@pytest.mark.parametrize("name", MISSED_CREDENTIAL_NAMES)
def test_a_credential_family_the_denylist_missed_is_redacted(name: str) -> None:
    """Each of these was returned verbatim by ``policy.payload`` before this change."""
    assert RedactionPolicy().payload({name: "SECRET"})[name] == REDACTED


@pytest.mark.parametrize("name", MISSED_CREDENTIAL_NAMES)
def test_a_missed_family_is_redacted_on_every_transport(name: str) -> None:
    """Still one rule. Widening it must not reintroduce a per-transport answer."""
    policy = RedactionPolicy()
    assert (
        policy.payload({name: "S"})[name],
        policy.query({name: ["S"]})[name],
        redact_headers({name: "S"})[name.lower()],
        is_sensitive_field(name),
        is_sensitive_header(name),
    ) == (REDACTED, [REDACTED], REDACTED, True, True)


#: A multi-word credential name arrives spelled every way a language and a framework can
#: spell it. One entry in the list has to cover all of them or the list is decoration.
COMPOUND_SPELLINGS = [
    "security_code",
    "SECURITY_CODE",
    "security-code",
    "security.code",
    "securityCode",
    "SecurityCode",
    "securitycode",
    "user_security_code",
    "SecurityCodeInput",
]


@pytest.mark.parametrize("name", COMPOUND_SPELLINGS)
def test_a_compound_credential_name_is_redacted_however_it_is_spelled(name: str) -> None:
    assert RedactionPolicy().payload({name: "S"})[name] == REDACTED


#: The short stems - ``pin``, ``sin``, ``seed``, ``salt``, ``cert``, ``otp`` - are matched
#: as whole words, not as substrings. This is the list that pays for that decision, and
#: every one of them is a field an operator would be embarrassed to find blanked.
NAMES_A_SHORT_STEM_MUST_NOT_BLANK = [
    "shipping",
    "shipping_address",
    "spinner_state",
    "pinned_at",
    "mapping_rules",
    "single_use",
    "using_cache",
    "business_unit",
    "seeded_at",
    "seedling_count",
    "certainty_score",
    "concert_venue",
    "certification_status",
    "assault_charges",
    "announcement_body",
    "insurance_plan",
    "security_group",
    "opt_in",
    "options_list",
    "zip_code",
    "promo_code",
    "coupon_code",
    "country_code",
    "status_code",
    "error_code",
    "invoice_number",
    "order_number",
    "phone_number",
    "tracking_number",
    "serial_number",
    "tax_rate",
    "tax_amount",
    "sales_tax",
    "line_items",
    "delivery_status",
    "customer_email",
    "order_total",
    "first_name",
    "created_at",
    "assignment_id",
    "designation",
]


@pytest.mark.parametrize("name", NAMES_A_SHORT_STEM_MUST_NOT_BLANK)
def test_an_ordinary_field_name_survives_the_widened_rule(name: str) -> None:
    """The other direction, tested as hard as the first.

    ``pin`` as a substring blanks ``shipping``, ``spinner`` and ``mapping``; ``sin`` blanks
    ``business`` and ``using``; ``cert`` blanks ``certain``. A ledger whose every field
    reads ``[REDACTED]`` is not an audit trail, so the short stems match whole words only.
    """
    assert RedactionPolicy().payload({name: "value"})[name] == "value"


#: Redacted on purpose despite being ordinary-ish English, and listed so the cost of the
#: widening is written down rather than discovered.
WIDENED_FALSE_POSITIVES = [
    # ``certificate`` is deliberate: a field holding a certificate usually holds the PEM
    # blob, and a PEM blob sometimes holds the private half.
    "certificate_authority",
    # A database ``seed_data`` fixture is not crypto material. ``seed`` is worth it anyway.
    "seed_data",
    "random_seed",
    # The cost of matching ``tax_id`` with separators stripped, so that ``taxId`` cannot
    # slip through: one contrived field name pays for every real spelling.
    "syntax_id",
]


@pytest.mark.parametrize("name", WIDENED_FALSE_POSITIVES)
def test_a_known_false_positive_of_the_widened_rule_is_redacted(name: str) -> None:
    """Wrong in the safe direction, and known to be wrong."""
    assert RedactionPolicy().payload({name: "value"})[name] == REDACTED


# --------------------------------------------------------------------------------------
# The fuzz. A reviewer generating 146,237 credential-shaped names found 73 misses that
# five rounds of hand-written examples had not. That generator belongs in CI, not in a
# review, so it lives here - deterministic and exhaustive rather than random, because a
# corpus that differs between runs turns a real miss into a flake.
# --------------------------------------------------------------------------------------

#: One stem per family the rule claims to cover, including the ones it always covered.
FUZZ_STEMS = [
    "otp",
    "totp",
    "hotp",
    "mfa",
    "2fa",
    "pin",
    "pincode",
    "passcode",
    "security_code",
    "verification_code",
    "confirmation_code",
    "recovery_code",
    "backup_code",
    "one_time_code",
    "salt",
    "seed",
    "seed_phrase",
    "mnemonic",
    "entropy",
    "nonce",
    "hmac",
    "signature",
    "bearer",
    "jwt",
    "creds",
    "cert",
    "certificate",
    "pem",
    "pfx",
    "p12",
    "account_number",
    "routing_number",
    "iban",
    "tax_id",
    "passport",
    "sin",
    "api_key",
    "password",
    "secret",
    "token",
    "session",
    "cvv",
    "ssn",
    "credential",
]

#: What deployments put in front of a credential name.
FUZZ_PREFIXES = ["", "user", "client", "customer", "current", "x", "old"]

#: And behind it. ``hash`` is included because a hashed credential is still an oracle.
FUZZ_SUFFIXES = ["", "value", "field", "input", "confirm", "hash", "1"]


def _seams_survive_camel_casing(words: list) -> bool:
    """Whether camelCasing these words leaves any seam a reader could still see.

    ``p12`` followed by ``1`` is ``p121``; ``x`` followed by ``p12`` is ``XP12`` in
    PascalCase. In both the boundary is gone from the string itself, and the result is a
    different word rather than a badly spelled one. Asserting that a matcher must find a
    word that is no longer there would be testing a fiction, so those spellings are left
    out of the corpus instead of being quietly special-cased in the matcher.
    """
    return all(
        not (before[-1].isdigit() and after[0].isdigit())
        and (len(before) > 1 or after[1:2].isalpha())
        for before, after in zip(words, words[1:])
    )


def _spellings(words: list) -> list:
    """Every way a real codebase writes one multi-word field name.

    Deliberately *not* including the all-lowercase run-together form for these: the short
    stems are matched as whole words, so ``userpin`` is a name this rule does not claim to
    catch, and a test that pretended otherwise would be testing a promise nobody made.
    """
    lower = [word.lower() for word in words]
    joined = "_".join(lower)
    spellings = [joined, joined.upper(), "-".join(lower), ".".join(lower)]
    if _seams_survive_camel_casing(lower):
        camel = lower[0] + "".join(word.capitalize() for word in lower[1:])
        spellings += [camel, camel[:1].upper() + camel[1:]]
    return spellings


def _fuzz_corpus() -> list:
    """stems x prefixes x suffixes x spelling. ~13k names, one pass, no randomness."""
    names = []
    for stem in FUZZ_STEMS:
        for prefix in FUZZ_PREFIXES:
            for suffix in FUZZ_SUFFIXES:
                words = [*([prefix] if prefix else []), *stem.split("_")]
                words += [suffix] if suffix else []
                names.extend(_spellings(words))
    return names


def test_every_credential_shaped_name_in_the_generated_corpus_is_redacted() -> None:
    """The test that is supposed to find the *next* missing family.

    A denylist is only as good as the names somebody thought to try, so the names are
    generated rather than typed. Failure prints the misses themselves - that list is the
    bug report.
    """
    misses = [name for name in _fuzz_corpus() if not is_sensitive_field(name)]
    assert misses == [], f"{len(misses)} credential-shaped names stored in the clear: {misses[:20]}"


def test_the_generated_corpus_is_large_enough_to_be_worth_running() -> None:
    """A generator that silently shrinks to nothing still passes. Pin the size."""
    assert len(_fuzz_corpus()) > 10_000


#: The negative corpus. Same spellings, ordinary names.
ORDINARY_FUZZ_NAMES = [
    "shipping_address",
    "spinner_state",
    "pinned_at",
    "mapping_rules",
    "single_use",
    "using_cache",
    "business_unit",
    "seeded_at",
    "seedling_count",
    "certainty_score",
    "concert_venue",
    "certification_status",
    "assault_charges",
    "announcement_body",
    "insurance_plan",
    "security_group",
    "opt_in",
    "options_list",
    "zip_code",
    "promo_code",
    "invoice_number",
    "phone_number",
    "tracking_number",
    "tax_rate",
    "sales_tax",
    "line_items",
    "delivery_status",
    "customer_email",
    "order_total",
    "first_name",
    "created_at",
    "passenger_count",
]


def test_no_ordinary_field_name_in_the_generated_corpus_is_blanked() -> None:
    """Over-redaction is free only until it starts eating the audit trail."""
    blanked = [
        name
        for ordinary in ORDINARY_FUZZ_NAMES
        for name in _spellings(ordinary.split("_"))
        if is_sensitive_field(name)
    ]
    assert blanked == [], f"{len(blanked)} ordinary names blanked: {blanked[:20]}"


# --------------------------------------------------------------------------------------
# What this cannot do. Written down as tests, because an undocumented limit reads as a
# guarantee to whoever is deciding what to capture.
# --------------------------------------------------------------------------------------


@pytest.mark.timeout(60)
def test_a_hostile_object_key_cannot_stall_the_audit_write() -> None:
    """Splitting a name into words has to stay linear in the length of the name.

    An object key is client-written text. Reading one name as ``user`` + ``2fa`` + a
    joined ``user2fa`` means considering runs of letters and digits together, and
    considering *every* span of them is quadratic: ``{"a1a1a1...": 1}`` at 4 KB took 79
    seconds before this bound, on one key, in a middleware that runs on every request.
    That is the same denial of service the depth limit and the ``Content-Length`` check
    already exist to prevent, arriving through the field name instead of the value.
    """
    hostile = "a1" * 1_000
    started = time.perf_counter()
    assert redact_payload({hostile: "x"})[hostile] == "x"
    assert time.perf_counter() - started < 1.0


def test_a_credential_inside_free_text_is_stored_verbatim() -> None:
    """The honest limit of a denylist of *field names*: it never reads a value.

    A reviewer recovered exactly this string from a raw store. ``notes`` is not a
    credential name and the sentence inside it is never scanned, so the password is
    notarised. Nothing in this module changes that; only not capturing the body does.
    """
    note = "the customer said their password is hunter2"
    assert redact_payload({"notes": note})["notes"] == note


def test_a_deployment_can_widen_the_rule_with_its_own_field_names() -> None:
    """``extra_sensitive_fields`` is the supported answer to the paragraph above."""
    policy = RedactionPolicy(extra_sensitive_fields=["patient_note"])
    assert policy.payload({"patient_note": "hunter2"})["patient_note"] == REDACTED


@pytest.mark.parametrize("name", ["patient_name", "patientName", "Patient-Name", "PATIENT.NAME"])
def test_an_extra_sensitive_field_matches_every_spelling_of_itself(name: str) -> None:
    """An operator writes the name once; the API that sends it chooses the separators."""
    policy = RedactionPolicy(extra_sensitive_fields=["patient_name"])
    assert policy.payload({name: "ada"})[name] == REDACTED


def test_an_extra_sensitive_field_does_not_widen_to_everything_else() -> None:
    policy = RedactionPolicy(extra_sensitive_fields=["patient_name"])
    assert policy.payload({"page": "2"})["page"] == "2"


def test_an_extra_sensitive_field_covers_headers_as_well_as_bodies() -> None:
    """B1's defect, one layer up: the *added* names have to be one rule too.

    The built-in list stopped depending on which transport a credential arrived on;
    ``extra_sensitive_fields`` still did. A deployment that added ``x-tenant-passcode``
    got it scrubbed from bodies and query strings and notarised from the header - the
    exact shape of the bug that made ``?api_key=`` safe and ``{"api_key": ...}`` not.
    """
    policy = RedactionPolicy(extra_sensitive_fields=["x-tenant-note"])
    assert policy.headers({"X-Tenant-Note": "hunter2"})["x-tenant-note"] == REDACTED


# ======================================================================================
# B2 - an unpaired surrogate is a remote kill switch
#
# ``json.loads('{"u": "\\ud800"}')`` happily yields a Python str holding a lone
# surrogate. SLCS/1 rejects it, because it has no UTF-8 encoding. So one request body
# either 500s every audited endpoint (fail_closed) or deletes its own audit record
# (fail_open) - and the attacker picks which.
# ======================================================================================

#: A JSON body that parses cleanly and cannot be canonicalized.
SURROGATE_BODY = b'{"username": "\\ud800"}'


def test_a_string_that_utf8_cannot_encode_is_replaced_by_a_marker() -> None:
    assert redact_payload({"u": "\ud800"})["u"] == UNENCODABLE


def test_a_body_holding_an_unpaired_surrogate_is_marked_not_passed_through() -> None:
    assert RedactionPolicy(capture_body=True).decode_body(SURROGATE_BODY) == {
        "username": UNENCODABLE
    }


def test_a_body_holding_an_unpaired_surrogate_can_still_be_written_to_the_ledger() -> None:
    """The denial of service, reproduced with no web framework in the picture."""
    body = RedactionPolicy(capture_body=True).decode_body(SURROGATE_BODY)
    with Ledger(backend=MemoryBackend(), ledger_id="b2") as led:
        assert led.append({"body": body}).sequence == 0


def test_an_object_key_holding_an_unpaired_surrogate_is_marked_too() -> None:
    """SLCS/1 sorts keys by UTF-16 code unit, so it rejects an unencodable *key* as well."""
    assert redact_payload({"\ud800": "x"}) == {UNENCODABLE: "x"}


def test_a_sensitive_key_is_still_recognised_when_it_holds_a_surrogate() -> None:
    """Marking the key must not lose the reason its value had to go."""
    assert redact_payload({"password\ud800": "hunter2"}) == {UNENCODABLE: REDACTED}


def test_a_surrogate_inside_a_list_is_marked() -> None:
    assert redact_payload({"xs": ["ok", "\ud800"]})["xs"] == ["ok", UNENCODABLE]


def test_a_header_value_that_utf8_cannot_encode_is_marked() -> None:
    assert redact_headers({"x-note": "\ud800"})["x-note"] == UNENCODABLE


def test_a_header_name_that_utf8_cannot_encode_is_marked() -> None:
    assert redact_headers({"x-\ud800": "v"}) == {UNENCODABLE: "v"}


def test_a_client_ip_that_utf8_cannot_encode_is_marked() -> None:
    """``X-Forwarded-For`` is client-written text, and it becomes ``client_ip``."""
    policy = RedactionPolicy(trusted_proxy_count=1)
    assert policy.client_ip(peer="10.0.0.9", forwarded_for="\ud800") == UNENCODABLE


def test_a_peer_address_that_utf8_cannot_encode_is_marked() -> None:
    assert RedactionPolicy().client_ip(peer="\ud800", forwarded_for=None) == UNENCODABLE


def test_a_valid_astral_character_is_left_alone() -> None:
    """A *paired* surrogate is an ordinary emoji and must survive verbatim."""
    assert redact_payload({"u": "\U0001f600"})["u"] == "\U0001f600"


def test_a_whole_hostile_request_survives_canonicalization() -> None:
    """Everything this module emits is storable - that is the module's contract."""
    from signledger.crypto.canonical import canonicalize

    policy = RedactionPolicy(capture_body=True)
    record = build_audit_record(
        integration="fastapi",
        method="POST",
        path="/x",
        status=200,
        client_ip=policy.client_ip(peer="\ud800", forwarded_for=None),
        headers=policy.headers({"x-\ud801": "\ud802"}),
        query=policy.query({"q": ["\ud803"]}),
        body=policy.decode_body(SURROGATE_BODY),
        body_reason="captured",
    )
    canonicalize(record, allow_datetime=False)


# ======================================================================================
# B3 - one query key, one stored type
#
# Middlewares hand over a multi-dict, so a retained parameter was stored as ``["2"]``
# while a redacted one collapsed to the bare string ``"[REDACTED]"``. Anything reading
# the ledger then had to branch on the type of every value.
# ======================================================================================


def test_a_redacted_and_a_retained_query_value_have_the_same_type() -> None:
    result = RedactionPolicy().query({"api_key": ["sk-live-1"], "page": ["2"]})
    assert type(result["api_key"]) is type(result["page"])


def test_a_flat_query_mapping_is_normalised_to_lists() -> None:
    """Not every caller hands over a multi-dict; the stored shape must not depend on it."""
    assert RedactionPolicy().query({"page": "2"}) == {"page": ["2"]}


def test_a_repeated_query_parameter_keeps_every_value() -> None:
    assert RedactionPolicy().query({"tag": ["a", "b"]})["tag"] == ["a", "b"]


def test_a_repeated_sensitive_query_parameter_redacts_every_value() -> None:
    """Arity is kept - how many were sent is auditable, what they were is not."""
    assert RedactionPolicy().query({"api_key": ["a", "b"]})["api_key"] == [REDACTED, REDACTED]


def test_a_query_value_is_sanitised_like_any_other_payload() -> None:
    assert RedactionPolicy().query({"n": [2**53 + 1]})["n"] == ["9007199254740993"]


def test_an_empty_query_mapping_stays_empty() -> None:
    assert RedactionPolicy().query({}) == {}


# ======================================================================================
# Live PostgreSQL: the ledger row a DBA can actually read
#
# The Flask consumer's proof was ``grep -a SECRETBODYAPIKEY55`` over the raw store. This
# is that grep, against a real server, on the real ``data`` column.
# ======================================================================================

_PG_DSN = os.environ.get(
    "SIGNLEDGER_TEST_POSTGRES_DSN", "postgresql://postgres@localhost:5432/signledger_test"
)


@pytest.fixture
def redaction_pg_table() -> Iterator[str]:
    """A table name nothing else in the suite can collide with."""
    psycopg = pytest.importorskip("psycopg", reason="psycopg (v3) is not installed")
    from urllib.parse import urlsplit

    parts = urlsplit(_PG_DSN)
    host, port = parts.hostname or "localhost", parts.port or 5432
    try:
        with socket.create_connection((host, port), timeout=1.0):
            pass
    except OSError as err:
        pytest.skip(f"no live PostgreSQL at {host}:{port} ({type(err).__name__})")
    table = "sl_redact_" + uuid.uuid4().hex[:10]
    try:
        yield table
    finally:
        with psycopg.connect(_PG_DSN, connect_timeout=5) as connection:
            with connection.cursor() as cursor:
                for suffix in ("_entries", "_checkpoints", "_entries_state"):
                    cursor.execute(f'DROP TABLE IF EXISTS "{table}{suffix}" CASCADE')
            connection.commit()


@pytest.mark.postgres
def test_a_body_credential_is_absent_from_the_raw_postgres_row(redaction_pg_table: str) -> None:
    psycopg = pytest.importorskip("psycopg", reason="psycopg (v3) is not installed")
    from signledger.backends.postgresql import PostgresBackend

    policy = RedactionPolicy(capture_body=True)
    record = build_audit_record(
        integration="flask",
        method="POST",
        path="/api/transfer",
        status=200,
        client_ip="127.0.0.1",
        headers=policy.headers({"X-Api-Key": "SECRETHEADERAPIKEY99"}),
        query=policy.query({"api_key": ["SECRETQUERYAPIKEY77"]}),
        body=policy.decode_body(b'{"api_key": "SECRETBODYAPIKEY55", "amount": 10}'),
        body_reason="captured",
    )
    backend = PostgresBackend(
        _PG_DSN,
        entries_table=f"{redaction_pg_table}_entries",
        checkpoints_table=f"{redaction_pg_table}_checkpoints",
    )
    with Ledger(backend=backend, ledger_id="b1-pg") as led:
        led.append(record)

    with psycopg.connect(_PG_DSN, connect_timeout=5) as connection:
        with connection.cursor() as cursor:
            cursor.execute(f'SELECT data FROM "{redaction_pg_table}_entries"')  # noqa: S608
            stored = "".join(row[0] for row in cursor.fetchall())
    assert "SECRETBODYAPIKEY55" not in stored
    assert "SECRETQUERYAPIKEY77" not in stored
    assert "SECRETHEADERAPIKEY99" not in stored


@pytest.mark.postgres
def test_no_missed_family_reaches_the_raw_postgres_row(redaction_pg_table: str) -> None:
    """The same grep, for the families the list did not have.

    One request carrying a one-time code, a wallet seed, a bearer credential, key material
    and a national identifier - the five categories that were being notarised in the clear
    - and then a read of the column a DBA, a backup tape or a replica actually holds.
    """
    psycopg = pytest.importorskip("psycopg", reason="psycopg (v3) is not installed")
    from signledger.backends.postgresql import PostgresBackend

    policy = RedactionPolicy(capture_body=True)
    body = json.dumps(
        {
            "otp": "OTPVALUE111",
            "mnemonic": "MNEMONICVALUE222",
            "jwt": "JWTVALUE333",
            "client_cert": "CERTVALUE444",
            "account": {"passport_number": "PASSPORTVALUE555"},
            "amount": 10,
        }
    ).encode("utf-8")
    record = build_audit_record(
        integration="flask",
        method="POST",
        path="/api/enroll",
        status=200,
        client_ip="127.0.0.1",
        headers=policy.headers({"X-Otp": "HEADEROTP666"}),
        query=policy.query({"backup_code": ["QUERYBACKUP777"]}),
        body=policy.decode_body(body),
        body_reason="captured",
    )
    backend = PostgresBackend(
        _PG_DSN,
        entries_table=f"{redaction_pg_table}_entries",
        checkpoints_table=f"{redaction_pg_table}_checkpoints",
    )
    with Ledger(backend=backend, ledger_id="b4-pg") as led:
        led.append(record)

    with psycopg.connect(_PG_DSN, connect_timeout=5) as connection:
        with connection.cursor() as cursor:
            cursor.execute(f'SELECT data FROM "{redaction_pg_table}_entries"')  # noqa: S608
            stored = "".join(row[0] for row in cursor.fetchall())
    leaked = [value for value in ("OTPVALUE111", "MNEMONICVALUE222", "JWTVALUE333",
                                  "CERTVALUE444", "PASSPORTVALUE555", "HEADEROTP666",
                                  "QUERYBACKUP777") if value in stored]
    assert leaked == []
