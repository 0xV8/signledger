"""The half of every integration that must not differ between frameworks.

``signledger.contrib`` exists so that the parts an auditor's query depends on are written
once. Three of them had drifted apart anyway, each in a way no single-framework test could
see, so each is pinned here against the shared implementation rather than against any one
web framework:

* what a settlement entry looks like - both integrations hand-assembled it, so it carried
  none of the fields the builder's own docstring promises are always present;
* whether a response will send a body at all - Flask and FastAPI disagreed about ``HEAD``;
* which paths are skipped - all three used a bare prefix match, so ``skip_paths=("/health",)``
  silently un-audited ``/healthz-admin-secret``.
"""

import pytest

from signledger.contrib import (
    COMPLETION_FINAL,
    COMPLETION_PROVISIONAL,
    KIND_COMPLETION,
    KIND_REQUEST,
    OUTCOME_COMPLETE,
    OUTCOME_INCOMPLETE,
    build_audit_record,
    is_skipped,
    will_send_body,
)

# --------------------------------------------------------------------------------------
# settlement entries carry the same shape as the request entry they settle
# --------------------------------------------------------------------------------------


def request_record(**overrides):
    fields = {
        "integration": "flask",
        "method": "GET",
        "path": "/download",
        "status": 200,
        "client_ip": "203.0.113.7",
        "headers": {"accept": "*/*"},
        "query": {"page": ["2"]},
        "body": None,
        "body_reason": "disabled",
        "user": "17",
        "request_id": "abc123",
    }
    fields.update(overrides)
    return build_audit_record(**fields)


def test_a_settlement_entry_carries_the_same_keys_as_the_request_entry():
    """The divergence this exists to stop. Both ``_settle`` implementations built a plain
    dict, so a settlement entry had no ``completion``, no ``client_ip``, no ``user``, no
    ``headers``, no ``query`` and no ``body`` - and an auditor asking "who" about the half
    of the request that failed could not be answered."""
    request = request_record(kind=KIND_REQUEST, completion=COMPLETION_PROVISIONAL)
    settlement = request_record(
        kind=KIND_COMPLETION,
        completion=COMPLETION_FINAL,
        outcome=OUTCOME_COMPLETE,
    )

    assert set(settlement) - {"outcome"} == set(request)


def test_a_settlement_entry_names_its_outcome():
    settlement = request_record(kind=KIND_COMPLETION, outcome=OUTCOME_INCOMPLETE)

    assert settlement["outcome"] == OUTCOME_INCOMPLETE


def test_a_settlement_entry_is_itself_final():
    """``completion`` says whether the *status* on this entry is a fact. On a settlement
    it always is: the settlement is written once the body has stopped moving."""
    settlement = request_record(kind=KIND_COMPLETION, outcome=OUTCOME_COMPLETE)

    assert settlement["completion"] == COMPLETION_FINAL


def test_a_request_entry_has_no_outcome_key():
    """A field that is always present and usually meaningless teaches people to stop
    reading it, which is the same reasoning ``error`` is absent rather than null."""
    assert "outcome" not in request_record(kind=KIND_REQUEST)


def test_a_settlement_path_goes_through_the_redaction_choke_point():
    """``_settle`` bypassed ``build_audit_record`` entirely, which meant it also bypassed
    the one place unencodable text is neutralised - reopened for exactly the entries
    written after the response has gone, where a refused write has nowhere left to be
    reported."""
    settlement = request_record(
        path="/files/\ud800", kind=KIND_COMPLETION, outcome=OUTCOME_COMPLETE
    )

    assert settlement["path"].encode("utf-8")


# --------------------------------------------------------------------------------------
# whether a body is coming at all
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("status", [200, 201, 404, 500])
def test_an_ordinary_response_sends_a_body(status):
    assert will_send_body("GET", status) is True


@pytest.mark.parametrize("status", [204, 304])
def test_a_bodyless_status_sends_no_body(status):
    assert will_send_body("GET", status) is False


def test_an_informational_status_sends_no_body():
    assert will_send_body("GET", 100) is False


def test_head_sends_no_body_whatever_the_status():
    """The divergence: Flask special-cased HEAD and FastAPI did not, so a HEAD against a
    streaming route was one final entry on Flask and Django and two entries with a
    provisional first on FastAPI - for byte-identical traffic."""
    assert will_send_body("HEAD", 200) is False


# --------------------------------------------------------------------------------------
# which paths are skipped
# --------------------------------------------------------------------------------------


def test_the_named_path_itself_is_skipped():
    assert is_skipped("/health", ("/health",)) is True


def test_a_path_below_the_named_prefix_is_skipped():
    assert is_skipped("/health/live", ("/health",)) is True


def test_a_trailing_slash_on_the_prefix_makes_no_difference():
    assert is_skipped("/health/live", ("/health/",)) is True


def test_a_sibling_route_sharing_the_prefix_is_not_skipped():
    """The audit-completeness hole. ``skip_paths=("/health",)`` is what the README tells
    deployers to use, and a bare ``startswith`` let a real route serving real data return
    200 with no entry written at all."""
    assert is_skipped("/healthz-admin-secret", ("/health",)) is False


def test_an_unrelated_path_is_not_skipped():
    assert is_skipped("/items", ("/health",)) is False


def test_no_prefixes_skips_nothing():
    assert is_skipped("/health", ()) is False


def test_any_one_matching_prefix_is_enough():
    assert is_skipped("/metrics", ("/health", "/metrics")) is True
