"""Tests for SLCS/1 canonical serialization (contract section 2)."""

import json
from datetime import date, datetime, timedelta, timezone, tzinfo
from decimal import Decimal

import pytest

from signledger.core.entry import Entry
from signledger.core.exceptions import CanonicalizationError
from signledger.crypto.canonical import (
    MAX_ENTRY_BYTES,
    MAX_NESTING_DEPTH,
    canonicalize,
    entry_preimage,
)


def test_object_is_encoded_without_insignificant_whitespace():
    assert canonicalize({"a": 1, "b": 2}) == b'{"a":1,"b":2}'


def test_keys_sort_by_utf16_code_unit_not_by_code_point():
    """RFC 8785 3.2.3.

    A non-BMP key becomes a surrogate pair starting at U+D83D, which sorts *before*
    U+FFFF in UTF-16 - but *after* it by Unicode code point, which is what Python's
    default ``sorted()`` uses. Getting this wrong makes two implementations of the
    same spec disagree on the digest of the very same object.
    """
    obj = {"￿": 1, "\U0001f600": 2}

    # Guard the premise: Python's own ordering really is the other way round.
    assert sorted(obj) == ["￿", "\U0001f600"]

    encoded = canonicalize(obj)
    assert encoded.index("\U0001f600".encode()) < encoded.index("￿".encode())


# --------------------------------------------------------------------------------------
# Rule 5 - only JSON-native types, rejection carries the JSON path
# --------------------------------------------------------------------------------------


def test_decimal_is_rejected_with_its_json_path():
    with pytest.raises(CanonicalizationError) as excinfo:
        canonicalize({"amount": Decimal("1.5")})
    assert "amount: Decimal is not JSON-native" in str(excinfo.value)


def test_tuple_and_list_do_not_collapse_to_the_same_bytes():
    """Silent tuple->list coercion is how two different payloads get one hash."""
    assert canonicalize({"v": [1, 2]}) == b'{"v":[1,2]}'
    with pytest.raises(CanonicalizationError):
        canonicalize({"v": (1, 2)})


def test_set_is_rejected():
    with pytest.raises(CanonicalizationError, match=r"v: set is not JSON-native"):
        canonicalize({"v": {1, 2}})


def test_bytes_are_rejected():
    with pytest.raises(CanonicalizationError, match=r"v: bytes is not JSON-native"):
        canonicalize({"v": b"x"})


def test_rejection_path_uses_dotted_keys_and_bracketed_indices():
    with pytest.raises(CanonicalizationError, match=r"a\.b\[2\]\.c: object is not JSON-native"):
        canonicalize({"a": {"b": [1, 2, {"c": object()}]}})


def test_rejection_at_the_root_names_the_root():
    with pytest.raises(CanonicalizationError, match=r"<root>: tuple is not JSON-native"):
        canonicalize(())


# --------------------------------------------------------------------------------------
# Rule 5 - non-str object keys
# --------------------------------------------------------------------------------------


def test_integer_object_key_is_rejected():
    """v1 collided {1: 'a'} with {'1': 'a'}."""
    with pytest.raises(CanonicalizationError, match="object keys must be str"):
        canonicalize({1: "a"})


def test_bool_object_key_is_rejected():
    with pytest.raises(CanonicalizationError, match="object keys must be str"):
        canonicalize({True: "a"})


def test_none_object_key_is_rejected():
    with pytest.raises(CanonicalizationError, match="object keys must be str"):
        canonicalize({None: "a"})


def test_non_str_key_rejection_names_the_containing_path():
    with pytest.raises(CanonicalizationError, match=r"^a\[0\]: object keys must be str"):
        canonicalize({"a": [{2: "b"}]})


# --------------------------------------------------------------------------------------
# Rule 4 - NaN / Infinity
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_float_is_rejected(value):
    with pytest.raises(CanonicalizationError, match="v: non-finite float"):
        canonicalize({"v": value})


# --------------------------------------------------------------------------------------
# Rule 6 - int range
# --------------------------------------------------------------------------------------


def test_int_at_the_positive_limit_is_accepted():
    assert canonicalize(2**53) == b"9007199254740992"


def test_int_one_past_the_positive_limit_is_rejected():
    with pytest.raises(CanonicalizationError, match=r"<root>: int exceeds \+/-2\*\*53"):
        canonicalize(2**53 + 1)


def test_int_at_the_negative_limit_is_accepted():
    assert canonicalize(-(2**53)) == b"-9007199254740992"


def test_int_one_past_the_negative_limit_is_rejected():
    with pytest.raises(CanonicalizationError, match=r"int exceeds"):
        canonicalize(-(2**53) - 1)


# --------------------------------------------------------------------------------------
# Rule 3 - datetime becomes integer microseconds since the epoch, UTC
# --------------------------------------------------------------------------------------


def test_datetime_encodes_as_integer_microseconds_since_epoch():
    moment = datetime(1970, 1, 1, 0, 0, 1, 500000, tzinfo=timezone.utc)
    assert canonicalize(moment) == b"1500000"


def test_datetime_before_the_epoch_encodes_as_a_negative_integer():
    moment = datetime(1969, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
    assert canonicalize(moment) == b"-1000000"


def test_same_instant_in_two_timezones_produces_identical_bytes():
    """v1 baked the UTC offset into the digest.

    A Postgres session running in a non-UTC timezone handed back the same instant with a
    different offset, every row re-hashed differently, and the whole log was reported
    tampered.
    """
    utc = datetime(2024, 3, 1, 12, 0, 0, tzinfo=timezone.utc)
    kolkata = utc.astimezone(timezone(timedelta(hours=5, minutes=30)))
    honolulu = utc.astimezone(timezone(timedelta(hours=-10)))

    assert kolkata.utcoffset() != honolulu.utcoffset()
    assert canonicalize({"t": kolkata}) == canonicalize({"t": honolulu})
    assert canonicalize({"t": utc}) == canonicalize({"t": kolkata})


def test_datetime_never_encodes_as_an_iso_string():
    moment = datetime(2024, 3, 1, 12, 0, 0, tzinfo=timezone.utc)
    assert b"2024" not in canonicalize(moment)


def test_naive_datetime_is_rejected():
    with pytest.raises(CanonicalizationError, match="t: naive datetime is ambiguous"):
        canonicalize({"t": datetime(2024, 3, 1, 12, 0, 0)})


def test_datetime_with_a_tzinfo_that_yields_no_offset_is_rejected():
    """A tzinfo subclass may return None from utcoffset(); that is still naive."""

    class NoOffset(tzinfo):
        def utcoffset(self, dt):
            return None

        def dst(self, dt):
            return None

        def tzname(self, dt):
            return "none"

    with pytest.raises(CanonicalizationError, match="naive datetime"):
        canonicalize({"t": datetime(2024, 3, 1, 12, 0, 0, tzinfo=NoOffset())})


def test_date_without_time_is_rejected():
    with pytest.raises(CanonicalizationError, match="d: date is not JSON-native"):
        canonicalize({"d": date(2024, 3, 1)})


# --------------------------------------------------------------------------------------
# Numbers - ECMAScript Number::toString (RFC 8785 3.2.2.3)
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1.0, b"1"),
        (-1.0, b"-1"),
        (0.0, b"0"),
        (-0.0, b"0"),
        (0.1, b"0.1"),
        (123.456, b"123.456"),
        (1e20, b"100000000000000000000"),
        (1e21, b"1e+21"),
        (1.5e21, b"1.5e+21"),
        (1e-6, b"0.000001"),
        (1e-7, b"1e-7"),
        (1.2e-7, b"1.2e-7"),
        (5e-324, b"5e-324"),
        (1.7976931348623157e308, b"1.7976931348623157e+308"),
    ],
)
def test_float_uses_ecmascript_number_formatting(value, expected):
    assert canonicalize(value) == expected


def test_integral_float_and_int_are_the_same_json_number():
    """1 and 1.0 are one JSON value.

    Encoding them differently would break `store -> load -> re-hash`: a JSON store hands
    back 1.0 as an int, and the re-hash would no longer match what was written.
    """
    assert canonicalize({"n": 1}) == canonicalize({"n": 1.0})


def test_negative_zero_and_positive_zero_agree():
    assert canonicalize({"n": -0.0}) == canonicalize({"n": 0.0})


# --------------------------------------------------------------------------------------
# Strings - RFC 8785 3.2.2.2 escaping
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("", b'""'),
        ("plain", b'"plain"'),
        ('he said "hi"', b'"he said \\"hi\\""'),
        ("back\\slash", b'"back\\\\slash"'),
        ("tab\there", b'"tab\\there"'),
        ("nl\n", b'"nl\\n"'),
        ("cr\r", b'"cr\\r"'),
        ("bs\b", b'"bs\\b"'),
        ("ff\f", b'"ff\\f"'),
        ("\x00", b'"\\u0000"'),
        ("\x1f", b'"\\u001f"'),
    ],
)
def test_string_escaping_matches_rfc8785(value, expected):
    assert canonicalize(value) == expected


def test_forward_slash_is_not_escaped():
    assert canonicalize("a/b") == b'"a/b"'


def test_del_and_other_non_control_characters_stay_literal():
    assert canonicalize("\x7f") == b'"\x7f"'


def test_non_ascii_characters_are_emitted_as_literal_utf8_not_escaped():
    assert canonicalize("café \U0001f600") == "café \U0001f600".encode().join((b'"', b'"'))


def test_unpaired_surrogate_in_a_value_is_rejected():
    with pytest.raises(CanonicalizationError, match="v: str contains an unpaired surrogate"):
        canonicalize({"v": "\ud800"})


def test_unpaired_surrogate_in_a_key_is_rejected():
    with pytest.raises(CanonicalizationError, match="unpaired surrogate"):
        canonicalize({"\udfff": 1})


# --------------------------------------------------------------------------------------
# Rule 7 - MAX_NESTING_DEPTH
# --------------------------------------------------------------------------------------


def _nest_lists(depth):
    obj = []
    for _ in range(depth - 1):
        obj = [obj]
    return obj


def _nest_dicts(depth):
    obj = {}
    for _ in range(depth - 1):
        obj = {"k": obj}
    return obj


def test_nesting_exactly_at_the_depth_limit_is_accepted():
    assert canonicalize(_nest_lists(MAX_NESTING_DEPTH)).count(b"[") == MAX_NESTING_DEPTH


def test_nesting_one_level_past_the_depth_limit_is_rejected():
    with pytest.raises(CanonicalizationError, match="nesting depth exceeds the maximum of 64"):
        canonicalize(_nest_lists(MAX_NESTING_DEPTH + 1))


def test_dict_nesting_one_level_past_the_depth_limit_is_rejected():
    with pytest.raises(CanonicalizationError, match="nesting depth exceeds"):
        canonicalize(_nest_dicts(MAX_NESTING_DEPTH + 1))


def test_depth_rejection_reports_the_path_of_the_offending_container():
    obj = {"a": _nest_dicts(MAX_NESTING_DEPTH)}  # one container too many, all told
    expected = "a" + ".k" * (MAX_NESTING_DEPTH - 1)
    with pytest.raises(CanonicalizationError) as excinfo:
        canonicalize(obj)
    assert str(excinfo.value).startswith(expected + ": nesting depth")


def test_deeply_nested_input_raises_canonicalization_error_not_recursion_error():
    """The depth guard must trip before Python's own stack does."""
    with pytest.raises(CanonicalizationError):
        canonicalize(_nest_lists(20_000))


# --------------------------------------------------------------------------------------
# Rule 7 - MAX_ENTRY_BYTES
# --------------------------------------------------------------------------------------


def test_encoding_exactly_at_the_byte_limit_is_accepted():
    payload = "x" * (MAX_ENTRY_BYTES - 2)  # two bytes for the surrounding quotes
    assert len(canonicalize(payload)) == MAX_ENTRY_BYTES


def test_encoding_one_byte_past_the_limit_is_rejected():
    payload = "x" * (MAX_ENTRY_BYTES - 1)
    with pytest.raises(CanonicalizationError, match="canonical form is 1048577 bytes"):
        canonicalize(payload)


def test_byte_limit_counts_utf8_bytes_not_characters():
    """A multi-byte character must consume its full byte cost against the budget."""
    emoji_count = (MAX_ENTRY_BYTES - 2) // 4
    padding = (MAX_ENTRY_BYTES - 2) - emoji_count * 4
    payload = "\U0001f600" * emoji_count + "x" * padding  # exactly at the limit
    assert len(payload) < MAX_ENTRY_BYTES  # fewer characters than bytes
    assert len(canonicalize(payload)) == MAX_ENTRY_BYTES
    with pytest.raises(CanonicalizationError):
        canonicalize(payload + "x")


# --------------------------------------------------------------------------------------
# Literals and containers
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, b"null"),
        (True, b"true"),
        (False, b"false"),
        ({}, b"{}"),
        ([], b"[]"),
        ([1, "a", None, True, False], b'[1,"a",null,true,false]'),
        ({"a": {"b": [1, {"c": None}]}}, b'{"a":{"b":[1,{"c":null}]}}'),
    ],
)
def test_literals_and_containers_encode_exactly(value, expected):
    assert canonicalize(value) == expected


def test_bool_is_encoded_as_a_boolean_not_as_an_integer():
    """bool subclasses int, so a naive isinstance order silently writes 1 and 0."""
    assert canonicalize({"v": True}) == b'{"v":true}'
    assert canonicalize({"v": 1}) == b'{"v":1}'
    assert canonicalize({"v": True}) != canonicalize({"v": 1})


def test_key_order_of_the_input_dict_does_not_change_the_bytes():
    assert canonicalize({"b": 1, "a": 2}) == canonicalize({"a": 2, "b": 1})


def test_keys_sort_across_nested_objects_too():
    assert canonicalize({"z": {"b": 1, "a": 2}}) == b'{"z":{"a":2,"b":1}}'


def test_list_order_is_preserved():
    assert canonicalize([3, 1, 2]) == b"[3,1,2]"


def test_ascii_keys_sort_by_code_unit():
    assert canonicalize({"B": 1, "a": 2, "A": 3}) == b'{"A":3,"B":1,"a":2}'


def test_path_prefix_is_applied_to_nested_error_messages():
    with pytest.raises(CanonicalizationError, match=r"data\.items\[1\]: tuple is not"):
        canonicalize({"items": [1, (2,)]}, path="data")


def test_path_prefix_does_not_change_the_produced_bytes():
    assert canonicalize({"a": 1}, path="data") == canonicalize({"a": 1})


# --------------------------------------------------------------------------------------
# entry_preimage
# --------------------------------------------------------------------------------------

TS = datetime(2024, 3, 1, 12, 0, 0, tzinfo=timezone.utc)


def make_entry(**overrides):
    fields = {
        "ledger_id": "11111111-1111-4111-8111-111111111111",
        "sequence": 7,
        "entry_id": "22222222-2222-4222-8222-222222222222",
        "timestamp": TS,
        "data": {"event": "created"},
        "metadata": {"actor": "svc"},
        "previous_hash": "00" * 32,
        "key_id": "kid-1",
        "signature_algorithm": "ed25519",
        "hash": "aa" * 32,
        "signature": "c2ln",
    }
    fields.update(overrides)
    return Entry(**fields)


def test_entry_preimage_covers_exactly_the_ten_contract_fields():
    decoded = json.loads(entry_preimage(make_entry()).decode())
    assert sorted(decoded) == [
        "data",
        "entry_id",
        "format_version",
        "key_id",
        "ledger_id",
        "metadata",
        "previous_hash",
        "sequence",
        "signature_algorithm",
        "timestamp",
    ]


def test_entry_preimage_excludes_the_hash_field():
    base = make_entry()
    assert entry_preimage(base) == entry_preimage(base.model_copy(update={"hash": "bb" * 32}))


def test_entry_preimage_excludes_the_signature_field():
    base = make_entry()
    assert entry_preimage(base) == entry_preimage(base.model_copy(update={"signature": "b3Rp"}))


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("format_version", 3),
        ("ledger_id", "99999999-9999-4999-8999-999999999999"),
        ("sequence", 8),
        ("entry_id", "33333333-3333-4333-8333-333333333333"),
        ("timestamp", datetime(2024, 3, 1, 12, 0, 1, tzinfo=timezone.utc)),
        ("data", {"event": "deleted"}),
        ("metadata", {"actor": "attacker"}),
        ("previous_hash", "11" * 32),
        ("key_id", "kid-2"),
        ("signature_algorithm", "rsa-pss-sha256"),
    ],
)
def test_changing_any_preimage_field_changes_the_bytes(field, replacement):
    base = make_entry()
    assert entry_preimage(base) != entry_preimage(base.model_copy(update={field: replacement}))


def test_entry_preimage_encodes_the_timestamp_as_microseconds():
    decoded = json.loads(entry_preimage(make_entry()).decode())
    assert decoded["timestamp"] == 1709294400000000


def test_entry_preimage_keeps_a_null_key_id_distinct_from_an_empty_one():
    """key_id lives in the preimage so a signing identity cannot be swapped silently."""
    unsigned = make_entry(key_id=None)
    named = make_entry(key_id="")
    assert entry_preimage(unsigned) != entry_preimage(named)


def _payload_of_depth(depth):
    """A data payload whose own outermost dict counts as level 1."""
    nested = {}
    for _ in range(depth - 1):
        nested = {"k": nested}
    return nested


def test_entry_preimage_gives_data_the_full_nesting_budget():
    """The preimage envelope must not eat into the payload's own depth allowance.

    Whatever Entry accepts for `data`, the preimage must be able to encode - otherwise
    an entry could be constructed and then never hashed.
    """
    at_limit = _payload_of_depth(MAX_NESTING_DEPTH)
    assert canonicalize(at_limit, path="data")  # Entry's own check passes
    assert entry_preimage(make_entry(data=at_limit))  # so the preimage must too


def test_entry_preimage_is_stable_across_equivalent_entries():
    assert entry_preimage(make_entry()) == entry_preimage(make_entry())


def test_allow_datetime_false_rejects_a_datetime_with_its_path():
    with pytest.raises(CanonicalizationError, match=r"^when: datetime is not allowed"):
        canonicalize({"when": TS}, allow_datetime=False)


def test_allow_datetime_false_leaves_every_other_type_alone():
    assert canonicalize({"a": 1}, allow_datetime=False) == canonicalize({"a": 1})


# --------------------------------------------------------------------------------------
# Mutation-sweep hardening
#
# Everything below was written to kill a specific surviving mutant found by the sweep in
# docs/ARCHITECTURE.md section 11. Each test names the production edit it would catch, so
# that a future reader can tell whether it is still earning its place.
# --------------------------------------------------------------------------------------


class _OffsetOnlyForRealDatetimes(tzinfo):
    """A tzinfo whose ``utcoffset`` needs the datetime, exactly like ``zoneinfo``.

    ``zoneinfo.ZoneInfo(...).utcoffset(None)`` returns ``None`` - a real zone cannot say
    what its offset is without knowing *when*. ``timezone(timedelta(...))`` answers
    happily either way, which is why every fixed-offset test in this file passes against
    an encoder that asks ``utcoffset(None)``. Reproduced here rather than importing
    ``zoneinfo`` so the test needs no tz database on the machine running it.
    """

    def utcoffset(self, dt):
        return None if dt is None else timedelta(hours=5, minutes=30)

    def dst(self, dt):
        return None

    def tzname(self, dt):
        return "IST"


def test_a_datetime_whose_offset_depends_on_the_instant_is_not_treated_as_naive():
    """Kills ``value.tzinfo.utcoffset(value)`` -> ``value.tzinfo.utcoffset(None)``.

    That edit rejects every ``zoneinfo``-tagged datetime as "naive", which is the single
    most common way a real application spells an aware timestamp.
    """
    zone = _OffsetOnlyForRealDatetimes()
    assert zone.utcoffset(None) is None  # guard the premise

    local = datetime(2024, 3, 1, 17, 30, tzinfo=zone)
    assert canonicalize({"t": local}) == canonicalize(
        {"t": datetime(2024, 3, 1, 12, 0, tzinfo=timezone.utc)}
    )


def test_a_datetime_inside_a_list_still_encodes_as_microseconds():
    """Kills dropping ``allow_datetime`` on the way into, or through, an array.

    Every other datetime test in this file passes a datetime as an object *value*, so an
    encoder that forgot to forward the flag through ``_encode_array`` looked correct.
    """
    moment = datetime(1970, 1, 1, 0, 0, 1, 500000, tzinfo=timezone.utc)
    assert canonicalize([moment]) == b"[1500000]"
    assert canonicalize({"v": [moment]}) == b'{"v":[1500000]}'


def test_a_far_future_datetime_is_rejected_with_the_path_of_the_field():
    """Microseconds since the epoch pass through the same +/-2**53 gate as any int.

    Kills dropping ``path`` on the way to ``_encode_int``: the offender is ``t``, not the
    root.
    """
    with pytest.raises(CanonicalizationError, match=r"^t: int exceeds \+/-2\*\*53"):
        canonicalize({"t": datetime(9999, 1, 1, tzinfo=timezone.utc)})


def test_an_oversized_int_inside_a_container_reports_its_own_path():
    """Kills dropping ``path`` on the way to ``_encode_int``.

    The existing limit tests all canonicalize a bare int at the root, where a lost path
    still renders as ``<root>``.
    """
    with pytest.raises(CanonicalizationError, match=r"^v\[1\]: int exceeds"):
        canonicalize({"v": [0, 2**53 + 1]})


def test_tuple_rejection_carries_the_documented_conversion_hint():
    """Kills every mutation of the ``_CONVERSION_HINTS`` lookup.

    The hint is the actionable half of the message; without it the error says only that
    something is wrong.
    """
    with pytest.raises(
        CanonicalizationError, match=r"^v: tuple is not JSON-native; convert to list$"
    ):
        canonicalize({"v": (1,)})


def test_a_type_with_no_specific_hint_falls_back_to_the_generic_one():
    """Kills ``_CONVERSION_HINTS.get(name)`` - the missing default renders as ``None``."""
    with pytest.raises(
        CanonicalizationError,
        match=r"^v: object is not JSON-native; "
        r"use only dict, list, str, int, float, bool or None$",
    ):
        canonicalize({"v": object()})


def test_non_str_key_rejection_names_the_type_of_the_offending_key():
    """Kills ``type(key).__name__`` -> a constant. The type is the whole diagnosis."""
    with pytest.raises(CanonicalizationError, match=r"object keys must be str, got int$"):
        canonicalize({1: "a"})


def test_a_surrogate_in_a_nested_key_reports_the_containing_path():
    """Kills dropping ``path`` in the key sort.

    The existing surrogate-key test uses a root-level object, where a lost path is
    indistinguishable from the real one.
    """
    with pytest.raises(CanonicalizationError, match=r"^a: str contains an unpaired surrogate"):
        canonicalize({"a": {"\ud800": 1}})


def test_depth_rejection_inside_a_list_reports_the_path_of_the_offending_array():
    """Kills dropping ``path`` in the array depth guard.

    Its object-shaped twin is already covered; the array branch is a separate call site.
    """
    obj = {"a": _nest_lists(MAX_NESTING_DEPTH)}  # one container too many, all told
    expected = "a" + "[0]" * (MAX_NESTING_DEPTH - 1)
    with pytest.raises(CanonicalizationError) as excinfo:
        canonicalize(obj)
    assert str(excinfo.value).startswith(expected + ": nesting depth")


@pytest.mark.parametrize(
    ("value", "name"),
    [(float("nan"), "NaN"), (float("inf"), "Infinity"), (float("-inf"), "-Infinity")],
)
def test_non_finite_rejection_names_which_non_finite_value_it_was(value, name):
    """Kills every mutation of ``_nonfinite_name``.

    RFC 8785 names these three exactly; a reader debugging a rejected payload needs to
    know which one arrived.
    """
    with pytest.raises(CanonicalizationError) as excinfo:
        canonicalize({"v": value})
    assert str(excinfo.value) == f"v: non-finite float ({name}) has no JSON form"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        # Trailing-zero stripping must look at the *last* digit. `digits[1]` happens to
        # agree for 1.0, 10.0 and 100.0 - every integral value already in the table
        # above - and disagrees the moment the second digit is non-zero.
        (12.0, b"12"),
        (120.0, b"120"),
        (1200.0, b"1200"),
        # k > n with n == 1: the "0 < n <= 21" branch. 0.1 has n == 0 and 123.456 has
        # n == 3, so the lower boundary of that range was never pinned.
        (1.5, b"1.5"),
        (9.75, b"9.75"),
    ],
)
def test_ecmascript_number_formatting_at_the_boundaries_the_table_above_misses(value, expected):
    assert canonicalize(value) == expected


def test_entry_preimage_refuses_data_one_level_past_the_nesting_limit():
    """Pins the preimage envelope's depth credit to exactly one level, no more.

    ``test_entry_preimage_gives_data_the_full_nesting_budget`` pins the floor: a payload
    at the limit must encode. This pins the ceiling: a payload one past it must not. With
    only the floor test, an envelope that handed ``data`` 65 levels looked correct, and an
    Entry rebuilt from a store - where the field validators do not run - could be hashed
    with a payload deeper than SLCS/1 admits.
    """
    # model_copy skips the field validators, exactly as a store rehydrating a row does.
    too_deep = make_entry().model_copy(
        update={"data": _payload_of_depth(MAX_NESTING_DEPTH + 1)}
    )
    with pytest.raises(CanonicalizationError) as excinfo:
        entry_preimage(too_deep)
    message = str(excinfo.value)
    assert "nesting depth exceeds the maximum of 64" in message
    assert message.startswith("data")  # the seeded path is the field name, not a constant
