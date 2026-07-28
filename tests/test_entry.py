"""Tests for the Entry model (contract section 3)."""

import copy
from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest
from pydantic import ValidationError as PydanticValidationError

from signledger.anchor.checkpoint import Checkpoint
from signledger.core import exceptions
from signledger.core.entry import Entry
from signledger.core.exceptions import CanonicalizationError, ValidationError
from signledger.crypto.canonical import MAX_ENTRY_BYTES, MAX_NESTING_DEPTH

TS = datetime(2024, 3, 1, 12, 0, 0, tzinfo=timezone.utc)


def make_entry(**overrides):
    fields = {
        "ledger_id": "11111111-1111-4111-8111-111111111111",
        "sequence": 0,
        "entry_id": "22222222-2222-4222-8222-222222222222",
        "timestamp": TS,
        "data": {"event": "created"},
        "previous_hash": "00" * 32,
    }
    fields.update(overrides)
    return Entry(**fields)


def test_entry_defaults_to_format_version_2():
    assert make_entry().format_version == 2


# --------------------------------------------------------------------------------------
# Immutability
# --------------------------------------------------------------------------------------


def test_entry_is_frozen():
    entry = make_entry()
    with pytest.raises(PydanticValidationError):
        entry.sequence = 99


def test_entry_rejects_assignment_to_the_hash_field():
    entry = make_entry()
    with pytest.raises(PydanticValidationError):
        entry.hash = "deadbeef"


# --------------------------------------------------------------------------------------
# Field defaults
# --------------------------------------------------------------------------------------


def test_metadata_defaults_to_an_empty_dict():
    assert make_entry().metadata == {}


def test_metadata_default_is_not_shared_between_entries():
    first = make_entry()
    second = make_entry()
    assert first.metadata is not second.metadata


def test_metadata_may_not_be_none():
    """Never NULL on disk - this one line broke the whole v1 SQLite and Postgres path."""
    with pytest.raises(ValidationError, match="metadata"):
        make_entry(metadata=None)


def test_previous_hash_may_not_be_none():
    with pytest.raises(ValidationError, match="previous_hash"):
        make_entry(previous_hash=None)


def test_previous_hash_may_not_be_empty():
    with pytest.raises(ValidationError, match="previous_hash"):
        make_entry(previous_hash="")


def test_key_id_and_signature_algorithm_default_to_none():
    entry = make_entry()
    assert entry.key_id is None
    assert entry.signature_algorithm is None


def test_signature_defaults_to_none():
    assert make_entry().signature is None


def test_hash_defaults_to_empty_so_a_draft_can_be_hashed_before_it_is_sealed():
    assert make_entry().hash == ""


# --------------------------------------------------------------------------------------
# data / metadata deep copy
# --------------------------------------------------------------------------------------


def test_mutating_the_source_data_dict_does_not_change_the_entry():
    source = {"event": "created"}
    entry = make_entry(data=source)
    source["event"] = "tampered"
    assert entry.data == {"event": "created"}


def test_mutating_a_nested_value_of_the_source_data_does_not_change_the_entry():
    source = {"outer": {"inner": [1, 2]}}
    entry = make_entry(data=source)
    source["outer"]["inner"].append(3)
    assert entry.data == {"outer": {"inner": [1, 2]}}


def test_mutating_the_source_metadata_dict_does_not_change_the_entry():
    source = {"actor": "svc"}
    entry = make_entry(metadata=source)
    source["actor"] = "attacker"
    assert entry.metadata == {"actor": "svc"}


def test_entry_does_not_hold_a_reference_to_the_caller_data_dict():
    source = {"event": "created"}
    assert make_entry(data=source).data is not source


# --------------------------------------------------------------------------------------
# timestamp
# --------------------------------------------------------------------------------------


def test_naive_timestamp_is_rejected():
    with pytest.raises(ValidationError, match="timestamp"):
        make_entry(timestamp=datetime(2024, 3, 1, 12, 0, 0))


def test_aware_timestamp_is_normalised_to_utc():
    kolkata = TS.astimezone(timezone(timedelta(hours=5, minutes=30)))
    entry = make_entry(timestamp=kolkata)
    assert entry.timestamp.utcoffset() == timedelta(0)
    assert entry.timestamp == TS


def test_epoch_number_is_not_accepted_as_a_timestamp():
    """Pydantic would read a bare int as *seconds*; microseconds would land in 56000 AD."""
    with pytest.raises(ValidationError, match="timestamp"):
        make_entry(timestamp=1709294400000000)


# --------------------------------------------------------------------------------------
# data payload rules
# --------------------------------------------------------------------------------------


def test_data_may_not_be_empty():
    with pytest.raises(ValidationError, match="data"):
        make_entry(data={})


def test_decimal_in_data_is_rejected_with_the_data_prefixed_json_path():
    with pytest.raises(CanonicalizationError) as excinfo:
        make_entry(data={"amount": Decimal("1.5")})
    assert str(excinfo.value) == "data.amount: Decimal is not JSON-native; convert to str"


def test_tuple_in_data_is_rejected():
    with pytest.raises(CanonicalizationError, match=r"data\.pair: tuple is not JSON-native"):
        make_entry(data={"pair": (1, 2)})


def test_datetime_inside_data_is_rejected():
    """Rule 5 is stricter than rule 3: only the `timestamp` field may be a datetime."""
    with pytest.raises(CanonicalizationError, match=r"data\.when: datetime is not allowed"):
        make_entry(data={"when": TS})


def test_datetime_inside_metadata_is_rejected():
    with pytest.raises(CanonicalizationError, match=r"metadata\.when: datetime is not allowed"):
        make_entry(metadata={"when": TS})


def test_non_str_key_in_data_is_rejected():
    with pytest.raises(CanonicalizationError, match="object keys must be str"):
        make_entry(data={1: "a"})


def test_oversized_int_in_data_is_rejected():
    with pytest.raises(CanonicalizationError, match="int exceeds"):
        make_entry(data={"n": 2**53 + 1})


def test_bad_type_in_metadata_reports_a_metadata_prefixed_path():
    with pytest.raises(CanonicalizationError, match=r"^metadata\.blob: bytes is not JSON-native"):
        make_entry(metadata={"blob": b"x"})


def test_oversized_data_is_rejected():
    with pytest.raises(CanonicalizationError, match="canonical form is"):
        make_entry(data={"big": "x" * MAX_ENTRY_BYTES})


def test_overly_nested_data_is_rejected():
    nested = {}
    for _ in range(MAX_NESTING_DEPTH):
        nested = {"k": nested}
    with pytest.raises(CanonicalizationError, match="nesting depth exceeds"):
        make_entry(data=nested)


# --------------------------------------------------------------------------------------
# sequence
# --------------------------------------------------------------------------------------


def test_sequence_zero_is_valid():
    assert make_entry(sequence=0).sequence == 0


def test_negative_sequence_is_rejected():
    with pytest.raises(ValidationError, match="sequence"):
        make_entry(sequence=-1)


# --------------------------------------------------------------------------------------
# from_dict
# --------------------------------------------------------------------------------------


def test_from_dict_builds_an_equal_entry():
    entry = make_entry()
    assert Entry.from_dict(entry.model_dump()) == entry


def test_from_dict_does_not_mutate_its_argument():
    source = {
        "ledger_id": "11111111-1111-4111-8111-111111111111",
        "sequence": 0,
        "entry_id": "22222222-2222-4222-8222-222222222222",
        "timestamp": TS,
        "data": {"event": "created"},
        "previous_hash": "00" * 32,
    }
    before = copy.deepcopy(source)
    Entry.from_dict(source)
    assert source == before


def test_from_dict_does_not_alias_the_nested_payload_it_was_given():
    source = {
        "ledger_id": "11111111-1111-4111-8111-111111111111",
        "sequence": 0,
        "entry_id": "22222222-2222-4222-8222-222222222222",
        "timestamp": TS,
        "data": {"nested": {"n": 1}},
        "previous_hash": "00" * 32,
    }
    entry = Entry.from_dict(source)
    source["data"]["nested"]["n"] = 2
    assert entry.data == {"nested": {"n": 1}}


def test_from_dict_accepts_an_iso_timestamp_string():
    entry = Entry.from_dict(
        {
            "ledger_id": "11111111-1111-4111-8111-111111111111",
            "sequence": 0,
            "entry_id": "22222222-2222-4222-8222-222222222222",
            "timestamp": "2024-03-01T12:00:00+00:00",
            "data": {"event": "created"},
            "previous_hash": "00" * 32,
        }
    )
    assert entry.timestamp == TS


def test_from_dict_rejects_a_naive_iso_timestamp_string():
    with pytest.raises(ValidationError, match="timestamp"):
        Entry.from_dict(
            {
                "ledger_id": "11111111-1111-4111-8111-111111111111",
                "sequence": 0,
                "entry_id": "22222222-2222-4222-8222-222222222222",
                "timestamp": "2024-03-01T12:00:00",
                "data": {"event": "created"},
                "previous_hash": "00" * 32,
            }
        )


# --------------------------------------------------------------------------------------
# Remaining field guards
# --------------------------------------------------------------------------------------


def test_data_that_is_not_an_object_is_rejected():
    with pytest.raises(ValidationError, match="data"):
        make_entry(data=["not", "an", "object"])


def test_metadata_that_is_not_an_object_is_rejected():
    with pytest.raises(ValidationError, match="metadata"):
        make_entry(metadata=["not", "an", "object"])


def test_format_version_below_one_is_rejected():
    with pytest.raises(ValidationError, match="format_version"):
        make_entry(format_version=0)


def test_empty_ledger_id_is_rejected():
    """An entry with no ledger_id is bound to no chain at all."""
    with pytest.raises(ValidationError, match="ledger_id"):
        make_entry(ledger_id="")


def test_empty_entry_id_is_rejected():
    with pytest.raises(ValidationError, match="entry_id"):
        make_entry(entry_id="")


def test_bool_is_not_mistaken_for_an_epoch_number_timestamp():
    """bool subclasses int; it must still fail as a type, not as an epoch guard."""
    with pytest.raises(ValidationError, match="timestamp"):
        make_entry(timestamp=True)


def test_metadata_accepts_an_empty_dict():
    assert make_entry(metadata={}).metadata == {}


def test_model_copy_can_seal_a_draft_with_its_hash():
    draft = make_entry()
    sealed = draft.model_copy(update={"hash": "ab" * 32})
    assert sealed.hash == "ab" * 32
    assert draft.hash == ""


# --------------------------------------------------------------------------------------
# Checkpoint (contract section 8)
#
# Kept in this file deliberately: tests/test_checkpoint.py is not in this agent's file
# list, and the anchor wave owns the sinks that would otherwise share such a module.
# --------------------------------------------------------------------------------------


def make_checkpoint(**overrides):
    fields = {
        "ledger_id": "11111111-1111-4111-8111-111111111111",
        "sequence": 41,
        "head_hash": "aa" * 32,
        "entry_count": 42,
        "merkle_root": "bb" * 32,
        "timestamp": TS,
    }
    fields.update(overrides)
    return Checkpoint(**fields)


def test_checkpoint_accepts_the_contract_field_order_positionally():
    cp = Checkpoint(
        "led-1", 41, "aa" * 32, 42, "bb" * 32, TS, "c2ln", "kid-1", 2
    )
    assert (cp.ledger_id, cp.sequence, cp.head_hash, cp.entry_count) == ("led-1", 41, "aa" * 32, 42)
    assert (cp.merkle_root, cp.timestamp, cp.signature, cp.key_id) == (
        "bb" * 32,
        TS,
        "c2ln",
        "kid-1",
    )
    assert cp.format_version == 2


def test_checkpoint_is_frozen():
    with pytest.raises(FrozenInstanceError):
        make_checkpoint().head_hash = "ff" * 32


def test_checkpoint_signature_and_key_id_default_to_none():
    cp = make_checkpoint()
    assert cp.signature is None
    assert cp.key_id is None


def test_checkpoint_defaults_to_format_version_2():
    assert make_checkpoint().format_version == 2


def test_checkpoints_with_identical_fields_are_equal():
    assert make_checkpoint() == make_checkpoint()


def test_checkpoints_differing_in_head_hash_are_not_equal():
    """head_hash and entry_count are what make tail truncation detectable."""
    assert make_checkpoint() != make_checkpoint(head_hash="cc" * 32)


def test_checkpoints_differing_in_entry_count_are_not_equal():
    assert make_checkpoint() != make_checkpoint(entry_count=41)


# --------------------------------------------------------------------------------------
# Package re-exports
# --------------------------------------------------------------------------------------


def test_core_package_re_exports_entry():
    import signledger.core as core

    assert core.Entry is Entry


def test_core_package_re_exports_the_whole_exception_hierarchy():
    import signledger.core as core

    assert core.SignLedgerError is exceptions.SignLedgerError
    assert set(exceptions.__all__) <= set(core.__all__)


def test_anchor_package_re_exports_checkpoint():
    import signledger.anchor as anchor

    assert anchor.Checkpoint is Checkpoint


def test_core_package_does_not_re_export_names_it_cannot_provide():
    import signledger.core as core

    for name in core.__all__:
        assert hasattr(core, name), name
