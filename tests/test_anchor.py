"""Checkpoints and external anchoring (architecture contract, section 8).

An anchor is the second of the two things that make a log tamper-*evident* rather than
merely hash-chained: a chain head published somewhere the adversary does not control.
Without it, an insider who can edit rows can also re-derive every digest after the one
they changed, and a bare chain will happily report itself intact.
"""

import json
from datetime import datetime, timedelta, timezone

import pytest

from signledger.anchor.checkpoint import Checkpoint
from signledger.anchor.sinks import (
    AnchorSink,
    FileAnchorSink,
    NullAnchorSink,
    checkpoint_signing_payload,
)
from signledger.core.exceptions import AnchorError

MOMENT = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)


def make_checkpoint(sequence=3, ledger_id="ledger-a", **overrides):
    fields = {
        "ledger_id": ledger_id,
        "sequence": sequence,
        "head_hash": f"{sequence:064x}",
        "entry_count": sequence + 1,
        "merkle_root": f"{sequence:064X}",
        "timestamp": MOMENT + timedelta(minutes=sequence),
    }
    fields.update(overrides)
    return Checkpoint(**fields)


@pytest.fixture
def anchor_path(tmp_path):
    return str(tmp_path / "anchors.jsonl")


@pytest.fixture
def sink(anchor_path):
    return FileAnchorSink(anchor_path)


# --------------------------------------------------------------------------------------
# the null sink
# --------------------------------------------------------------------------------------


def test_the_null_sink_names_itself():
    assert NullAnchorSink().name == "null"


def test_the_null_sink_returns_a_receipt_for_the_checkpoint_it_was_given():
    receipt = NullAnchorSink().publish(make_checkpoint(7))

    assert (receipt.sink, receipt.ledger_id, receipt.sequence) == ("null", "ledger-a", 7)


def test_the_null_sink_never_hands_a_checkpoint_back():
    """It discards, so it can never confer ANCHORED assurance - which is the honest answer
    for a sink that stores nothing outside the process."""
    null = NullAnchorSink()
    null.publish(make_checkpoint(7))

    assert null.fetch_latest("ledger-a") is None


# --------------------------------------------------------------------------------------
# the file sink
# --------------------------------------------------------------------------------------


def test_the_file_sink_names_itself(sink):
    assert sink.name == "file"


def test_a_published_checkpoint_comes_back_field_for_field(sink):
    published = make_checkpoint(4, signature="c2ln", key_id="kid-1")
    sink.publish(published)

    assert sink.fetch_latest("ledger-a") == published


def test_an_unsigned_checkpoint_comes_back_unsigned(sink):
    sink.publish(make_checkpoint(2))
    fetched = sink.fetch_latest("ledger-a")

    assert (fetched.signature, fetched.key_id) == (None, None)


def test_fetching_from_a_sink_that_has_published_nothing_is_none(sink):
    assert sink.fetch_latest("ledger-a") is None


def test_fetching_a_ledger_the_sink_has_never_seen_is_none(sink):
    sink.publish(make_checkpoint(1, ledger_id="ledger-a"))

    assert sink.fetch_latest("ledger-b") is None


def test_the_newest_checkpoint_wins(sink):
    for sequence in (2, 9, 5):
        sink.publish(make_checkpoint(sequence))

    assert sink.fetch_latest("ledger-a").sequence == 9


def test_checkpoints_for_other_ledgers_are_ignored(sink):
    sink.publish(make_checkpoint(3, ledger_id="ledger-a"))
    sink.publish(make_checkpoint(99, ledger_id="ledger-b"))

    assert sink.fetch_latest("ledger-a").sequence == 3


def test_publishing_appends_rather_than_replacing(sink, anchor_path):
    """An anchor file that can be rewritten is not an anchor."""
    for sequence in (1, 2, 3):
        sink.publish(make_checkpoint(sequence))

    with open(anchor_path, encoding="utf-8") as handle:
        lines = [line for line in handle.read().splitlines() if line]

    assert len(lines) == 3


def test_each_record_is_one_self_describing_json_object(sink, anchor_path):
    """A second implementation must be able to read the anchor without this library."""
    sink.publish(make_checkpoint(4, signature="c2ln", key_id="kid-1"))

    with open(anchor_path, encoding="utf-8") as handle:
        record = json.loads(handle.read().strip())

    assert record["head_hash"] == f"{4:064x}"


def test_a_receipt_records_where_the_checkpoint_landed(sink, anchor_path):
    receipt = sink.publish(make_checkpoint(4))

    assert receipt.location == anchor_path


def test_a_receipt_is_stamped_with_a_timezone_aware_instant(sink):
    receipt = sink.publish(make_checkpoint(4))

    assert receipt.published_at.utcoffset() is not None


def test_publishing_fsyncs_so_the_record_survives_a_power_cut(sink, monkeypatch):
    """An append-only JSONL that is not fsynced is a durability claim the sink cannot keep."""
    import signledger.anchor.sinks as sinks_module

    synced = []
    monkeypatch.setattr(sinks_module.os, "fsync", lambda fd: synced.append(fd))

    sink.publish(make_checkpoint(4))

    assert len(synced) == 1


def test_a_second_reader_sees_a_published_checkpoint_immediately(anchor_path):
    FileAnchorSink(anchor_path).publish(make_checkpoint(6))

    assert FileAnchorSink(anchor_path).fetch_latest("ledger-a").sequence == 6


# --------------------------------------------------------------------------------------
# a damaged anchor is reported, never skipped
# --------------------------------------------------------------------------------------


def test_a_corrupt_record_raises_rather_than_being_skipped(sink, anchor_path):
    """Silently skipping a bad line lets an attacker delete an anchor by corrupting it."""
    sink.publish(make_checkpoint(3))
    with open(anchor_path, "a", encoding="utf-8") as handle:
        handle.write("{not json at all\n")

    with pytest.raises(AnchorError):
        sink.fetch_latest("ledger-a")


def test_a_record_missing_a_checkpoint_field_raises(sink, anchor_path):
    with open(anchor_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps({"ledger_id": "ledger-a"}) + "\n")

    with pytest.raises(AnchorError):
        sink.fetch_latest("ledger-a")


def test_a_record_that_is_not_a_json_object_raises(sink, anchor_path):
    """Matched on the message: a bare ``in`` test against a list would also raise, but for
    the wrong reason, and the shape check could then be deleted with the suite still green.
    """
    with open(anchor_path, "a", encoding="utf-8") as handle:
        handle.write("[1, 2, 3]\n")

    with pytest.raises(AnchorError, match="not an object"):
        sink.fetch_latest("ledger-a")


def test_blank_lines_are_tolerated(sink, anchor_path):
    sink.publish(make_checkpoint(3))
    with open(anchor_path, "a", encoding="utf-8") as handle:
        handle.write("\n\n")

    assert sink.fetch_latest("ledger-a").sequence == 3


def test_publishing_into_a_directory_that_does_not_exist_raises_anchor_error(tmp_path):
    sink = FileAnchorSink(str(tmp_path / "nope" / "anchors.jsonl"))

    with pytest.raises(AnchorError):
        sink.publish(make_checkpoint(1))


# --------------------------------------------------------------------------------------
# the signing payload
# --------------------------------------------------------------------------------------


def test_the_signing_payload_is_deterministic():
    checkpoint = make_checkpoint(5)

    assert checkpoint_signing_payload(checkpoint) == checkpoint_signing_payload(checkpoint)


def test_the_signing_payload_excludes_the_signature_it_will_carry():
    """Otherwise a checkpoint could never be signed: the payload would depend on the
    signature that does not exist yet."""
    unsigned = make_checkpoint(5)
    signed = make_checkpoint(5, signature="c2ln")

    assert checkpoint_signing_payload(unsigned) == checkpoint_signing_payload(signed)


def test_the_signing_payload_covers_the_head_hash():
    left = make_checkpoint(5)
    right = make_checkpoint(5, head_hash="f" * 64)

    assert checkpoint_signing_payload(left) != checkpoint_signing_payload(right)


def test_the_signing_payload_covers_the_entry_count():
    left = make_checkpoint(5)
    right = make_checkpoint(5, entry_count=99)

    assert checkpoint_signing_payload(left) != checkpoint_signing_payload(right)


def test_the_signing_payload_covers_the_key_identity():
    """``key_id`` is inside the payload so an attacker cannot re-point a checkpoint at a
    key they control without invalidating the signature."""
    left = make_checkpoint(5, key_id="kid-1")
    right = make_checkpoint(5, key_id="kid-2")

    assert checkpoint_signing_payload(left) != checkpoint_signing_payload(right)


def test_the_signing_payload_cannot_collide_with_a_genesis_preimage():
    """Both are canonical JSON objects, so they need distinct type tags."""
    payload = checkpoint_signing_payload(make_checkpoint(5))

    assert b'"type":"signledger/checkpoint"' in payload


# --------------------------------------------------------------------------------------
# the protocol
# --------------------------------------------------------------------------------------


def test_the_file_sink_satisfies_the_anchor_sink_protocol(sink):
    assert isinstance(sink, AnchorSink)


def test_the_null_sink_satisfies_the_anchor_sink_protocol():
    assert isinstance(NullAnchorSink(), AnchorSink)


def test_an_unreadable_anchor_path_raises_anchor_error(tmp_path):
    """A directory where a file should be. Failing loud beats reporting 'not anchored'."""
    directory = tmp_path / "anchors.jsonl"
    directory.mkdir()

    with pytest.raises(AnchorError):
        FileAnchorSink(str(directory)).fetch_latest("ledger-a")


def test_a_record_with_an_unreadable_timestamp_raises(sink, anchor_path):
    sink.publish(make_checkpoint(3))
    with open(anchor_path, encoding="utf-8") as handle:
        record = json.loads(handle.read().strip())
    record["timestamp"] = "the third of never"
    with open(anchor_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")

    with pytest.raises(AnchorError):
        sink.fetch_latest("ledger-a")


def test_a_record_with_a_numeric_timestamp_raises(sink, anchor_path):
    """A bare epoch number cannot be read back: its unit is not recoverable."""
    record = dict.fromkeys(
        ("ledger_id", "head_hash", "merkle_root", "signature", "key_id"), "x"
    )
    record.update({"sequence": 1, "entry_count": 1, "format_version": 2, "timestamp": 1717243200})
    with open(anchor_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")

    with pytest.raises(AnchorError):
        sink.fetch_latest("ledger-a")


def test_an_anchor_error_never_quotes_the_path(tmp_path):
    """Contract section 10: a path can carry a deployment layout an operator would rather
    not see in a log."""
    directory = tmp_path / "secret-deployment-layout.jsonl"
    directory.mkdir()

    with pytest.raises(AnchorError) as raised:
        FileAnchorSink(str(directory)).fetch_latest("ledger-a")

    assert "secret-deployment-layout" not in str(raised.value)


# --------------------------------------------------------------------------------------
# package surface
# --------------------------------------------------------------------------------------


def test_the_anchor_package_exports_everything_a_caller_needs():
    import signledger.anchor as anchor

    exported = {
        "AnchorReceipt",
        "AnchorSink",
        "Checkpoint",
        "FileAnchorSink",
        "NullAnchorSink",
        "checkpoint_signing_payload",
    }

    assert exported <= set(anchor.__all__)


def test_the_anchor_package_binds_every_name_it_advertises():
    import signledger.anchor as anchor

    assert all(hasattr(anchor, name) for name in anchor.__all__)
