"""``Ledger`` - the public surface (architecture contract, section 9).

Almost every rule exercised here is a v1 defect written down as an assertion: a ledger
that reported ``integrity_verified = True`` without checking, a cache that served
unverified entries, a subscriber that deadlocked the whole log, and an ``auto_verify``
that was on by default and swallowed the failures it found.
"""

import pytest

from signledger.backends.memory import MemoryBackend
from signledger.backends.sqlite import SQLiteBackend
from signledger.core.exceptions import ConfigurationError
from signledger.core.ledger import Ledger


@pytest.fixture
def ledger():
    instance = Ledger()
    try:
        yield instance
    finally:
        instance.close()


# --------------------------------------------------------------------------------------
# construction
# --------------------------------------------------------------------------------------


def test_a_ledger_can_be_built_with_no_arguments_at_all(ledger):
    assert ledger.count() == 0


def test_a_ledger_without_a_backend_gets_an_in_process_one(ledger):
    assert isinstance(ledger.backend, MemoryBackend)


def test_a_fresh_ledger_invents_its_own_identity(ledger):
    assert len(ledger.ledger_id) == 36


def test_two_fresh_ledgers_do_not_share_an_identity():
    first, second = Ledger(), Ledger()
    try:
        assert first.ledger_id != second.ledger_id
    finally:
        first.close()
        second.close()


def test_an_explicit_ledger_id_is_honoured():
    instance = Ledger(ledger_id="ledger-of-record")
    try:
        assert instance.ledger_id == "ledger-of-record"
    finally:
        instance.close()


def test_an_unsupported_hash_algorithm_is_refused_at_construction():
    with pytest.raises(ConfigurationError):
        Ledger(hash_algorithm="md5")


def test_signatures_without_a_signer_are_refused_at_construction():
    """v1 raised on every append instead, so the misconfiguration surfaced in production."""
    with pytest.raises(ConfigurationError):
        Ledger(enable_signatures=True)


def test_auto_verify_without_a_failure_handler_is_refused_at_construction():
    """v1 defaulted auto_verify on and swallowed IntegrityError into a log line."""
    with pytest.raises(ConfigurationError):
        Ledger(auto_verify=True)


def test_auto_verify_is_off_by_default(ledger):
    assert ledger.auto_verify is False


def test_a_negative_cache_size_is_refused():
    with pytest.raises(ConfigurationError):
        Ledger(max_cache_entries=-1)


def test_a_non_positive_verify_interval_is_refused():
    with pytest.raises(ConfigurationError):
        Ledger(auto_verify=True, on_integrity_failure=lambda report: None, verify_interval=0)


# --------------------------------------------------------------------------------------
# identity of an existing store
# --------------------------------------------------------------------------------------


def test_reopening_a_store_adopts_the_identity_already_written_there(tmp_path):
    """Inventing a new id would make entry 0 fail its genesis check forever."""
    path = str(tmp_path / "ledger.db")
    first = Ledger(SQLiteBackend(path))
    first.append({"event": "one"})
    original = first.ledger_id
    first.close()

    second = Ledger(SQLiteBackend(path))
    try:
        assert second.ledger_id == original
    finally:
        second.close()


def test_an_explicit_id_that_contradicts_the_store_is_refused(tmp_path):
    path = str(tmp_path / "ledger.db")
    first = Ledger(SQLiteBackend(path))
    first.append({"event": "one"})
    first.close()

    with pytest.raises(ConfigurationError):
        Ledger(SQLiteBackend(path), ledger_id="a-different-ledger")


# --------------------------------------------------------------------------------------
# append
# --------------------------------------------------------------------------------------


def test_the_first_append_lands_at_sequence_zero(ledger):
    assert ledger.append({"event": "created"}).sequence == 0


def test_each_append_advances_the_sequence_by_one(ledger):
    sequences = [ledger.append({"n": index}).sequence for index in range(4)]

    assert sequences == [0, 1, 2, 3]


def test_each_entry_links_to_the_digest_of_the_one_before_it(ledger):
    first = ledger.append({"n": 0})
    second = ledger.append({"n": 1})

    assert second.previous_hash == first.hash


def test_entry_zero_links_to_the_genesis_derived_from_its_own_timestamp(ledger):
    """Entry 0's timestamp *is* the ledger's creation instant, so a reopened ledger can
    recompute the genesis hash from the store alone. Both inputs are inside the preimage,
    so neither can be moved without breaking the digest."""
    from signledger.crypto.hashing import genesis_hash

    first = ledger.append({"event": "created"})

    assert first.previous_hash == genesis_hash(ledger.ledger_id, first.timestamp)


def test_an_appended_entry_carries_the_payload_it_was_given(ledger):
    assert ledger.append({"event": "created", "n": 7}).data == {"event": "created", "n": 7}


def test_metadata_defaults_to_an_empty_mapping(ledger):
    assert ledger.append({"event": "created"}).metadata == {}


def test_metadata_is_stored_when_it_is_given(ledger):
    entry = ledger.append({"event": "created"}, {"actor": "svc"})

    assert entry.metadata == {"actor": "svc"}


def test_the_timestamp_defaults_to_a_timezone_aware_instant(ledger):
    assert ledger.append({"event": "created"}).timestamp.utcoffset() is not None


def test_an_explicit_timestamp_is_honoured(ledger):
    from datetime import datetime, timezone

    moment = datetime(2024, 5, 5, 6, 7, 8, tzinfo=timezone.utc)

    assert ledger.append({"event": "created"}, timestamp=moment).timestamp == moment


def test_a_timestamp_that_is_not_a_datetime_is_refused(ledger):
    """An ISO string would be accepted by Entry but would derive a different genesis hash,
    so a reopened ledger would report GENESIS_MISMATCH on a pristine chain."""
    from signledger.core.exceptions import ValidationError

    with pytest.raises(ValidationError):
        ledger.append({"event": "created"}, timestamp="2024-05-05T06:07:08+00:00")


def test_each_entry_gets_its_own_identifier(ledger):
    identifiers = {ledger.append({"n": index}).entry_id for index in range(5)}

    assert len(identifiers) == 5


def test_an_explicit_entry_id_is_honoured(ledger):
    assert ledger.append({"event": "created"}, entry_id="entry-of-record").entry_id == (
        "entry-of-record"
    )


def test_an_empty_payload_is_refused(ledger):
    from signledger.core.exceptions import ValidationError

    with pytest.raises(ValidationError):
        ledger.append({})


def test_an_appended_entry_is_sealed_with_a_verifiable_digest(ledger):
    from signledger.crypto.hashing import HashChain

    assert HashChain().verify_hash(ledger.append({"event": "created"}))


def test_an_appended_entry_is_actually_in_the_store(ledger):
    appended = ledger.append({"event": "created"})

    assert ledger.backend.get_by_sequence(0) == appended


# --------------------------------------------------------------------------------------
# integrity
# --------------------------------------------------------------------------------------


@pytest.fixture
def signer():
    pytest.importorskip("cryptography", reason="cryptography is not installed")
    from signledger.crypto.signatures import Ed25519Signer

    return Ed25519Signer.generate()


@pytest.fixture
def signed_ledger(signer):
    instance = Ledger(enable_signatures=True, signer=signer)
    try:
        yield instance
    finally:
        instance.close()


def test_a_freshly_written_ledger_verifies(ledger):
    for index in range(5):
        ledger.append({"n": index})

    assert ledger.check_integrity().ok is True


def test_a_reopened_ledger_still_verifies(tmp_path):
    """The genesis anchor has to be recomputable from the store alone, or every restart
    would report a pristine ledger as tampered."""
    path = str(tmp_path / "ledger.db")
    first = Ledger(SQLiteBackend(path))
    for index in range(4):
        first.append({"n": index})
    first.close()

    second = Ledger(SQLiteBackend(path))
    try:
        assert second.check_integrity().ok is True
    finally:
        second.close()


def test_a_reopened_ledger_can_be_appended_to_and_still_verifies(tmp_path):
    path = str(tmp_path / "ledger.db")
    first = Ledger(SQLiteBackend(path))
    first.append({"n": 0})
    first.close()

    second = Ledger(SQLiteBackend(path))
    try:
        second.append({"n": 1})
        assert second.check_integrity().ok is True
    finally:
        second.close()


def test_an_empty_ledger_never_reports_integrity(ledger):
    assert ledger.check_integrity().ok is False


def test_verification_covers_every_entry(ledger):
    for index in range(6):
        ledger.append({"n": index})

    assert ledger.check_integrity().entries_verified == 6


def test_verification_can_start_part_way_through(ledger):
    for index in range(6):
        ledger.append({"n": index})

    report = ledger.check_integrity(from_sequence=4)

    assert (report.ok, report.entries_verified) == (True, 2)


def test_an_unsigned_ledger_earns_chain_only(ledger):
    from signledger.core.verification import AssuranceLevel

    ledger.append({"n": 0})

    assert ledger.assurance_level() is AssuranceLevel.CHAIN_ONLY


def test_a_signed_ledger_verifies_its_own_signatures(signed_ledger):
    for index in range(3):
        signed_ledger.append({"n": index})

    assert signed_ledger.check_integrity().ok is True


def test_a_signed_ledger_earns_signed_assurance(signed_ledger):
    from signledger.core.verification import AssuranceLevel

    signed_ledger.append({"n": 0})

    assert signed_ledger.assurance_level() is AssuranceLevel.SIGNED


def test_a_signed_entry_names_the_key_that_signed_it(signed_ledger, signer):
    entry = signed_ledger.append({"n": 0})

    assert (entry.key_id, entry.signature_algorithm) == (signer.key_id, signer.algorithm)


def test_a_signed_entry_actually_carries_a_signature(signed_ledger):
    assert signed_ledger.append({"n": 0}).signature is not None


def test_the_signing_identity_is_covered_by_the_entry_digest(signed_ledger, signer):
    """Contract section 6: swapping key_id must break the hash, not merely the signature."""
    from signledger.crypto.hashing import HashChain

    entry = signed_ledger.append({"n": 0})
    swapped = entry.model_copy(update={"key_id": "someone-elses-key"})

    assert not HashChain().verify_hash(swapped)


def test_a_key_resolver_that_knows_nothing_makes_signatures_fail_closed(signer):
    from signledger.core.verification import FailureKind
    from signledger.crypto.keys import InMemoryKeyResolver

    instance = Ledger(enable_signatures=True, signer=signer, key_resolver=InMemoryKeyResolver())
    try:
        instance.append({"n": 0})
        report = instance.check_integrity()
        assert [f.kind for f in report.failures] == [FailureKind.UNKNOWN_KEY]
    finally:
        instance.close()


# --------------------------------------------------------------------------------------
# the boolean wrapper
# --------------------------------------------------------------------------------------


def test_verify_integrity_is_true_for_a_healthy_ledger(ledger):
    ledger.append({"n": 0})

    assert ledger.verify_integrity() is True


def test_verify_integrity_raises_for_a_damaged_ledger(ledger):
    from signledger.core.exceptions import IntegrityError

    ledger.append({"n": 0})
    ledger.backend._entries[0] = ledger.backend._entries[0].model_copy(
        update={"data": {"event": "forged"}}
    )

    with pytest.raises(IntegrityError):
        ledger.verify_integrity()


def test_verify_integrity_raises_on_an_empty_ledger(ledger):
    from signledger.core.exceptions import IntegrityError

    with pytest.raises(IntegrityError):
        ledger.verify_integrity()


def test_an_integrity_error_never_quotes_a_payload(ledger):
    from signledger.core.exceptions import IntegrityError

    ledger.append({"card_number": "4111111111111111"})
    ledger.backend._entries[0] = ledger.backend._entries[0].model_copy(
        update={"data": {"card_number": "4111111111111111", "extra": "forged"}}
    )

    with pytest.raises(IntegrityError) as raised:
        ledger.verify_integrity()

    assert "4111111111111111" not in str(raised.value)


# --------------------------------------------------------------------------------------
# reads
# --------------------------------------------------------------------------------------


def test_an_entry_can_be_read_back_by_its_position(ledger):
    appended = ledger.append({"n": 0})

    assert ledger.get_by_sequence(0) == appended


def test_reading_past_the_head_is_empty(ledger):
    ledger.append({"n": 0})

    assert ledger.get_by_sequence(99) is None


def test_an_entry_can_be_read_back_by_its_identifier(ledger):
    appended = ledger.append({"n": 0})

    assert ledger.get_entry(appended.entry_id) == appended


def test_reading_an_unknown_identifier_is_empty(ledger):
    ledger.append({"n": 0})

    assert ledger.get_entry("no-such-entry") is None


def test_entries_iterate_in_sequence_order(ledger):
    for index in range(5):
        ledger.append({"n": index})

    assert [e.sequence for e in ledger.iter_entries()] == [0, 1, 2, 3, 4]


def test_iteration_passes_its_window_through_to_the_store(ledger):
    for index in range(6):
        ledger.append({"n": index})

    assert [e.sequence for e in ledger.iter_entries(start_seq=2, limit=2)] == [2, 3]


def test_the_head_is_the_newest_entry(ledger):
    ledger.append({"n": 0})
    newest = ledger.append({"n": 1})

    assert ledger.head() == newest


def test_an_empty_ledger_has_no_head(ledger):
    assert ledger.head() is None


# --------------------------------------------------------------------------------------
# the cache
# --------------------------------------------------------------------------------------


def test_a_recently_appended_entry_is_served_from_the_cache(ledger):
    """Proved by removing the row from the store: only a cache could still answer."""
    appended = ledger.append({"n": 0})
    ledger.backend._entries.clear()

    assert ledger.get_by_sequence(0) == appended


def test_a_cached_entry_whose_digest_no_longer_matches_is_evicted(ledger):
    """v1 served whatever was in the cache, so an in-process corruption was invisible."""
    appended = ledger.append({"n": 0})
    ledger._cache[0] = appended.model_copy(update={"data": {"n": "forged"}})

    assert ledger.get_by_sequence(0) == appended


def test_a_cached_entry_in_an_unhashable_format_is_evicted(ledger):
    appended = ledger.append({"n": 0})
    ledger._cache[0] = appended.model_copy(update={"format_version": 99})

    assert ledger.get_by_sequence(0) == appended


def test_a_cached_entry_is_also_reachable_by_identifier(ledger):
    appended = ledger.append({"n": 0})
    ledger.backend._entries.clear()
    ledger.backend._by_entry_id.clear()

    assert ledger.get_entry(appended.entry_id) == appended


def test_the_cache_never_grows_past_its_limit():
    instance = Ledger(max_cache_entries=3)
    try:
        for index in range(10):
            instance.append({"n": index})
        assert len(instance._cache) <= 3
    finally:
        instance.close()


def test_the_cache_keeps_the_newest_entries():
    instance = Ledger(max_cache_entries=3)
    try:
        for index in range(10):
            instance.append({"n": index})
        assert set(instance._cache) == {7, 8, 9}
    finally:
        instance.close()


def test_a_reopened_ledger_warms_its_cache_with_the_newest_entries(tmp_path):
    """v1 loaded the *oldest* N, so the cache never held anything anyone was reading."""
    path = str(tmp_path / "ledger.db")
    first = Ledger(SQLiteBackend(path))
    for index in range(10):
        first.append({"n": index})
    first.close()

    second = Ledger(SQLiteBackend(path), max_cache_entries=3)
    try:
        assert set(second._cache) == {7, 8, 9}
    finally:
        second.close()


def test_caching_can_be_turned_off_entirely():
    instance = Ledger(max_cache_entries=0)
    try:
        instance.append({"n": 0})
        assert instance._cache == {}
    finally:
        instance.close()


def test_a_ledger_with_caching_off_still_reads_from_the_store():
    instance = Ledger(max_cache_entries=0)
    try:
        appended = instance.append({"n": 0})
        assert instance.get_by_sequence(0) == appended
    finally:
        instance.close()


def test_the_cache_honours_a_byte_budget():
    instance = Ledger(max_cache_entries=1000, max_cache_bytes=400)
    try:
        for index in range(20):
            instance.append({"n": index, "filler": "x" * 100})
        assert len(instance._cache) < 20
    finally:
        instance.close()


def test_verification_ignores_the_cache_entirely(ledger):
    """The cached copy is pristine; the stored row is not. v1 reported clean."""
    appended = ledger.append({"n": 0})
    ledger.backend._entries[0] = appended.model_copy(update={"data": {"n": "forged"}})

    assert ledger.check_integrity().ok is False


# --------------------------------------------------------------------------------------
# checkpoints
# --------------------------------------------------------------------------------------


def test_a_checkpoint_records_the_current_head(ledger):
    for index in range(4):
        ledger.append({"n": index})
    head = ledger.head()

    checkpoint = ledger.checkpoint()

    assert (checkpoint.sequence, checkpoint.head_hash) == (head.sequence, head.hash)


def test_a_checkpoint_records_how_many_entries_there_were(ledger):
    for index in range(4):
        ledger.append({"n": index})

    assert ledger.checkpoint().entry_count == 4


def test_a_checkpoint_names_the_ledger_it_describes(ledger):
    ledger.append({"n": 0})

    assert ledger.checkpoint().ledger_id == ledger.ledger_id


def test_a_checkpoint_carries_a_merkle_root_over_every_entry(ledger):
    """Bound to the leaf count, so the root cannot be replayed for a shorter chain."""
    from signledger.crypto.merkle import MerkleTree

    for index in range(5):
        ledger.append({"n": index})
    expected = MerkleTree(bytes.fromhex(e.hash) for e in ledger.iter_entries()).root

    assert ledger.checkpoint().merkle_root == expected


def test_a_checkpoint_is_stamped_with_a_timezone_aware_instant(ledger):
    ledger.append({"n": 0})

    assert ledger.checkpoint().timestamp.utcoffset() is not None


def test_a_checkpoint_is_persisted_to_the_store(ledger):
    ledger.append({"n": 0})

    created = ledger.checkpoint()

    assert ledger.backend.get_latest_checkpoint() == created


def test_checkpointing_an_empty_ledger_is_refused(ledger):
    from signledger.core.exceptions import ValidationError

    with pytest.raises(ValidationError):
        ledger.checkpoint()


def test_a_checkpoint_is_unsigned_when_the_ledger_is(ledger):
    ledger.append({"n": 0})

    assert ledger.checkpoint().signature is None


def test_a_checkpoint_is_signed_when_the_ledger_signs(signed_ledger):
    signed_ledger.append({"n": 0})

    assert signed_ledger.checkpoint().signature is not None


def test_a_checkpoint_names_the_key_that_signed_it(signed_ledger, signer):
    signed_ledger.append({"n": 0})

    assert signed_ledger.checkpoint().key_id == signer.key_id


def test_a_checkpoint_signature_verifies_over_its_own_payload(signed_ledger, signer):
    from signledger.anchor.sinks import checkpoint_signing_payload
    from signledger.crypto.signatures import verify

    signed_ledger.append({"n": 0})
    checkpoint = signed_ledger.checkpoint()

    assert verify(checkpoint_signing_payload(checkpoint), checkpoint.signature, signer.public_key)


def test_a_checkpoint_is_published_to_the_anchor_sink(tmp_path):
    from signledger.anchor.sinks import FileAnchorSink

    path = str(tmp_path / "anchors.jsonl")
    instance = Ledger(anchor_sink=FileAnchorSink(path))
    try:
        instance.append({"n": 0})
        created = instance.checkpoint()
        assert FileAnchorSink(path).fetch_latest(instance.ledger_id) == created
    finally:
        instance.close()


def test_verification_uses_the_checkpoint_that_was_taken(ledger):
    ledger.append({"n": 0})
    created = ledger.checkpoint()

    assert ledger.check_integrity().checkpoint_used == created


def test_verification_prefers_the_externally_anchored_checkpoint(tmp_path, signer):
    """A locally stored checkpoint sits in the same store an insider can truncate."""
    from signledger.anchor.sinks import FileAnchorSink

    path = str(tmp_path / "anchors.jsonl")
    sink = FileAnchorSink(path)
    instance = Ledger(anchor_sink=sink)
    try:
        instance.append({"n": 0})
        anchored = instance.checkpoint()
        instance.backend._checkpoints.clear()

        assert instance.check_integrity().checkpoint_used == anchored
    finally:
        instance.close()


def test_a_signed_and_anchored_ledger_earns_anchored_assurance(tmp_path, signer):
    from signledger.anchor.sinks import FileAnchorSink
    from signledger.core.verification import AssuranceLevel

    instance = Ledger(
        enable_signatures=True,
        signer=signer,
        anchor_sink=FileAnchorSink(str(tmp_path / "anchors.jsonl")),
    )
    try:
        instance.append({"n": 0})
        instance.checkpoint()
        assert instance.assurance_level() is AssuranceLevel.ANCHORED
    finally:
        instance.close()


def test_a_signed_anchored_append_only_store_earns_worm(tmp_path, signer):
    from signledger.anchor.sinks import FileAnchorSink
    from signledger.core.verification import AssuranceLevel

    instance = Ledger(
        SQLiteBackend(str(tmp_path / "ledger.db")),
        enable_signatures=True,
        signer=signer,
        anchor_sink=FileAnchorSink(str(tmp_path / "anchors.jsonl")),
    )
    try:
        instance.append({"n": 0})
        instance.checkpoint()
        assert instance.assurance_level() is AssuranceLevel.WORM
    finally:
        instance.close()


def test_a_null_sink_never_confers_anchored_assurance(signer):
    """It keeps nothing, so there is nothing outside the process to compare against."""
    from signledger.anchor.sinks import NullAnchorSink
    from signledger.core.verification import AssuranceLevel

    instance = Ledger(enable_signatures=True, signer=signer, anchor_sink=NullAnchorSink())
    try:
        instance.append({"n": 0})
        instance.checkpoint()
        assert instance.assurance_level() is AssuranceLevel.SIGNED
    finally:
        instance.close()


# --------------------------------------------------------------------------------------
# subscribers - notified off the write path (contract section 9)
# --------------------------------------------------------------------------------------


def collector(seen, event=None):
    def _callback(entry):
        seen.append(entry)
        if event is not None and len(seen) >= event[1]:
            event[0].set()

    return _callback


def test_a_subscriber_is_told_about_an_appended_entry(ledger):
    import threading

    seen, arrived = [], threading.Event()
    ledger.subscribe(collector(seen, (arrived, 1)))
    appended = ledger.append({"n": 0})

    assert arrived.wait(timeout=5) and seen == [appended]


def test_every_subscriber_is_told(ledger):
    import threading

    first, second = [], []
    both = threading.Event()
    ledger.subscribe(collector(first))
    ledger.subscribe(collector(second, (both, 1)))
    ledger.append({"n": 0})
    both.wait(timeout=5)

    assert len(first) == 1


def test_an_unsubscribed_callback_stops_hearing_about_entries(ledger):
    import threading

    seen, arrived = [], threading.Event()
    callback = collector(seen, (arrived, 1))
    ledger.subscribe(callback)
    ledger.append({"n": 0})
    arrived.wait(timeout=5)

    ledger.unsubscribe(callback)
    ledger.append({"n": 1})
    ledger.close()

    assert len(seen) == 1


def test_unsubscribing_something_that_never_subscribed_is_harmless(ledger):
    ledger.unsubscribe(lambda entry: None)


def test_a_subscriber_that_appends_does_not_deadlock(ledger):
    """The v1 killer: subscribers were called while the write lock was held, so the first
    subscriber that appended froze the ledger permanently."""
    import threading

    finished = threading.Event()
    nested = []

    def subscriber(entry):
        if not nested:
            nested.append(entry)
            ledger.append({"event": "from-inside-a-subscriber"})
        else:
            finished.set()

    ledger.subscribe(subscriber)
    ledger.append({"event": "first"})

    assert finished.wait(timeout=10)


def test_a_subscriber_that_raises_does_not_silence_the_others(ledger):
    import threading

    seen, arrived = [], threading.Event()

    def explodes(entry):
        raise RuntimeError("subscriber bug")

    ledger.subscribe(explodes)
    ledger.subscribe(collector(seen, (arrived, 2)))
    ledger.append({"n": 0})
    ledger.append({"n": 1})

    assert arrived.wait(timeout=5)


def test_a_subscriber_failure_is_counted_rather_than_swallowed(ledger):
    import threading

    arrived = threading.Event()

    def explodes(entry):
        arrived.set()
        raise RuntimeError("subscriber bug")

    ledger.subscribe(explodes)
    ledger.append({"n": 0})
    arrived.wait(timeout=5)
    ledger.close()

    assert ledger.get_stats_snapshot()["subscriber_errors"] == 1


def test_notifications_are_dropped_rather_than_blocking_the_writer():
    """A bounded queue must shed load; blocking here would stall the append path."""
    import threading

    release = threading.Event()
    instance = Ledger(notify_queue_size=1)
    try:
        instance.subscribe(lambda entry: release.wait(timeout=5))
        for index in range(40):
            instance.append({"n": index})
        release.set()
        assert instance.get_stats_snapshot()["dropped_notifications"] > 0
    finally:
        release.set()
        instance.close()


def test_a_ledger_with_no_subscribers_starts_no_notifier_thread(ledger):
    ledger.append({"n": 0})

    assert ledger._notifier is None


# --------------------------------------------------------------------------------------
# statistics - measured, never assumed
# --------------------------------------------------------------------------------------


def test_stats_report_the_real_entry_count(ledger):
    for index in range(3):
        ledger.append({"n": index})

    assert ledger.get_stats()["entry_count"] == 3


def test_stats_report_integrity_as_measured(ledger):
    ledger.append({"n": 0})

    assert ledger.get_stats()["integrity_ok"] is True


def test_stats_report_a_damaged_ledger_as_damaged(ledger):
    """v1 hardcoded integrity_verified = True, so this number meant nothing to an auditor."""
    appended = ledger.append({"n": 0})
    ledger.backend._entries[0] = appended.model_copy(update={"data": {"n": "forged"}})

    assert ledger.get_stats()["integrity_ok"] is False


def test_stats_name_the_first_failure(ledger):
    appended = ledger.append({"n": 0})
    ledger.backend._entries[0] = appended.model_copy(update={"data": {"n": "forged"}})

    assert ledger.get_stats()["last_failure"] is not None


def test_stats_have_no_failure_to_report_on_a_healthy_ledger(ledger):
    ledger.append({"n": 0})

    assert ledger.get_stats()["last_failure"] is None


def test_stats_record_when_verification_last_ran(ledger):
    ledger.append({"n": 0})

    assert ledger.get_stats()["last_verification_time"].utcoffset() is not None


def test_stats_report_the_assurance_level(ledger):
    from signledger.core.verification import AssuranceLevel

    ledger.append({"n": 0})

    assert ledger.get_stats()["assurance_level"] is AssuranceLevel.CHAIN_ONLY


def test_stats_report_the_backend_capabilities(ledger):
    assert "atomic_append" in ledger.get_stats()["capabilities"]


def test_stats_name_the_anchor_sink_in_use(tmp_path):
    from signledger.anchor.sinks import FileAnchorSink

    instance = Ledger(anchor_sink=FileAnchorSink(str(tmp_path / "a.jsonl")))
    try:
        assert instance.get_stats()["anchor_sink"] == "file"
    finally:
        instance.close()


def test_stats_report_no_anchor_sink_when_there_is_none(ledger):
    assert ledger.get_stats()["anchor_sink"] is None


def test_stats_propagate_a_backend_failure_instead_of_reporting_zeros(ledger):
    """v1 caught the error and returned healthy-looking zeros."""
    from signledger.core.exceptions import StorageError

    ledger.append({"n": 0})
    ledger.backend.close()

    with pytest.raises(StorageError):
        ledger.get_stats()


# --------------------------------------------------------------------------------------
# shutdown
# --------------------------------------------------------------------------------------


def test_close_is_idempotent(ledger):
    ledger.close()
    ledger.close()


def test_close_closes_the_backend(ledger):
    from signledger.core.exceptions import StorageError

    ledger.close()

    with pytest.raises(StorageError):
        ledger.backend.count()


def test_close_stops_the_notifier_thread(ledger):
    ledger.subscribe(lambda entry: None)
    notifier = ledger._notifier
    ledger.close()

    assert not notifier.is_alive()


def test_a_ledger_can_be_used_as_a_context_manager():
    from signledger.core.exceptions import StorageError

    with Ledger() as instance:
        instance.append({"n": 0})

    with pytest.raises(StorageError):
        instance.backend.count()


# --------------------------------------------------------------------------------------
# automatic verification
# --------------------------------------------------------------------------------------


def test_auto_verify_reports_a_break_to_the_handler():
    import threading

    called = threading.Event()
    reports = []

    def handler(report):
        reports.append(report)
        called.set()

    instance = Ledger(auto_verify=True, on_integrity_failure=handler, verify_interval=0.01)
    try:
        appended = instance.append({"n": 0})
        instance.backend._entries[0] = appended.model_copy(update={"data": {"n": "forged"}})
        assert called.wait(timeout=10)
    finally:
        instance.close()


def test_auto_verify_leaves_a_healthy_ledger_alone():
    import time

    reports = []
    instance = Ledger(
        auto_verify=True, on_integrity_failure=reports.append, verify_interval=0.01
    )
    try:
        instance.append({"n": 0})
        time.sleep(0.2)
        assert reports == []
    finally:
        instance.close()


def test_a_handler_that_raises_does_not_kill_the_verifier():
    import threading

    calls = []
    seen_twice = threading.Event()

    def handler(report):
        calls.append(report)
        if len(calls) >= 2:
            seen_twice.set()
        raise RuntimeError("handler bug")

    instance = Ledger(auto_verify=True, on_integrity_failure=handler, verify_interval=0.01)
    try:
        appended = instance.append({"n": 0})
        instance.backend._entries[0] = appended.model_copy(update={"data": {"n": "forged"}})
        assert seen_twice.wait(timeout=10)
    finally:
        instance.close()


def test_close_stops_the_verifier_thread():
    instance = Ledger(auto_verify=True, on_integrity_failure=lambda r: None, verify_interval=0.01)
    instance.append({"n": 0})
    verifier = instance._verifier
    instance.close()

    assert not verifier.is_alive()


# --------------------------------------------------------------------------------------
# remaining guards
# --------------------------------------------------------------------------------------


def test_a_negative_cache_byte_budget_is_refused():
    with pytest.raises(ConfigurationError):
        Ledger(max_cache_bytes=-1)


def test_an_empty_notification_queue_is_refused():
    """A zero-length queue would drop every notification while looking configured."""
    with pytest.raises(ConfigurationError):
        Ledger(notify_queue_size=0)


def test_a_stale_identifier_index_falls_through_to_the_store(ledger):
    """The identifier index is a hint, not an authority. If it points at a slot holding a
    different record, the answer comes from the store."""
    ledger.append({"n": 0})
    ledger._cache_ids["an-identifier-that-was-never-written"] = 0

    assert ledger.get_entry("an-identifier-that-was-never-written") is None


def test_the_derived_key_resolver_is_built_once_and_reused(signed_ledger):
    signed_ledger.append({"n": 0})
    signed_ledger.check_integrity()
    first = signed_ledger._derived_resolver

    signed_ledger.check_integrity()

    assert signed_ledger._derived_resolver is first


def test_the_verifier_survives_a_backend_that_goes_away():
    """A verifier thread that dies on the first error is a monitor that silently stops
    monitoring - which is the failure mode auto_verify exists to prevent."""
    import time

    instance = Ledger(auto_verify=True, on_integrity_failure=lambda report: None,
                      verify_interval=0.01)
    try:
        instance.append({"n": 0})
        instance.backend.close()

        deadline = time.time() + 10
        while time.time() < deadline:
            if instance.get_stats_snapshot()["last_verify_error"] is not None:
                break
            time.sleep(0.01)

        assert instance.get_stats_snapshot()["last_verify_error"] == "StorageError"
        assert instance._verifier.is_alive()
    finally:
        instance.close()


# --------------------------------------------------------------------------------------
# cold start: several processes, one empty store
# --------------------------------------------------------------------------------------


def _two_cold_started_ledgers(tmp_path):
    """Two ledgers opened against the *same empty* store, each having minted its own id.

    This is a `gunicorn -w 4` cold start, made deterministic: the race in production is
    only about which worker reaches entry zero first, and here that is decided by the
    order the test appends in rather than by the scheduler.
    """
    path = str(tmp_path / "ledger.db")
    return Ledger(SQLiteBackend(path)), Ledger(SQLiteBackend(path))


def test_two_cold_started_workers_start_out_disagreeing_about_the_identity(tmp_path):
    """The premise of the whole defect: against an empty store there is nothing to adopt,
    so every process invents a different uuid4."""
    first, second = _two_cold_started_ledgers(tmp_path)
    try:
        assert first.ledger_id != second.ledger_id
    finally:
        first.close()
        second.close()


def test_a_worker_that_loses_the_race_to_entry_zero_can_still_append(tmp_path):
    """Measured on 2.0.0: Flask + `gunicorn -w 4` + SQLite served 129 of 300 requests and
    FastAPI + `uvicorn --workers 4` + PostgreSQL served exactly 75 of 300 - one worker
    alive out of four. The three losers minted an id nobody had written, found a tip that
    disagreed, and raised ConfigurationError on every append for the life of the process.
    Nothing was actually wrong: the store simply already had a name."""
    first, second = _two_cold_started_ledgers(tmp_path)
    try:
        second.append({"event": "second-wins-the-race"})

        assert first.append({"event": "first-arrives-late"}).sequence == 1
    finally:
        first.close()
        second.close()


def test_the_loser_adopts_the_identity_the_store_ended_up_with(tmp_path):
    first, second = _two_cold_started_ledgers(tmp_path)
    try:
        second.append({"event": "second-wins-the-race"})
        first.append({"event": "first-arrives-late"})

        assert first.ledger_id == second.ledger_id
    finally:
        first.close()
        second.close()


def test_the_repaired_workers_leave_exactly_one_contiguous_chain(tmp_path):
    """The repair is only worth anything if the result is still one verifiable log."""
    first, second = _two_cold_started_ledgers(tmp_path)
    try:
        second.append({"n": 0})
        first.append({"n": 1})
        second.append({"n": 2})
        first.append({"n": 3})

        entries = list(first.iter_entries())
        assert [entry.sequence for entry in entries] == [0, 1, 2, 3]
        assert len({entry.ledger_id for entry in entries}) == 1
        assert first.check_integrity().ok
        assert second.check_integrity().ok
    finally:
        first.close()
        second.close()


def test_the_repair_is_not_a_property_of_one_backend():
    """Same story on any store: it is the ledger's identity that was frozen too early."""
    store = MemoryBackend()
    first, second = Ledger(store), Ledger(store)
    try:
        second.append({"n": 0})
        first.append({"n": 1})

        assert first.ledger_id == second.ledger_id
        assert first.check_integrity().ok
    finally:
        first.close()
        second.close()


def test_a_ledger_id_the_caller_chose_is_never_silently_replaced(tmp_path):
    """The fail-closed refusal is correct and stays. An operator who named this ledger
    meant it, and adopting some other chain's identity would splice two logs into one
    while every hash in the result stayed perfectly valid."""
    path = str(tmp_path / "ledger.db")
    writer = Ledger(SQLiteBackend(path))
    claimant = Ledger(SQLiteBackend(path), ledger_id="ledger-of-record")
    try:
        writer.append({"event": "first"})

        with pytest.raises(ConfigurationError):
            claimant.append({"event": "second"})
    finally:
        claimant.close()
        writer.close()


def test_a_refused_ledger_keeps_the_identity_it_was_given(tmp_path):
    path = str(tmp_path / "ledger.db")
    writer = Ledger(SQLiteBackend(path))
    claimant = Ledger(SQLiteBackend(path), ledger_id="ledger-of-record")
    try:
        writer.append({"event": "first"})
        with pytest.raises(ConfigurationError):
            claimant.append({"event": "second"})

        assert claimant.ledger_id == "ledger-of-record"
    finally:
        claimant.close()
        writer.close()


def test_a_ledger_that_has_already_written_refuses_a_foreign_tip():
    """The repair is for a worker that never wrote anything under its own name. A ledger
    whose own entries are already in the store is looking at a chain that forked
    underneath it, which is an integrity event and must stay loud."""
    from datetime import datetime, timezone

    from signledger.backends.base import build_next_entry
    from signledger.crypto.hashing import HashChain

    store = MemoryBackend()
    instance = Ledger(store)
    try:
        instance.append({"n": 0})
        store.append_atomic(
            lambda tip: build_next_entry(
                tip,
                ledger_id="a-ledger-that-is-not-this-one",
                created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
                data={"n": 1},
                chain=HashChain(),
            )
        )

        with pytest.raises(ConfigurationError):
            instance.append({"n": 2})
    finally:
        instance.close()


def test_the_refusal_to_continue_a_foreign_chain_names_the_remedy(tmp_path):
    """2.0.0 said only that it was refusing. An operator watching three of four workers
    return 500 had nothing telling them to configure a stable ledger_id."""
    path = str(tmp_path / "ledger.db")
    writer = Ledger(SQLiteBackend(path))
    claimant = Ledger(SQLiteBackend(path), ledger_id="ledger-of-record")
    try:
        writer.append({"event": "first"})

        with pytest.raises(ConfigurationError) as raised:
            claimant.append({"event": "second"})

        assert "ledger_id=" in str(raised.value)
        assert "every process" in str(raised.value)
    finally:
        claimant.close()
        writer.close()


def test_the_construction_time_refusal_names_the_remedy(tmp_path):
    path = str(tmp_path / "ledger.db")
    writer = Ledger(SQLiteBackend(path))
    writer.append({"event": "first"})
    writer.close()

    with pytest.raises(ConfigurationError) as raised:
        Ledger(SQLiteBackend(path), ledger_id="a-different-ledger")

    assert "ledger_id=" in str(raised.value)
    assert "every process" in str(raised.value)


def test_a_refusal_to_continue_a_foreign_chain_never_quotes_a_payload(tmp_path):
    """Contract section 10: no exception message may carry entry payload data."""
    path = str(tmp_path / "ledger.db")
    writer = Ledger(SQLiteBackend(path))
    claimant = Ledger(SQLiteBackend(path), ledger_id="ledger-of-record")
    try:
        writer.append({"event": "first"})

        with pytest.raises(ConfigurationError) as raised:
            claimant.append({"cardholder": "4111111111111111"})

        assert "4111111111111111" not in str(raised.value)
        assert "cardholder" not in str(raised.value)
    finally:
        claimant.close()
        writer.close()


# --------------------------------------------------------------------------------------
# a construction that fails still releases the store
# --------------------------------------------------------------------------------------


def test_a_construction_refused_by_the_store_closes_the_store(tmp_path):
    """close() owns the backend, so a constructor that raises must not be the one path
    that leaks it. Measured on 2.0.0: 227 descriptors held open across 1000 failed
    constructions, because nothing ever closed the connection the ledger had adopted."""
    from signledger.core.exceptions import StorageError

    path = str(tmp_path / "ledger.db")
    seed = Ledger(SQLiteBackend(path))
    seed.append({"n": 0})
    seed.close()

    orphan = SQLiteBackend(path)
    with pytest.raises(ConfigurationError):
        Ledger(orphan, ledger_id="a-different-ledger")

    with pytest.raises(StorageError):
        orphan.count()


def test_a_construction_refused_by_its_own_arguments_closes_the_store():
    """The argument checks run before a single entry is written, but the store handed in
    is already open by then - and only the ledger is left holding it."""
    from signledger.core.exceptions import StorageError

    store = MemoryBackend()
    with pytest.raises(ConfigurationError):
        Ledger(store, enable_signatures=True)

    with pytest.raises(StorageError):
        store.count()


def test_a_construction_refused_for_its_hash_algorithm_closes_the_store():
    from signledger.core.exceptions import StorageError

    store = MemoryBackend()
    with pytest.raises(ConfigurationError):
        Ledger(store, hash_algorithm="md5")

    with pytest.raises(StorageError):
        store.count()


@pytest.fixture
def paired_postgres_stores():
    """Two independent handles onto **one** freshly created PostgreSQL ledger store.

    Two connections to one table pair is what four ``uvicorn`` workers are, minus the
    scheduler. Nothing here is mocked - contract section 11 - and the tables are dropped
    afterwards, or a full run leaves a pile of ``sl_*`` behind.
    """
    import os
    import socket
    import uuid
    from urllib.parse import urlsplit

    pytest.importorskip("psycopg", reason="psycopg (v3) is not installed")
    import psycopg

    from signledger.backends.postgresql import PostgresBackend

    dsn = os.environ.get(
        "SIGNLEDGER_TEST_POSTGRES_DSN",
        "postgresql://postgres@localhost:5432/signledger_test",
    )
    parts = urlsplit(dsn)
    host, port = parts.hostname or "localhost", parts.port or 5432
    try:
        with socket.create_connection((host, port), timeout=1.0):
            pass
    except OSError as err:
        pytest.skip(f"no live PostgreSQL server at {host}:{port} ({type(err).__name__})")
    try:
        psycopg.connect(dsn, connect_timeout=2).close()
    except Exception as err:
        pytest.skip(f"cannot connect to PostgreSQL at {host}:{port} ({type(err).__name__})")

    table = "sl_" + uuid.uuid4().hex[:12]
    entries, checkpoints = f"{table}_entries", f"{table}_checkpoints"
    stores = [
        PostgresBackend(dsn, entries_table=entries, checkpoints_table=checkpoints)
        for _ in range(2)
    ]
    try:
        yield stores
    finally:
        for store in stores:
            store.close()
        with psycopg.connect(dsn, connect_timeout=5) as connection:
            with connection.cursor() as cursor:
                for name in (entries, checkpoints, f"{table}_entries_state"):
                    cursor.execute(f'DROP TABLE IF EXISTS "{name}" CASCADE')
            connection.commit()


@pytest.mark.postgres
def test_a_cold_start_race_repairs_itself_on_postgresql(paired_postgres_stores):
    """The deployment that was actually measured: FastAPI + `uvicorn --workers 4` +
    PostgreSQL served exactly 75 of 300 requests - one worker in four, forever."""
    first, second = (Ledger(store) for store in paired_postgres_stores)
    try:
        assert first.ledger_id != second.ledger_id

        second.append({"n": 0})
        first.append({"n": 1})
        second.append({"n": 2})

        assert first.ledger_id == second.ledger_id
        assert [entry.sequence for entry in first.iter_entries()] == [0, 1, 2]
        assert first.check_integrity().ok
    finally:
        first.close()
        second.close()
