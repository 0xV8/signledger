"""Adversary regression tests (architecture contract, sections 1, 5 and 8).

Every attack in this file was **demonstrated succeeding against v1.0.0**. Each one must
now fail to succeed. They are not hypotheticals and they are not unit tests: each runs
against a real store, and the damage is inflicted the way the adversary would inflict it -
directly on the rows, never through the public API.

For SQLite that means dropping the append-only triggers first. That is not cheating: a
trigger is defence in depth against an application bug, and contract section 4 is explicit
that no backend reports ``worm``. An insider with DDL rights removes it in one statement,
which is exactly why signatures and external anchors have to carry the real guarantee.
"""

import pytest

from signledger.anchor.sinks import FileAnchorSink
from signledger.backends.base import serialize_payload
from signledger.backends.memory import MemoryBackend
from signledger.backends.sqlite import SQLiteBackend
from signledger.core.ledger import Ledger
from signledger.core.verification import FailureKind, verify_chain

# --------------------------------------------------------------------------------------
# adversaries - real write access to a real store
# --------------------------------------------------------------------------------------


class MemoryAdversary:
    """Someone running inside the process with a reference to the store."""

    def __init__(self, backend):
        self.backend = backend

    def truncate_from(self, sequence):
        for entry in self.backend._entries[sequence:]:
            del self.backend._by_entry_id[entry.entry_id]
        del self.backend._entries[sequence:]

    def overwrite_data(self, sequence, payload):
        original = self.backend._entries[sequence]
        forged = original.model_copy(update={"data": payload})
        self.backend._entries[sequence] = forged
        self.backend._by_entry_id[forged.entry_id] = forged

    def delete(self, sequence):
        removed = self.backend._entries.pop(sequence)
        del self.backend._by_entry_id[removed.entry_id]

    def strip_signatures(self):
        for index, entry in enumerate(self.backend._entries):
            stripped = entry.model_copy(update={"signature": None})
            self.backend._entries[index] = stripped
            self.backend._by_entry_id[stripped.entry_id] = stripped


class SQLiteAdversary:
    """A DBA with UPDATE and DELETE on the table, plus the DDL rights to unlock it."""

    def __init__(self, backend):
        self.backend = backend
        self.table = backend.entries_table
        connection = backend._connection()
        # The append-only triggers are the first thing an insider removes.
        connection.execute(f"DROP TRIGGER IF EXISTS {self.table}_no_update")
        connection.execute(f"DROP TRIGGER IF EXISTS {self.table}_no_delete")

    def truncate_from(self, sequence):
        self.backend._connection().execute(
            f"DELETE FROM {self.table} WHERE sequence >= ?", (sequence,)  # noqa: S608
        )

    def overwrite_data(self, sequence, payload):
        self.backend._connection().execute(
            f"UPDATE {self.table} SET data = ? WHERE sequence = ?",  # noqa: S608
            (serialize_payload(payload), sequence),
        )

    def delete(self, sequence):
        self.backend._connection().execute(
            f"DELETE FROM {self.table} WHERE sequence = ?", (sequence,)  # noqa: S608
        )

    def strip_signatures(self):
        self.backend._connection().execute(
            f"UPDATE {self.table} SET signature = NULL"  # noqa: S608
        )


@pytest.fixture(params=["memory", "sqlite"])
def store(request, tmp_path):
    """A real store of each durable kind, so no attack is proved on one backend only."""
    if request.param == "memory":
        backend = MemoryBackend()
    else:
        backend = SQLiteBackend(str(tmp_path / "victim.db"))
    try:
        yield backend
    finally:
        backend.close()


@pytest.fixture
def adversary(store):
    return MemoryAdversary(store) if isinstance(store, MemoryBackend) else SQLiteAdversary(store)


@pytest.fixture
def signer():
    pytest.importorskip("cryptography", reason="cryptography is not installed")
    from signledger.crypto.signatures import Ed25519Signer

    return Ed25519Signer.generate()


@pytest.fixture
def sink(tmp_path):
    """An anchor on a path the adversary in these tests does not touch."""
    return FileAnchorSink(str(tmp_path / "anchors.jsonl"))


def kinds(report):
    return [failure.kind for failure in report.failures]


# --------------------------------------------------------------------------------------
# 1. tail truncation - v1 returned True
# --------------------------------------------------------------------------------------


def test_truncating_the_tail_is_detected(store, adversary, sink):
    """Delete the newest three of six entries. What is left is a *perfect* hash chain:
    every link matches, every digest recomputes. Only the anchored checkpoint remembers
    that the log used to be longer."""
    ledger = Ledger(store, anchor_sink=sink)
    for index in range(6):
        ledger.append({"n": index})
    ledger.checkpoint()

    adversary.truncate_from(3)

    assert ledger.check_integrity().ok is False


def test_truncating_the_tail_is_reported_as_a_head_mismatch(store, adversary, sink):
    ledger = Ledger(store, anchor_sink=sink)
    for index in range(6):
        ledger.append({"n": index})
    ledger.checkpoint()

    adversary.truncate_from(3)

    assert FailureKind.HEAD_MISMATCH in kinds(ledger.check_integrity())


def test_truncating_the_tail_is_reported_as_a_count_mismatch(store, adversary, sink):
    ledger = Ledger(store, anchor_sink=sink)
    for index in range(6):
        ledger.append({"n": index})
    ledger.checkpoint()

    adversary.truncate_from(3)

    assert FailureKind.COUNT_MISMATCH in kinds(ledger.check_integrity())


def test_the_truncated_chain_really_is_internally_consistent(store, adversary, sink):
    """Proves the previous three tests are earned by the anchor and not by the chain:
    with no checkpoint to compare against, the shortened log verifies clean."""
    ledger = Ledger(store, anchor_sink=sink)
    for index in range(6):
        ledger.append({"n": index})
    adversary.truncate_from(3)

    unanchored = verify_chain(store, chain=ledger._chain, genesis=ledger._genesis())

    assert unanchored.ok is True


def test_deleting_the_local_checkpoints_too_does_not_hide_the_truncation(
    store, adversary, sink
):
    """The local checkpoint lives in the store the adversary already controls. The
    external anchor is the copy they cannot reach."""
    ledger = Ledger(store, anchor_sink=sink)
    for index in range(6):
        ledger.append({"n": index})
    ledger.checkpoint()

    adversary.truncate_from(3)
    if isinstance(store, MemoryBackend):
        store._checkpoints.clear()
    else:
        store._connection().execute(f"DELETE FROM {store.checkpoints_table}")  # noqa: S608

    assert ledger.check_integrity().ok is False


# --------------------------------------------------------------------------------------
# 2. wholesale rewrite
# --------------------------------------------------------------------------------------


def test_a_freshly_built_chain_does_not_satisfy_another_ledgers_checkpoint(store, sink):
    """The impostor's chain is internally flawless - it just is not the log that was
    anchored."""
    victim = Ledger(store, anchor_sink=sink)
    for index in range(5):
        victim.append({"n": index})
    anchored = victim.checkpoint()

    impostor_store = MemoryBackend()
    try:
        impostor = Ledger(impostor_store, ledger_id="an-entirely-different-ledger")
        for index in range(5):
            impostor.append({"n": index})

        report = verify_chain(
            impostor_store,
            chain=impostor._chain,
            genesis=impostor._genesis(),
            checkpoint=anchored,
        )

        assert report.ok is False
    finally:
        impostor_store.close()


def test_the_impostor_chain_is_otherwise_perfectly_valid():
    """Which is the whole point: a hash chain cannot tell an honest log from a forged one."""
    impostor_store = MemoryBackend()
    try:
        impostor = Ledger(impostor_store, ledger_id="an-entirely-different-ledger")
        for index in range(5):
            impostor.append({"n": index})

        assert impostor.check_integrity().ok is True
    finally:
        impostor_store.close()


def test_replacing_the_store_under_the_same_identity_is_detected(store, sink):
    """The stronger form: the attacker keeps the victim's ledger_id and rebuilds the log
    from scratch, so the anchor is looked up under the right name."""
    victim = Ledger(store, anchor_sink=sink)
    for index in range(5):
        victim.append({"n": index})
    victim.checkpoint()
    stolen_id = victim.ledger_id

    replacement = MemoryBackend()
    try:
        impostor = Ledger(replacement, ledger_id=stolen_id, anchor_sink=sink)
        for index in range(5):
            impostor.append({"n": index * 100})

        assert impostor.check_integrity().ok is False
    finally:
        replacement.close()


# --------------------------------------------------------------------------------------
# 3. signature stripping - v1 returned True
# --------------------------------------------------------------------------------------


def test_stripping_every_signature_is_detected(store, adversary, signer):
    ledger = Ledger(store, enable_signatures=True, signer=signer)
    for index in range(4):
        ledger.append({"n": index})

    adversary.strip_signatures()

    assert ledger.check_integrity().ok is False


def test_stripping_every_signature_is_reported_as_missing_signature(store, adversary, signer):
    ledger = Ledger(store, enable_signatures=True, signer=signer)
    for index in range(4):
        ledger.append({"n": index})

    adversary.strip_signatures()

    assert kinds(ledger.check_integrity()) == [FailureKind.MISSING_SIGNATURE] * 4


def test_the_stripped_chain_still_hashes_perfectly(store, adversary, signer):
    """``signature`` is not in the preimage, so removing it cannot break a single digest.
    Nothing but an explicit missing-signature rule can catch this."""
    from signledger.crypto.hashing import HashChain

    ledger = Ledger(store, enable_signatures=True, signer=signer)
    for index in range(4):
        ledger.append({"n": index})
    adversary.strip_signatures()

    assert all(HashChain().verify_hash(entry) for entry in store.iter_entries())


# --------------------------------------------------------------------------------------
# 4. signature forgery with an unknown key
# --------------------------------------------------------------------------------------


def test_entries_signed_by_a_key_nobody_trusts_are_rejected(store, signer):
    pytest.importorskip("cryptography", reason="cryptography is not installed")
    from signledger.crypto.keys import InMemoryKeyResolver
    from signledger.crypto.signatures import Ed25519Signer

    trusted = Ed25519Signer.generate()
    resolver = InMemoryKeyResolver()
    resolver.add_pem(trusted.public_key_pem())

    # The attacker holds their own perfectly good key - it is simply not the trusted one.
    ledger = Ledger(store, enable_signatures=True, signer=signer, key_resolver=resolver)
    for index in range(3):
        ledger.append({"n": index})

    assert kinds(ledger.check_integrity()) == [FailureKind.UNKNOWN_KEY] * 3


def test_a_forged_signature_never_downgrades_to_looks_signed(store, signer):
    pytest.importorskip("cryptography", reason="cryptography is not installed")
    from signledger.crypto.keys import InMemoryKeyResolver

    ledger = Ledger(
        store, enable_signatures=True, signer=signer, key_resolver=InMemoryKeyResolver()
    )
    ledger.append({"n": 0})

    assert ledger.check_integrity().ok is False


# --------------------------------------------------------------------------------------
# 5. mid-chain edit
# --------------------------------------------------------------------------------------


def test_editing_one_entry_in_the_middle_is_detected(store, adversary):
    ledger = Ledger(store)
    for index in range(5):
        ledger.append({"n": index})

    adversary.overwrite_data(2, {"n": "forged"})

    assert ledger.check_integrity().ok is False


def test_a_mid_chain_edit_is_reported_as_a_bad_hash(store, adversary):
    ledger = Ledger(store)
    for index in range(5):
        ledger.append({"n": index})

    adversary.overwrite_data(2, {"n": "forged"})

    assert FailureKind.BAD_HASH in kinds(ledger.check_integrity())


def test_a_mid_chain_edit_names_the_sequence_that_was_edited(store, adversary):
    """An operator has to be told *which* record moved, not merely that one did."""
    ledger = Ledger(store)
    for index in range(5):
        ledger.append({"n": index})

    adversary.overwrite_data(2, {"n": "forged"})
    report = ledger.check_integrity()

    assert [f.sequence for f in report.failures if f.kind is FailureKind.BAD_HASH] == [2]


def test_editing_entry_zero_is_detected(store, adversary):
    ledger = Ledger(store)
    for index in range(3):
        ledger.append({"n": index})

    adversary.overwrite_data(0, {"n": "forged"})

    assert FailureKind.BAD_HASH in kinds(ledger.check_integrity())


def test_editing_the_head_is_detected(store, adversary):
    ledger = Ledger(store)
    for index in range(3):
        ledger.append({"n": index})

    adversary.overwrite_data(2, {"n": "forged"})

    assert FailureKind.BAD_HASH in kinds(ledger.check_integrity())


# --------------------------------------------------------------------------------------
# 6. cache masking - v1 served the cache and reported clean
# --------------------------------------------------------------------------------------


def test_tampering_is_detected_even_with_a_pristine_copy_in_the_cache(store, adversary):
    ledger = Ledger(store, max_cache_entries=100)
    for index in range(4):
        ledger.append({"n": index})

    adversary.overwrite_data(2, {"n": "forged"})

    assert ledger.check_integrity().ok is False


def test_the_cache_really_was_holding_the_undamaged_copy(store, adversary):
    """Without this, the previous test could be passing because the cache was empty."""
    ledger = Ledger(store, max_cache_entries=100)
    for index in range(4):
        ledger.append({"n": index})

    adversary.overwrite_data(2, {"n": "forged"})

    assert ledger._cache[2].data == {"n": 2}


def test_a_cache_hit_still_returns_the_stale_entry_while_verification_reports_the_damage(
    store, adversary
):
    """Reads may legitimately be served from cache; verification may not. The two answers
    disagreeing is the tell that verification is reading the store."""
    ledger = Ledger(store, max_cache_entries=100)
    for index in range(4):
        ledger.append({"n": index})
    adversary.overwrite_data(2, {"n": "forged"})

    assert ledger.get_by_sequence(2).data == {"n": 2}
    assert ledger.check_integrity().ok is False


def test_get_stats_reports_the_damage_rather_than_the_cache(store, adversary):
    ledger = Ledger(store, max_cache_entries=100)
    for index in range(4):
        ledger.append({"n": index})

    adversary.overwrite_data(1, {"n": "forged"})

    assert ledger.get_stats()["integrity_ok"] is False


# --------------------------------------------------------------------------------------
# 7. excising one incriminating record from the middle
# --------------------------------------------------------------------------------------


def test_deleting_one_record_from_the_middle_is_detected(store, adversary):
    """Unlike a truncated tail, this needs no checkpoint: the surviving entry's
    ``previous_hash`` still points at a digest that is no longer in the store."""
    ledger = Ledger(store)
    for index in range(5):
        ledger.append({"n": index})

    adversary.delete(2)

    assert ledger.check_integrity().ok is False


def test_deleting_one_record_from_the_middle_breaks_the_link(store, adversary):
    ledger = Ledger(store)
    for index in range(5):
        ledger.append({"n": index})

    adversary.delete(2)

    assert FailureKind.BROKEN_LINK in kinds(ledger.check_integrity())


def test_deleting_one_record_from_the_middle_leaves_a_sequence_gap(store, adversary):
    ledger = Ledger(store)
    for index in range(5):
        ledger.append({"n": index})

    adversary.delete(2)

    assert FailureKind.SEQUENCE_GAP in kinds(ledger.check_integrity())


def test_the_break_names_the_entry_that_now_dangles(store, adversary):
    ledger = Ledger(store)
    for index in range(5):
        ledger.append({"n": index})

    adversary.delete(2)
    report = ledger.check_integrity()

    assert [f.sequence for f in report.failures if f.kind is FailureKind.BROKEN_LINK] == [3]


def test_deleting_entry_zero_is_detected(store, adversary):
    """Removing the anchor entry cannot promote entry 1 into a valid genesis."""
    ledger = Ledger(store)
    for index in range(4):
        ledger.append({"n": index})

    adversary.delete(0)

    assert ledger.check_integrity().ok is False


# --------------------------------------------------------------------------------------
# 8. re-rooting the chain onto a genesis of the attacker's choosing
# --------------------------------------------------------------------------------------


def test_rebuilding_the_chain_onto_a_different_genesis_is_detected(store):
    """An attacker who rewrites the whole log gets a chain that links perfectly. What they
    cannot reproduce is this ledger's genesis anchor, which an auditor holds out of band.
    Without the entry-zero check the whole prefix of a chain is replaceable."""
    from datetime import datetime, timedelta, timezone

    victim = Ledger(store)
    for index in range(4):
        victim.append({"n": index})
    genuine_genesis = victim._genesis()

    forged_store = MemoryBackend()
    try:
        forged = Ledger(forged_store, ledger_id=victim.ledger_id)
        start = datetime(2030, 1, 1, tzinfo=timezone.utc)
        for index in range(4):
            forged.append({"n": index}, timestamp=start + timedelta(seconds=index))

        report = verify_chain(
            forged_store, chain=forged._chain, genesis=genuine_genesis
        )

        assert FailureKind.GENESIS_MISMATCH in kinds(report)
    finally:
        forged_store.close()


def test_the_re_rooted_chain_is_internally_flawless(store):
    """Which is why entry zero has to be checked against something outside the chain."""
    from datetime import datetime, timezone

    forged_store = MemoryBackend()
    try:
        forged = Ledger(forged_store, ledger_id="victim-ledger")
        for index in range(4):
            forged.append({"n": index}, timestamp=datetime(2030, 1, 1, tzinfo=timezone.utc))

        assert forged.check_integrity().ok is True
    finally:
        forged_store.close()
