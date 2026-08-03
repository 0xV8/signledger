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

import json
from dataclasses import replace

import pytest

from signledger.anchor.sinks import FileAnchorSink
from signledger.backends.base import serialize_payload
from signledger.backends.memory import MemoryBackend
from signledger.backends.sqlite import SQLiteBackend
from signledger.core.ledger import Ledger
from signledger.core.verification import AssuranceLevel, FailureKind, verify_chain

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

    def clear_checkpoints(self):
        self.backend._checkpoints.clear()

    def corrupt_digest(self, sequence, digest):
        original = self.backend._entries[sequence]
        forged = original.model_copy(update={"hash": digest})
        self.backend._entries[sequence] = forged
        self.backend._by_entry_id[forged.entry_id] = forged


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

    def clear_checkpoints(self):
        # No DDL needed: the checkpoints table carries no append-only triggers, which is
        # itself why a local checkpoint is not evidence against the insider.
        self.backend._connection().execute(
            f"DELETE FROM {self.backend.checkpoints_table}"  # noqa: S608
        )

    def corrupt_digest(self, sequence, digest):
        self.backend._connection().execute(
            f"UPDATE {self.table} SET hash = ? WHERE sequence = ?",  # noqa: S608
            (digest, sequence),
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


def _resign(checkpoint, signer):
    """Re-sign a doctored checkpoint so its signature is genuinely valid.

    Without this the signature check would catch every attack below and the other checks
    would never be exercised - each one has to be shown to stand on its own.
    """
    from signledger.anchor.sinks import checkpoint_signing_payload

    stamped = replace(checkpoint, key_id=signer.key_id)
    return replace(stamped, signature=signer.sign(checkpoint_signing_payload(stamped)))


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


# --------------------------------------------------------------------------------------
# 6. checkpoint forgery - the checkpoint is signed, so it has to be verified
# --------------------------------------------------------------------------------------
#
# A checkpoint is the only thing that sees a truncated tail, which makes it the thing an
# attacker goes for next. Every attack below was demonstrated succeeding against
# rebuild/v2.0.0 at commit 6681640: each one restored ``ok=True``, and several *raised*
# the reported assurance above what the intact ledger had earned.


def anchor_lines(path):
    with open(path, encoding="utf-8") as handle:
        return [line for line in handle.read().splitlines() if line.strip()]


def rewrite_anchor(path, lines):
    with open(path, "w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(line + "\n")


def forge_anchor_line(line, **fields):
    """Rewrite one published anchor record the way someone with the file would."""
    record = json.loads(line)
    record.update(fields)
    return json.dumps(record, sort_keys=True, ensure_ascii=False)


@pytest.fixture
def reachable_sink(tmp_path):
    """An anchor on a path the adversary in *these* tests does reach.

    The separate fixture is the point: the attacks above are earned by an anchor the
    insider cannot touch, and the attacks below ask what is left when they can.
    """
    return FileAnchorSink(str(tmp_path / "reachable.jsonl"))


def signed_ledger(store, sink, signer, count=6):
    ledger = Ledger(store, anchor_sink=sink, enable_signatures=True, signer=signer)
    for index in range(count):
        ledger.append({"n": index})
    return ledger


def test_a_forged_anchor_line_cannot_launder_a_truncated_chain(
    store, adversary, reachable_sink, signer
):
    """The attack that motivated this section: truncate, then rewrite the anchor to
    describe the shorter chain. The signature is garbage and the key is one no resolver
    can supply, which is exactly what verification never looked at."""
    ledger = signed_ledger(store, reachable_sink, signer)
    ledger.checkpoint()

    adversary.truncate_from(4)
    head = store.get_by_sequence(3)
    # The store's own copy goes too. Leaving it would have the genuine local checkpoint
    # outrank the forgery and catch the truncation without the forged line ever being
    # inspected - which is the right behaviour, and is covered by the rollback test. What
    # is under test here is the anchor path standing on its own.
    adversary.clear_checkpoints()
    rewrite_anchor(
        reachable_sink.path,
        [
            forge_anchor_line(
                anchor_lines(reachable_sink.path)[0],
                sequence=3,
                head_hash=head.hash,
                entry_count=4,
                merkle_root="00" * 32,
                signature="Zm9yZ2VkLXNpZ25hdHVyZQ==",
                key_id="attacker-key-not-in-any-resolver",
            )
        ],
    )

    assert ledger.check_integrity().ok is False


def test_a_forged_anchor_is_reported_as_an_unknown_checkpoint_key(
    store, adversary, reachable_sink, signer
):
    ledger = signed_ledger(store, reachable_sink, signer)
    ledger.checkpoint()

    adversary.truncate_from(4)
    head = store.get_by_sequence(3)
    # The store's own copy goes too. Leaving it would have the genuine local checkpoint
    # outrank the forgery and catch the truncation without the forged line ever being
    # inspected - which is the right behaviour, and is covered by the rollback test. What
    # is under test here is the anchor path standing on its own.
    adversary.clear_checkpoints()
    rewrite_anchor(
        reachable_sink.path,
        [
            forge_anchor_line(
                anchor_lines(reachable_sink.path)[0],
                sequence=3,
                head_hash=head.hash,
                entry_count=4,
                merkle_root="00" * 32,
                signature="Zm9yZ2VkLXNpZ25hdHVyZQ==",
                key_id="attacker-key-not-in-any-resolver",
            )
        ],
    )

    assert FailureKind.CHECKPOINT_UNKNOWN_KEY in kinds(ledger.check_integrity())


def test_a_forged_anchor_cannot_raise_assurance_above_none(
    store, adversary, reachable_sink, signer
):
    """The forgery did not merely hide the truncation - it *upgraded* the reported
    assurance, because a forged anchor line reads as an anchor."""
    ledger = signed_ledger(store, reachable_sink, signer)
    ledger.checkpoint()

    adversary.truncate_from(4)
    head = store.get_by_sequence(3)
    # The store's own copy goes too. Leaving it would have the genuine local checkpoint
    # outrank the forgery and catch the truncation without the forged line ever being
    # inspected - which is the right behaviour, and is covered by the rollback test. What
    # is under test here is the anchor path standing on its own.
    adversary.clear_checkpoints()
    rewrite_anchor(
        reachable_sink.path,
        [
            forge_anchor_line(
                anchor_lines(reachable_sink.path)[0],
                sequence=3,
                head_hash=head.hash,
                entry_count=4,
                merkle_root="00" * 32,
                signature="Zm9yZ2VkLXNpZ25hdHVyZQ==",
                key_id="attacker-key-not-in-any-resolver",
            )
        ],
    )

    assert ledger.check_integrity().assurance is AssuranceLevel.NONE


def test_a_forged_local_checkpoint_cannot_launder_a_truncated_chain(
    store, adversary, signer
):
    """No anchor, no DDL, no key: one UPDATE on a table that carries no append-only
    trigger. This is the insider the contract puts in scope, and it was the cheapest of
    all the attacks in this section."""
    ledger = Ledger(store, enable_signatures=True, signer=signer)
    for index in range(6):
        ledger.append({"n": index})
    ledger.checkpoint()

    adversary.truncate_from(4)
    head = store.get_by_sequence(3)
    genuine = store.get_latest_checkpoint()
    adversary.clear_checkpoints()
    store.save_checkpoint(
        replace(
            genuine,
            sequence=3,
            head_hash=head.hash,
            entry_count=4,
            merkle_root="ff" * 32,
            signature="Zm9yZ2Vk",
            key_id="attacker-key",
        )
    )

    assert ledger.check_integrity().ok is False


def test_a_checkpoint_whose_signature_does_not_verify_is_rejected(
    store, adversary, signer
):
    """The key resolves; the signature over it does not. Distinct from an unknown key,
    and it earns its own failure kind so an operator can tell them apart."""
    ledger = Ledger(store, enable_signatures=True, signer=signer)
    for index in range(6):
        ledger.append({"n": index})
    ledger.checkpoint()

    genuine = store.get_latest_checkpoint()
    adversary.clear_checkpoints()
    store.save_checkpoint(replace(genuine, signature="Zm9yZ2VkLXNpZ25hdHVyZQ=="))

    assert FailureKind.CHECKPOINT_BAD_SIGNATURE in kinds(ledger.check_integrity())


def test_a_checkpoint_carrying_no_signature_is_rejected_when_signing_is_on(
    store, adversary, signer
):
    ledger = Ledger(store, enable_signatures=True, signer=signer)
    for index in range(6):
        ledger.append({"n": index})
    ledger.checkpoint()

    genuine = store.get_latest_checkpoint()
    adversary.clear_checkpoints()
    store.save_checkpoint(replace(genuine, signature=None, key_id=None))

    assert FailureKind.CHECKPOINT_UNSIGNED in kinds(ledger.check_integrity())


def test_a_checkpoint_with_the_wrong_merkle_root_is_rejected(store, adversary, signer):
    """``merkle_root`` is computed, signed and published, and was then never recomputed.
    The store holds exactly the entries needed to check it."""
    ledger = Ledger(store, enable_signatures=True, signer=signer)
    for index in range(6):
        ledger.append({"n": index})
    ledger.checkpoint()

    genuine = store.get_latest_checkpoint()
    adversary.clear_checkpoints()
    store.save_checkpoint(_resign(replace(genuine, merkle_root="00" * 32), signer))

    assert FailureKind.CHECKPOINT_MERKLE_MISMATCH in kinds(ledger.check_integrity())


def test_a_checkpoint_belonging_to_another_ledger_is_rejected(store, adversary, signer):
    """Signed by a key this deployment trusts, and describing somebody else's chain."""
    ledger = Ledger(store, enable_signatures=True, signer=signer)
    for index in range(6):
        ledger.append({"n": index})
    ledger.checkpoint()

    genuine = store.get_latest_checkpoint()
    adversary.clear_checkpoints()
    store.save_checkpoint(_resign(replace(genuine, ledger_id="someone-else"), signer))

    assert FailureKind.CHECKPOINT_LEDGER_MISMATCH in kinds(ledger.check_integrity())


def test_a_checkpoint_in_an_unreadable_format_is_rejected(store, adversary, signer):
    """Refusing beats guessing: nothing in this build knows what a format it has never
    seen was supposed to bind."""
    ledger = Ledger(store, enable_signatures=True, signer=signer)
    for index in range(6):
        ledger.append({"n": index})
    ledger.checkpoint()

    genuine = store.get_latest_checkpoint()
    adversary.clear_checkpoints()
    store.save_checkpoint(_resign(replace(genuine, format_version=999), signer))

    assert FailureKind.CHECKPOINT_FORMAT_UNSUPPORTED in kinds(ledger.check_integrity())


def test_a_rolled_back_anchor_does_not_outrank_what_the_store_remembers(
    store, reachable_sink, signer
):
    """The anchor was preferred over the local checkpoint unconditionally, on the
    reasoning that the local copy is the one an insider controls. That makes anyone who
    reaches the anchor file a downgrade oracle: roll it back and the store's own
    contradicting evidence is discarded unread. Both are consulted now, and the newer
    wins - so the rollback buys nothing on its own."""
    ledger = signed_ledger(store, reachable_sink, signer, count=4)
    ledger.checkpoint()
    for index in range(4):
        ledger.append({"n": 100 + index})
    ledger.checkpoint()

    rewrite_anchor(reachable_sink.path, [anchor_lines(reachable_sink.path)[0]])

    assert ledger.check_integrity().checkpoint_used.sequence == 7


def test_rolling_the_anchor_back_does_not_hide_a_truncation(
    store, adversary, reachable_sink, signer
):
    """The rollback only pays off combined with damage, which is what this pins: the
    older anchor describes a chain the truncated one is a prefix of, so on the old
    behaviour it reconciled. The store's own newer checkpoint is what refuses it."""
    ledger = signed_ledger(store, reachable_sink, signer, count=4)
    ledger.checkpoint()
    for index in range(4):
        ledger.append({"n": 100 + index})
    ledger.checkpoint()

    adversary.truncate_from(4)
    rewrite_anchor(reachable_sink.path, [anchor_lines(reachable_sink.path)[0]])

    assert ledger.check_integrity().ok is False


def test_an_anchor_that_is_behind_confers_no_anchored_assurance(
    store, reachable_sink, signer
):
    """An anchor holding an older head is not holding *this* head, so the deployment has
    not earned the anchored label. Reported as assurance rather than as a failure: the
    commonest cause by far is a publish that did not land, and an integrity report is the
    wrong place to raise an operational alarm."""
    ledger = signed_ledger(store, reachable_sink, signer, count=4)
    ledger.checkpoint()
    for index in range(4):
        ledger.append({"n": 100 + index})
    ledger.checkpoint()

    rewrite_anchor(reachable_sink.path, [anchor_lines(reachable_sink.path)[0]])

    report = ledger.check_integrity()
    assert report.ok is True
    assert report.assurance is AssuranceLevel.SIGNED


def test_emptying_the_anchor_costs_the_anchored_assurance(
    store, reachable_sink, signer
):
    """Deleting the anchor used to be free: the local copy was never consulted once a
    sink was configured, so an empty anchor file meant no checkpoint, and no checkpoint
    meant no check *and* no failure. The store's copy is read now, so the chain is still
    checked - and nobody outside the process holds this head, so the anchored label is
    gone."""
    ledger = signed_ledger(store, reachable_sink, signer)
    ledger.checkpoint()

    rewrite_anchor(reachable_sink.path, [])

    report = ledger.check_integrity()
    assert report.assurance is AssuranceLevel.SIGNED


def test_emptying_the_anchor_does_not_hide_a_truncation(
    store, adversary, reachable_sink, signer
):
    """The half that is a security property rather than a label."""
    ledger = signed_ledger(store, reachable_sink, signer)
    ledger.checkpoint()

    adversary.truncate_from(4)
    rewrite_anchor(reachable_sink.path, [])

    assert ledger.check_integrity().ok is False


def test_an_attacker_holding_both_the_store_and_the_anchor_is_not_detected(
    store, adversary, reachable_sink, signer
):
    """The honest limit of this mechanism, pinned so nobody claims otherwise.

    Roll the chain *and* both copies of the evidence back to a genuinely signed older
    checkpoint and everything reconciles, because it is a true statement about a chain
    that really did exist. Freshness cannot be established from inside a system the
    attacker fully controls. The defence is operational - an anchor the ledger host
    cannot write - and SECURITY.md says so rather than implying the chain self-defends.

    The same limit covers wiping every copy of the evidence: a ledger that was
    checkpointed and then wiped is indistinguishable from one that was never
    checkpointed, so the reportable case is the *inconsistent* one - an anchor that has
    lost what the store still holds, which the two tests above cover.
    """
    ledger = signed_ledger(store, reachable_sink, signer, count=4)
    ledger.checkpoint()
    for index in range(4):
        ledger.append({"n": 100 + index})
    ledger.checkpoint()

    adversary.truncate_from(4)
    rewrite_anchor(reachable_sink.path, [anchor_lines(reachable_sink.path)[0]])
    adversary.clear_checkpoints()

    assert ledger.check_integrity().ok is True


def test_an_anchor_holding_an_unencodable_field_is_refused_not_crashed(
    store, adversary, reachable_sink, signer
):
    """Verification must answer, not raise.

    JSON carries a lone surrogate happily - ``"\\ud800"`` is a valid escape - and CPython
    holds the string, but UTF-8 cannot encode it, so such a checkpoint cannot be
    canonicalised into a signing payload at all. The anchor file is the reachable way in:
    a store refuses the value on the way down, and an attacker who can rewrite the anchor
    does not have to go through one. Letting the encode error propagate would turn
    tampering into a crash, which denies the answer as effectively as a forged pass.
    """
    ledger = signed_ledger(store, reachable_sink, signer, count=4)
    ledger.checkpoint()

    record = json.loads(anchor_lines(reachable_sink.path)[0])
    record["head_hash"] = "\ud800"
    adversary.clear_checkpoints()
    rewrite_anchor(reachable_sink.path, [json.dumps(record, sort_keys=True)])

    assert FailureKind.CHECKPOINT_BAD_SIGNATURE in kinds(ledger.check_integrity())


def test_a_non_hex_digest_below_the_checkpoint_is_not_reported_as_a_merkle_mismatch(
    store, adversary, signer
):
    """The entry-level check already names that damage exactly.

    A digest that is not hex cannot be turned into a Merkle leaf, so the tree cannot be
    built. Reporting a Merkle mismatch on top of the BAD_HASH would name a symptom of the
    corruption rather than the corruption, and send an operator to the wrong field.
    """
    ledger = Ledger(store, enable_signatures=True, signer=signer)
    for index in range(4):
        ledger.append({"n": index})
    ledger.checkpoint()

    adversary.corrupt_digest(1, "zz" * 32)

    report = ledger.check_integrity()
    assert FailureKind.CHECKPOINT_MERKLE_MISMATCH not in kinds(report)
    assert FailureKind.BAD_HASH in kinds(report)
