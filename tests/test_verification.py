"""Chain verification (architecture contract, section 5).

Verification never raises for integrity reasons - it returns a report, and the report
carries **every** failure, because an operator repairing a damaged ledger needs the whole
damage list rather than whichever break happened to come first.

Nothing here mocks a backend. Damage is inflicted on a **real** store, the same way an
adversary with write access would: the row is overwritten or removed underneath the
ledger. A verifier tested against a fake store proves nothing about a real one.
"""

import contextlib
import os
import socket
import sqlite3
import uuid
from datetime import datetime, timezone
from urllib.parse import urlsplit

import pytest

from signledger.backends.base import CAP_APPEND_ONLY_ENFORCED
from signledger.backends.memory import MemoryBackend
from signledger.core.ledger import Ledger
from signledger.core.verification import AssuranceLevel, FailureKind, verify_chain
from signledger.crypto.hashing import DEFAULT_ALGORITHM

BASE_INSTANT = datetime(2024, 1, 1, tzinfo=timezone.utc)


@pytest.fixture
def store():
    """A real in-process store whose rows a test can overwrite behind the ledger's back."""
    backend = MemoryBackend()
    try:
        yield backend
    finally:
        backend.close()


def replace_stored(backend, sequence, **changes):
    """Overwrite a stored row in place - adversary B from the section 1 threat table."""
    tampered = backend._entries[sequence].model_copy(update=changes)
    backend._entries[sequence] = tampered
    backend._by_entry_id[tampered.entry_id] = tampered
    return tampered


def reseal_stored(backend, sequence, chain, **changes):
    """Overwrite a row *and* re-derive its digest.

    This is the adversary contract section 1 warns about: SHA-256 is public, so anyone
    with write access can edit a row and recompute every hash after it. A bare chain
    cannot see this; only the link, the signature or the anchor can.
    """
    draft = backend._entries[sequence].model_copy(update=changes)
    return replace_stored(backend, sequence, hash=chain.calculate_hash(draft), **changes)


def drop_stored(backend, sequence):
    """Delete a row outright - direct ``DELETE``, no API involved."""
    removed = backend._entries.pop(sequence)
    del backend._by_entry_id[removed.entry_id]
    return removed


def kinds(report):
    return [failure.kind for failure in report.failures]


# --------------------------------------------------------------------------------------
# the healthy case
# --------------------------------------------------------------------------------------


def test_a_healthy_chain_verifies_ok(backend, sample_entries, chain, genesis):
    sample_entries(backend, 5)

    report = verify_chain(backend, chain=chain, genesis=genesis)

    assert report.ok is True


# --------------------------------------------------------------------------------------
# hashes
# --------------------------------------------------------------------------------------


def test_a_tampered_payload_is_reported_as_bad_hash(store, sample_entries, chain, genesis):
    sample_entries(store, 4)
    replace_stored(store, 2, data={"event": "forged"})

    report = verify_chain(store, chain=chain, genesis=genesis)

    assert kinds(report) == [FailureKind.BAD_HASH]


# --------------------------------------------------------------------------------------
# genesis
# --------------------------------------------------------------------------------------


def test_entry_zero_linking_to_the_wrong_genesis_is_reported(store, sample_entries, chain):
    """Entry 0 anchors the chain. Without this check the whole prefix can be replaced."""
    sample_entries(store, 3)

    report = verify_chain(store, chain=chain, genesis="0" * 64)

    assert kinds(report) == [FailureKind.GENESIS_MISMATCH]


# --------------------------------------------------------------------------------------
# links
# --------------------------------------------------------------------------------------


def test_an_entry_not_linking_to_its_predecessor_is_reported_as_broken_link(
    store, sample_entries, chain, genesis
):
    entries = sample_entries(store, 4)
    reseal_stored(store, 3, chain, previous_hash=entries[0].hash)

    report = verify_chain(store, chain=chain, genesis=genesis)

    assert kinds(report) == [FailureKind.BROKEN_LINK]


# --------------------------------------------------------------------------------------
# sequence
# --------------------------------------------------------------------------------------


def test_a_deleted_middle_entry_is_reported_as_a_sequence_gap(
    store, sample_entries, chain, genesis
):
    sample_entries(store, 5)
    drop_stored(store, 2)

    report = verify_chain(store, chain=chain, genesis=genesis)

    assert FailureKind.SEQUENCE_GAP in kinds(report)


# --------------------------------------------------------------------------------------
# every failure, not just the first
# --------------------------------------------------------------------------------------


def test_three_separate_corruptions_produce_three_failures(store, sample_entries, chain):
    """An operator repairing a ledger needs the whole damage report, not the first break."""
    entries = sample_entries(store, 5)
    replace_stored(store, 1, data={"event": "forged"})
    reseal_stored(store, 4, chain, previous_hash=entries[0].hash)

    report = verify_chain(store, chain=chain, genesis="0" * 64)

    assert kinds(report) == [
        FailureKind.GENESIS_MISMATCH,
        FailureKind.BAD_HASH,
        FailureKind.BROKEN_LINK,
    ]


def test_each_failure_names_the_sequence_it_belongs_to(store, sample_entries, chain):
    entries = sample_entries(store, 5)
    replace_stored(store, 1, data={"event": "forged"})
    reseal_stored(store, 4, chain, previous_hash=entries[0].hash)

    report = verify_chain(store, chain=chain, genesis="0" * 64)

    assert [failure.sequence for failure in report.failures] == [0, 1, 4]


def test_a_single_break_does_not_cascade_into_a_failure_per_following_entry(
    store, sample_entries, chain, genesis
):
    """Verification resumes from what is stored, so one edit reports once."""
    sample_entries(store, 8)
    replace_stored(store, 2, data={"event": "forged"})

    report = verify_chain(store, chain=chain, genesis=genesis)

    assert len(report.failures) == 1


# --------------------------------------------------------------------------------------
# an empty range is never a pass
# --------------------------------------------------------------------------------------


def test_an_empty_ledger_is_not_ok(store, chain, genesis):
    """v1's CLI printed a success tick on a ledger with nothing in it."""
    report = verify_chain(store, chain=chain, genesis=genesis)

    assert report.ok is False


def test_an_empty_ledger_verifies_zero_entries(store, chain, genesis):
    report = verify_chain(store, chain=chain, genesis=genesis)

    assert report.entries_verified == 0


def test_an_empty_ledger_reports_a_count_mismatch(store, chain, genesis):
    report = verify_chain(store, chain=chain, genesis=genesis)

    assert kinds(report) == [FailureKind.COUNT_MISMATCH]


def test_an_empty_ledger_has_no_sequence_range(store, chain, genesis):
    report = verify_chain(store, chain=chain, genesis=genesis)

    assert (report.first_sequence, report.last_sequence) == (None, None)


# --------------------------------------------------------------------------------------
# incremental verification (rule 6)
# --------------------------------------------------------------------------------------


def test_verifying_from_a_sequence_is_ok_on_a_healthy_ledger(
    store, sample_entries, chain, genesis
):
    """v1's equivalent raised IntegrityError on every clean ledger, so nobody could use it."""
    sample_entries(store, 6)

    report = verify_chain(store, chain=chain, genesis=genesis, from_sequence=3)

    assert report.ok is True


def test_verifying_from_a_sequence_covers_only_the_tail(store, sample_entries, chain, genesis):
    sample_entries(store, 6)

    report = verify_chain(store, chain=chain, genesis=genesis, from_sequence=3)

    assert (report.entries_verified, report.first_sequence, report.last_sequence) == (3, 3, 5)


def test_verifying_from_a_sequence_verifies_the_start_entry_itself(
    store, sample_entries, chain, genesis
):
    """Rule 6: the entry the run starts at is checked, not merely trusted as a seed."""
    sample_entries(store, 6)
    replace_stored(store, 3, data={"event": "forged"})

    report = verify_chain(store, chain=chain, genesis=genesis, from_sequence=3)

    assert kinds(report) == [FailureKind.BAD_HASH]


def test_verifying_from_a_sequence_does_not_check_the_start_entry_against_genesis(
    store, sample_entries, chain
):
    """Entry 3's ``previous_hash`` is entry 2's digest, never the genesis hash."""
    sample_entries(store, 6)

    report = verify_chain(store, chain=chain, genesis="0" * 64, from_sequence=3)

    assert report.failures == ()


def test_verifying_from_a_sequence_past_the_head_is_not_ok(store, sample_entries, chain, genesis):
    sample_entries(store, 3)

    report = verify_chain(store, chain=chain, genesis=genesis, from_sequence=99)

    assert report.ok is False


def test_verifying_from_a_sequence_still_requires_the_range_to_start_there(
    store, sample_entries, chain, genesis
):
    sample_entries(store, 6)
    drop_stored(store, 3)

    report = verify_chain(store, chain=chain, genesis=genesis, from_sequence=3)

    assert FailureKind.SEQUENCE_GAP in kinds(report)


# --------------------------------------------------------------------------------------
# format versions (contract section 12: every historical version stays verifiable)
# --------------------------------------------------------------------------------------


def test_an_entry_in_an_unknown_format_is_reported_as_format_unsupported(
    store, sample_entries, chain, genesis
):
    sample_entries(store, 3)
    replace_stored(store, 1, format_version=99)

    report = verify_chain(store, chain=chain, genesis=genesis)

    assert FailureKind.FORMAT_UNSUPPORTED in kinds(report)


def test_an_entry_in_an_unknown_format_is_not_also_reported_as_a_bad_hash(
    store, sample_entries, chain, genesis
):
    """There is no encoder to hash it with, so claiming the digest is wrong is a guess."""
    sample_entries(store, 3)
    replace_stored(store, 1, format_version=99)

    report = verify_chain(store, chain=chain, genesis=genesis)

    assert FailureKind.BAD_HASH not in kinds(report)


def test_a_later_forgery_is_still_reported_after_an_unsupported_format(
    store, sample_entries, chain, genesis
):
    """An entry this build cannot hash ends *that* entry's checks, not the whole run.

    Abandoning the loop at the first unsupported format would leave every entry after it
    unexamined while the report still says how many entries it verified - a forgery
    hidden behind a single unreadable row, which is a cheap thing for an adversary to
    plant.
    """
    sample_entries(store, 3)
    replace_stored(store, 0, format_version=99)
    replace_stored(store, 2, data={"event": "forged"})

    report = verify_chain(store, chain=chain, genesis=genesis)

    assert kinds(report) == [FailureKind.FORMAT_UNSUPPORTED, FailureKind.BAD_HASH]
    assert report.entries_verified == 3


def test_a_non_ascii_hash_makes_verification_answer_no_rather_than_raise(
    store, sample_entries, chain, genesis
):
    """``hmac.compare_digest`` raises TypeError on a non-ASCII ``str``; bytes do not."""
    sample_entries(store, 3)
    replace_stored(store, 1, hash="not-a-digest-é中")

    report = verify_chain(store, chain=chain, genesis=genesis)

    assert FailureKind.BAD_HASH in kinds(report)


# --------------------------------------------------------------------------------------
# signatures (contract section 6 + section 5 rule 5)
# --------------------------------------------------------------------------------------


@pytest.fixture
def signer():
    pytest.importorskip("cryptography", reason="cryptography is not installed")
    from signledger.crypto.signatures import Ed25519Signer

    return Ed25519Signer.generate()


@pytest.fixture
def resolver(signer):
    from signledger.crypto.keys import InMemoryKeyResolver

    known = InMemoryKeyResolver()
    known.add(signer.public_key)
    return known


@pytest.fixture
def empty_resolver(signer):
    from signledger.crypto.keys import InMemoryKeyResolver

    return InMemoryKeyResolver()


@pytest.fixture
def signed_entries(ledger_id, ledger_created_at, chain, signer):
    """Factory: append ``count`` correctly signed entries to a real store.

    ``key_id`` and ``signature_algorithm`` go into the **draft**, before it is hashed -
    contract section 6 puts them in the preimage precisely so a signing identity cannot
    be swapped without breaking the digest.
    """
    from signledger.backends.base import build_next_entry
    from signledger.crypto.signatures import signing_payload

    def _append(target, count=4, data=None):
        appended = []
        for index in range(count):
            payload = {"event": "signed", "index": index} if data is None else dict(data)

            def build(tip, payload=payload):
                sealed = build_next_entry(
                    tip,
                    ledger_id=ledger_id,
                    created_at=ledger_created_at,
                    data=payload,
                    chain=chain,
                    key_id=signer.key_id,
                    signature_algorithm=signer.algorithm,
                )
                payload = signing_payload(sealed.ledger_id, sealed.hash)
                return sealed.model_copy(update={"signature": signer.sign(payload)})

            appended.append(target.append_atomic(build))
        return appended

    return _append


def test_a_correctly_signed_chain_verifies_ok(store, signed_entries, chain, genesis, resolver):
    signed_entries(store, 4)

    report = verify_chain(
        store, chain=chain, genesis=genesis, enable_signatures=True, key_resolver=resolver
    )

    assert report.ok is True


def test_a_stripped_signature_is_reported_as_missing_signature(
    store, signed_entries, chain, genesis, resolver
):
    """v1 returned True for a ledger whose signatures had all been set to NULL."""
    signed_entries(store, 4)
    replace_stored(store, 2, signature=None)

    report = verify_chain(
        store, chain=chain, genesis=genesis, enable_signatures=True, key_resolver=resolver
    )

    assert kinds(report) == [FailureKind.MISSING_SIGNATURE]


def test_a_signature_over_different_bytes_is_reported_as_bad_signature(
    store, signed_entries, chain, genesis, resolver, signer
):
    signed_entries(store, 4)
    replace_stored(store, 2, signature=signer.sign(b"a payload this entry never had"))

    report = verify_chain(
        store, chain=chain, genesis=genesis, enable_signatures=True, key_resolver=resolver
    )

    assert kinds(report) == [FailureKind.BAD_SIGNATURE]


def test_a_key_id_the_resolver_does_not_know_is_reported_as_unknown_key(
    store, signed_entries, chain, genesis, empty_resolver
):
    """Forgery with an unknown key must fail closed, never fall back to 'looks signed'."""
    signed_entries(store, 3)

    report = verify_chain(
        store, chain=chain, genesis=genesis, enable_signatures=True, key_resolver=empty_resolver
    )

    assert kinds(report) == [FailureKind.UNKNOWN_KEY] * 3


def test_signatures_required_without_any_resolver_is_reported_as_unknown_key(
    store, signed_entries, chain, genesis
):
    signed_entries(store, 2)

    report = verify_chain(store, chain=chain, genesis=genesis, enable_signatures=True)

    assert kinds(report) == [FailureKind.UNKNOWN_KEY] * 2


def test_a_signed_entry_whose_key_id_was_removed_is_reported_as_unknown_key(
    store, signed_entries, chain, genesis, resolver
):
    signed_entries(store, 3)
    reseal_stored(store, 2, chain, key_id=None)

    report = verify_chain(
        store, chain=chain, genesis=genesis, enable_signatures=True, key_resolver=resolver
    )

    # The message matters: handing a null key_id to a third-party resolver is a caller
    # bug waiting to happen, and "names no key" is a different diagnosis for an operator
    # than "names a key I do not have".
    assert kinds(report) == [FailureKind.UNKNOWN_KEY]
    assert "names no key" in report.failures[0].message


def test_an_unverifiable_entry_hash_is_reported_as_bad_signature_not_a_crash(
    store, signed_entries, chain, genesis, resolver
):
    """``signing_payload`` refuses a non-hex digest; that must not escape as an exception."""
    signed_entries(store, 3)
    replace_stored(store, 2, hash="not hex at all")

    report = verify_chain(
        store, chain=chain, genesis=genesis, enable_signatures=True, key_resolver=resolver
    )

    assert FailureKind.BAD_SIGNATURE in kinds(report)


def test_signatures_are_not_examined_when_the_feature_is_off(
    store, signed_entries, chain, genesis
):
    signed_entries(store, 3)
    replace_stored(store, 1, signature=None)

    report = verify_chain(store, chain=chain, genesis=genesis)

    assert report.failures == ()


def test_an_unsigned_chain_with_signatures_required_is_reported_as_missing_signature(
    store, sample_entries, chain, genesis, resolver
):
    sample_entries(store, 3)

    report = verify_chain(
        store, chain=chain, genesis=genesis, enable_signatures=True, key_resolver=resolver
    )

    assert kinds(report) == [FailureKind.MISSING_SIGNATURE] * 3


# --------------------------------------------------------------------------------------
# checkpoints - the only thing that can see a truncated tail (contract section 8)
# --------------------------------------------------------------------------------------


def checkpoint_over(backend, ledger_id, algorithm=DEFAULT_ALGORITHM, signer=None):
    """A checkpoint describing the store exactly as it stands right now.

    ``merkle_root`` used to be a placeholder of 64 zeroes here, which was harmless only
    for as long as nothing recomputed it. It is real now, because a helper whose whole
    purpose is to describe the store accurately cannot be the thing that hands the
    verifier a value the product would never produce.

    ``signer`` is required whenever the verification under test has signing enabled, for
    the same reason: a ledger that signs its entries signs its checkpoints too, so an
    unsigned one is not a checkpoint that deployment could have produced.
    """
    from dataclasses import replace as _replace

    from signledger.anchor.checkpoint import Checkpoint
    from signledger.crypto.merkle import MerkleTree

    head = backend.get_head()
    leaves = [bytes.fromhex(entry.hash) for entry in backend.iter_entries()]
    checkpoint = Checkpoint(
        ledger_id=ledger_id,
        sequence=head.sequence,
        head_hash=head.hash,
        entry_count=backend.count(),
        merkle_root=MerkleTree(leaves, algorithm=algorithm).root,
        timestamp=BASE_INSTANT,
        key_id=None if signer is None else signer.key_id,
    )
    if signer is None:
        return checkpoint

    from signledger.anchor.sinks import checkpoint_signing_payload

    return _replace(
        checkpoint, signature=signer.sign(checkpoint_signing_payload(checkpoint))
    )


def test_a_checkpoint_matching_the_store_produces_no_failure(
    store, sample_entries, chain, genesis, ledger_id
):
    sample_entries(store, 5)
    cp = checkpoint_over(store, ledger_id)

    report = verify_chain(store, chain=chain, genesis=genesis, checkpoint=cp)

    assert report.failures == ()


def test_a_truncated_tail_is_reported_as_a_head_mismatch(
    store, sample_entries, chain, genesis, ledger_id
):
    """A bare hash chain cannot see this: what is left is perfectly self-consistent."""
    sample_entries(store, 6)
    cp = checkpoint_over(store, ledger_id)
    for sequence in (5, 4, 3):
        drop_stored(store, sequence)

    report = verify_chain(store, chain=chain, genesis=genesis, checkpoint=cp)

    assert FailureKind.HEAD_MISMATCH in kinds(report)


def test_a_truncated_tail_is_also_reported_as_a_count_mismatch(
    store, sample_entries, chain, genesis, ledger_id
):
    sample_entries(store, 6)
    cp = checkpoint_over(store, ledger_id)
    for sequence in (5, 4, 3):
        drop_stored(store, sequence)

    report = verify_chain(store, chain=chain, genesis=genesis, checkpoint=cp)

    assert FailureKind.COUNT_MISMATCH in kinds(report)


def test_a_rewritten_checkpointed_entry_is_reported_as_a_head_mismatch(
    store, sample_entries, chain, genesis, ledger_id
):
    sample_entries(store, 4)
    cp = checkpoint_over(store, ledger_id)
    reseal_stored(store, 3, chain, data={"event": "rewritten"})

    report = verify_chain(store, chain=chain, genesis=genesis, checkpoint=cp)

    assert FailureKind.HEAD_MISMATCH in kinds(report)


def test_a_ledger_that_has_grown_past_its_checkpoint_is_still_ok(
    store, sample_entries, append, chain, genesis, ledger_id
):
    """Growth is the normal case. Flagging it would make checkpoints unusable."""
    sample_entries(store, 3)
    cp = checkpoint_over(store, ledger_id)
    append(store, {"event": "later"})

    report = verify_chain(store, chain=chain, genesis=genesis, checkpoint=cp)

    assert report.ok is True


def test_the_checkpoint_that_was_used_is_named_in_the_report(
    store, sample_entries, chain, genesis, ledger_id
):
    sample_entries(store, 3)
    cp = checkpoint_over(store, ledger_id)

    report = verify_chain(store, chain=chain, genesis=genesis, checkpoint=cp)

    assert report.checkpoint_used == cp


def test_no_checkpoint_leaves_checkpoint_used_empty(store, sample_entries, chain, genesis):
    sample_entries(store, 3)

    report = verify_chain(store, chain=chain, genesis=genesis)

    assert report.checkpoint_used is None


def test_an_emptied_store_with_a_checkpoint_reports_both_head_and_count_mismatch(
    store, sample_entries, chain, genesis, ledger_id
):
    sample_entries(store, 3)
    cp = checkpoint_over(store, ledger_id)
    for sequence in (2, 1, 0):
        drop_stored(store, sequence)

    report = verify_chain(store, chain=chain, genesis=genesis, checkpoint=cp)

    assert {FailureKind.HEAD_MISMATCH, FailureKind.COUNT_MISMATCH} <= set(kinds(report))


# --------------------------------------------------------------------------------------
# assurance levels - computed, never asserted in prose (contract section 1)
# --------------------------------------------------------------------------------------


def test_a_broken_chain_earns_no_assurance(store, sample_entries, chain, genesis):
    sample_entries(store, 3)
    replace_stored(store, 1, data={"event": "forged"})

    report = verify_chain(store, chain=chain, genesis=genesis)

    assert report.assurance is AssuranceLevel.NONE


def test_an_unsigned_chain_earns_chain_only(store, sample_entries, chain, genesis):
    sample_entries(store, 3)

    report = verify_chain(store, chain=chain, genesis=genesis)

    assert report.assurance is AssuranceLevel.CHAIN_ONLY


def test_a_verified_signed_chain_earns_signed(store, signed_entries, chain, genesis, resolver):
    signed_entries(store, 3)

    report = verify_chain(
        store, chain=chain, genesis=genesis, enable_signatures=True, key_resolver=resolver
    )

    assert report.assurance is AssuranceLevel.SIGNED


def test_signatures_alone_do_not_earn_anchored(store, signed_entries, chain, genesis, resolver):
    signed_entries(store, 3)

    report = verify_chain(
        store, chain=chain, genesis=genesis, enable_signatures=True, key_resolver=resolver
    )

    assert report.assurance is not AssuranceLevel.ANCHORED


def test_a_signed_chain_with_an_external_anchor_earns_anchored(
    store, signed_entries, chain, genesis, resolver, ledger_id, signer
):
    signed_entries(store, 3)
    cp = checkpoint_over(store, ledger_id, signer=signer)

    report = verify_chain(
        store,
        chain=chain,
        genesis=genesis,
        enable_signatures=True,
        key_resolver=resolver,
        checkpoint=cp,
        anchored=True,
    )

    assert report.assurance is AssuranceLevel.ANCHORED


def test_an_anchored_chain_on_an_append_only_store_earns_worm(
    store, signed_entries, chain, genesis, resolver, ledger_id, signer
):
    signed_entries(store, 3)
    cp = checkpoint_over(store, ledger_id, signer=signer)

    report = verify_chain(
        store,
        chain=chain,
        genesis=genesis,
        enable_signatures=True,
        key_resolver=resolver,
        checkpoint=cp,
        anchored=True,
        capabilities=frozenset({"append_only_enforced"}),
    )

    assert report.assurance is AssuranceLevel.WORM


def test_an_append_only_store_without_signatures_never_claims_worm(
    store, sample_entries, chain, genesis, ledger_id
):
    """A capability token is not evidence; it only upgrades assurance already earned."""
    sample_entries(store, 3)
    cp = checkpoint_over(store, ledger_id)

    report = verify_chain(
        store,
        chain=chain,
        genesis=genesis,
        checkpoint=cp,
        anchored=True,
        capabilities=frozenset({"append_only_enforced"}),
    )

    assert report.assurance is AssuranceLevel.CHAIN_ONLY


def test_an_unverified_empty_ledger_earns_no_assurance(store, chain, genesis):
    report = verify_chain(store, chain=chain, genesis=genesis)

    assert report.assurance is AssuranceLevel.NONE


# --------------------------------------------------------------------------------------
# rule 7 - constant-time comparison
# --------------------------------------------------------------------------------------


def test_digest_comparisons_go_through_compare_digest(store, sample_entries, chain, genesis,
                                                      monkeypatch):
    """Contract section 5 rule 7.

    ``==`` answers identically, so no assertion about the *result* can tell the two apart -
    the difference is timing, and a byte-at-a-time comparison leaks the digest an attacker
    is trying to forge. The call itself is therefore what gets pinned.
    """
    import signledger.core.verification as verification

    real = verification.hmac.compare_digest
    calls = []

    def counting_compare(left, right):
        calls.append((type(left), type(right)))
        return real(left, right)

    monkeypatch.setattr(verification.hmac, "compare_digest", counting_compare)
    sample_entries(store, 3)

    verify_chain(store, chain=chain, genesis=genesis)

    assert calls


def test_digest_comparisons_are_made_over_bytes_never_str(store, sample_entries, chain, genesis,
                                                          monkeypatch):
    """``compare_digest`` raises TypeError on a non-ASCII ``str``, so a tampered row could
    otherwise crash the verifier instead of failing it."""
    import signledger.core.verification as verification

    real = verification.hmac.compare_digest
    seen = []

    def counting_compare(left, right):
        seen.append((type(left), type(right)))
        return real(left, right)

    monkeypatch.setattr(verification.hmac, "compare_digest", counting_compare)
    sample_entries(store, 3)

    verify_chain(store, chain=chain, genesis=genesis)

    assert all(pair == (bytes, bytes) for pair in seen)


# --------------------------------------------------------------------------------------
# a key the resolver does not hold (FailureKind.UNKNOWN_KEY, contract section 6)
# --------------------------------------------------------------------------------------
#
# The realistic shape of this is not "no keys configured". It is a resolver that holds
# *some* keys and not the one that signed: a rotation where the retired public key was
# dropped, or an entry signed by a key that was never trusted at all. The resolver must
# answer on the key the entry names, so a chain signed under key A cannot be waved
# through by a verifier that happens to trust key B.


@pytest.fixture
def stranger_resolver():
    """A populated resolver that holds a *different* key from the one that signed."""
    pytest.importorskip("cryptography", reason="cryptography is not installed")
    from signledger.crypto.keys import InMemoryKeyResolver
    from signledger.crypto.signatures import Ed25519Signer

    known = InMemoryKeyResolver()
    known.add(Ed25519Signer.generate().public_key)
    return known


def test_a_chain_signed_by_a_key_the_resolver_does_not_hold_is_not_ok(
    store, signed_entries, chain, genesis, stranger_resolver
):
    """The resolver holds a real key - just not this one. Failing closed is the whole point."""
    signed_entries(store, 3)

    report = verify_chain(
        store, chain=chain, genesis=genesis, enable_signatures=True, key_resolver=stranger_resolver
    )

    assert report.ok is False


def test_a_chain_signed_by_a_key_the_resolver_does_not_hold_is_reported_as_unknown_key(
    store, signed_entries, chain, genesis, stranger_resolver
):
    signed_entries(store, 3)

    report = verify_chain(
        store, chain=chain, genesis=genesis, enable_signatures=True, key_resolver=stranger_resolver
    )

    assert kinds(report) == [FailureKind.UNKNOWN_KEY] * 3


def test_a_resolver_holding_the_wrong_key_is_not_offered_as_a_substitute(
    store, signed_entries, chain, genesis, stranger_resolver
):
    """A resolver that answered with whatever key it had would report BAD_SIGNATURE, which
    reads like an attack. The lookup is by ``key_id``, so the honest diagnosis is that the
    key is missing."""
    signed_entries(store, 3)

    report = verify_chain(
        store, chain=chain, genesis=genesis, enable_signatures=True, key_resolver=stranger_resolver
    )

    assert FailureKind.BAD_SIGNATURE not in kinds(report)


def test_an_unknown_key_failure_names_the_key_it_could_not_supply(
    store, signed_entries, chain, genesis, stranger_resolver, signer
):
    """Without the fingerprint the operator cannot tell rotation from forgery."""
    signed_entries(store, 2)

    report = verify_chain(
        store, chain=chain, genesis=genesis, enable_signatures=True, key_resolver=stranger_resolver
    )

    assert [failure.expected for failure in report.failures] == [signer.key_id] * 2


def test_a_chain_signed_by_an_unknown_key_earns_no_assurance(
    store, signed_entries, chain, genesis, stranger_resolver
):
    signed_entries(store, 2)

    report = verify_chain(
        store, chain=chain, genesis=genesis, enable_signatures=True, key_resolver=stranger_resolver
    )

    assert report.assurance is AssuranceLevel.NONE


def test_an_unknown_key_failure_never_quotes_the_entry_payload(
    store, signed_entries, chain, genesis, stranger_resolver
):
    """Contract section 10. An UNKNOWN_KEY report is exactly what an operator pastes into
    a ticket, so the one thing it must not carry is the record itself."""
    signed_entries(store, 1, data={"diagnosis": "MARKER-7f2c-do-not-log"})

    report = verify_chain(
        store, chain=chain, genesis=genesis, enable_signatures=True, key_resolver=stranger_resolver
    )

    assert "MARKER-7f2c-do-not-log" not in repr(report)


# --------------------------------------------------------------------------------------
# assurance measured on a REAL on-disk store (contract section 1)
# --------------------------------------------------------------------------------------
#
# Everything above runs against MemoryBackend, which reports no ``append_only_enforced``
# and therefore can never distinguish ANCHORED from WORM. SQLite and PostgreSQL can: they
# arm a real append-only guard when they create the entries table, and ``capabilities()``
# probes the live catalog for it rather than asserting it. What that means for the five
# levels is established here, on live stores, with the damage inflicted the way an insider
# would inflict it - over a second connection, in SQL, behind the ledger's back.

POSTGRES_DSN_ENV = "SIGNLEDGER_TEST_POSTGRES_DSN"
DEFAULT_POSTGRES_DSN = "postgresql://postgres@localhost:5432/signledger_test"


def _postgres_dsn():
    return os.environ.get(POSTGRES_DSN_ENV, DEFAULT_POSTGRES_DSN)


def _require_postgres():
    """Skip - never fail - when no server answers. Reports host:port, never the DSN."""
    dsn = _postgres_dsn()
    parts = urlsplit(dsn)
    host, port = parts.hostname or "localhost", parts.port or 5432
    try:
        with socket.create_connection((host, port), timeout=1.0):
            pass
    except OSError as err:
        pytest.skip(f"no live PostgreSQL server at {host}:{port} ({type(err).__name__})")
    return dsn


class DurableStore:
    """A live on-disk backend, plus the two moves an insider with database access has.

    ``disarm`` and ``rewrite`` are issued over a **separate connection** in raw SQL, which
    is the only honest way to test this: the backend's own API cannot perform either, and
    that is precisely what the append-only guard is for.
    """

    def __init__(self, backend, disarm, rewrite, close):
        self.backend = backend
        self.disarm = disarm
        self.rewrite = rewrite
        self._close = close

    def close(self):
        self._close()


def _make_sqlite_store(tmp_path, index):
    from signledger.backends.sqlite import SQLiteBackend

    path = str(tmp_path / f"assurance-{index}.db")
    backend = SQLiteBackend(path)
    table = backend.entries_table

    def disarm():
        """Drop every append-only trigger on the entries table, as a DBA would."""
        with contextlib.closing(sqlite3.connect(path)) as connection:
            cursor = connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'trigger' AND tbl_name = ?",
                (table,),
            )
            names = [str(row[0]) for row in cursor.fetchall()]
            cursor.close()
            for name in names:
                connection.execute(f"DROP TRIGGER {name}").close()
            connection.commit()
        return names

    def rewrite(sequence, payload):
        with contextlib.closing(sqlite3.connect(path)) as connection:
            connection.execute(
                f"UPDATE {table} SET data = ? WHERE sequence = ?", (payload, sequence)
            ).close()
            connection.commit()

    return DurableStore(backend, disarm, rewrite, backend.close)


def _make_postgres_store(tmp_path, index):
    psycopg = pytest.importorskip("psycopg", reason="psycopg (v3) is not installed")
    from psycopg import sql

    from signledger.backends.postgresql import PostgresBackend

    dsn = _require_postgres()
    prefix = f"sl_{uuid.uuid4().hex[:10]}_{index}"
    backend = PostgresBackend(
        dsn, entries_table=f"{prefix}_entries", checkpoints_table=f"{prefix}_checkpoints"
    )
    table = backend.entries_table

    def disarm():
        with psycopg.connect(dsn, autocommit=True) as connection:
            with connection.cursor() as cur:
                cur.execute(
                    "SELECT t.tgname FROM pg_trigger t JOIN pg_class c ON c.oid = t.tgrelid "
                    "WHERE c.relname = %s AND NOT t.tgisinternal",
                    (table,),
                )
                names = [str(row[0]) for row in cur.fetchall()]
                for name in names:
                    cur.execute(
                        sql.SQL("DROP TRIGGER {trigger} ON {entries}").format(
                            trigger=sql.Identifier(name), entries=sql.Identifier(table)
                        )
                    )
        return names

    def rewrite(sequence, payload):
        with psycopg.connect(dsn, autocommit=True) as connection:
            with connection.cursor() as cur:
                cur.execute(
                    sql.SQL("UPDATE {entries} SET data = %s WHERE sequence = %s").format(
                        entries=sql.Identifier(table)
                    ),
                    (payload, sequence),
                )

    def close():
        backend.close()
        with psycopg.connect(dsn, autocommit=True) as connection:
            with connection.cursor() as cur:
                for name in (backend.entries_table, backend.checkpoints_table):
                    cur.execute(
                        sql.SQL("DROP TABLE IF EXISTS {name} CASCADE").format(
                            name=sql.Identifier(name)
                        )
                    )
                cur.execute(
                    sql.SQL("DROP FUNCTION IF EXISTS {name}() CASCADE").format(
                        name=sql.Identifier(backend.append_only_function)
                    )
                )

    return DurableStore(backend, disarm, rewrite, close)


_DURABLE_MAKERS = {"sqlite": _make_sqlite_store, "postgres": _make_postgres_store}


@pytest.fixture(
    params=[
        pytest.param("sqlite", id="sqlite"),
        pytest.param("postgres", id="postgres", marks=pytest.mark.postgres),
    ]
)
def durable(request, tmp_path):
    """Factory: a fresh live on-disk store of the parametrized kind, torn down after."""
    made = []

    def _make():
        store = _DURABLE_MAKERS[request.param](tmp_path, len(made))
        made.append(store)
        return store

    try:
        yield _make
    finally:
        for store in made:
            store.close()


@pytest.fixture
def external_anchor(tmp_path):
    """A sink outside the ledger's own database - the only kind that can confer ANCHORED."""
    from signledger.anchor.sinks import FileAnchorSink

    return FileAnchorSink(str(tmp_path / "external-anchor.jsonl"))


def test_a_fresh_on_disk_store_really_does_enforce_append_only(durable):
    """The premise for everything below. If this token were not real on these stores,
    every WORM assertion here would be measuring a claim rather than a guarantee."""
    store = durable()

    assert CAP_APPEND_ONLY_ENFORCED in store.backend.capabilities()


def test_an_unsigned_ledger_on_an_append_only_store_still_earns_only_chain_only(durable):
    """The store genuinely enforces append-only here - unlike MemoryBackend, where this
    same assertion cannot tell "capabilities do not upgrade unearned assurance" from
    "there was no capability to upgrade with"."""
    store = durable()

    with Ledger(store.backend) as ledger:
        ledger.append({"event": "opened"})

        assert ledger.assurance_level() is AssuranceLevel.CHAIN_ONLY


def test_a_signed_ledger_with_no_anchor_earns_signed_on_a_real_store(durable, signer):
    store = durable()

    with Ledger(store.backend, enable_signatures=True, signer=signer) as ledger:
        ledger.append({"event": "opened"})

        assert ledger.assurance_level() is AssuranceLevel.SIGNED


def test_a_checkpoint_kept_in_the_ledgers_own_database_is_not_an_anchor(durable, signer):
    """Anyone who can truncate the chain can delete the local record of how long it was.
    A checkpoint only anchors anything once it is somewhere they do not control."""
    store = durable()

    with Ledger(store.backend, enable_signatures=True, signer=signer) as ledger:
        ledger.append({"event": "opened"})
        ledger.checkpoint()

        assert ledger.assurance_level() is AssuranceLevel.SIGNED


def test_an_externally_anchored_signed_ledger_on_a_guarded_store_earns_worm(
    durable, signer, external_anchor
):
    store = durable()

    with Ledger(
        store.backend, enable_signatures=True, signer=signer, anchor_sink=external_anchor
    ) as ledger:
        ledger.append({"event": "opened"})
        ledger.checkpoint()

        assert ledger.assurance_level() is AssuranceLevel.WORM


def test_removing_the_append_only_guard_downgrades_worm_to_anchored(
    durable, signer, external_anchor
):
    """ANCHORED, on a real on-disk store, at last.

    SQLite and Postgres arm the guard when they create the entries table, so a healthy
    signed-and-anchored deployment reports WORM and skips straight past ANCHORED. The
    level is what remains after somebody with DDL rights removes the guard - and saying
    so, instead of continuing to say WORM, is the entire reason ``capabilities()`` probes
    the catalog. It used to assert the token unconditionally, which made this pair
    indistinguishable and ANCHORED unreachable on disk.
    """
    store = durable()

    with Ledger(
        store.backend, enable_signatures=True, signer=signer, anchor_sink=external_anchor
    ) as ledger:
        ledger.append({"event": "opened"})
        ledger.checkpoint()
        before = ledger.assurance_level()
        store.disarm()

        assert (before, ledger.assurance_level()) == (
            AssuranceLevel.WORM,
            AssuranceLevel.ANCHORED,
        )


def test_a_row_rewritten_behind_the_ledger_earns_no_assurance(durable, signer, external_anchor):
    store = durable()

    with Ledger(
        store.backend, enable_signatures=True, signer=signer, anchor_sink=external_anchor
    ) as ledger:
        ledger.append({"event": "opened"})
        ledger.append({"event": "second"})
        ledger.checkpoint()
        store.disarm()
        store.rewrite(1, '{"event": "forged"}')

        assert ledger.assurance_level() is AssuranceLevel.NONE


def test_a_row_rewritten_behind_the_ledger_is_reported_as_a_bad_hash(durable, signer):
    """An UPDATE the guard would have refused, once the guard is gone."""
    store = durable()

    with Ledger(store.backend, enable_signatures=True, signer=signer) as ledger:
        ledger.append({"event": "opened"})
        ledger.append({"event": "second"})
        store.disarm()
        store.rewrite(1, '{"event": "forged"}')

        assert FailureKind.BAD_HASH in kinds(ledger.check_integrity())


def test_every_assurance_level_is_reachable_on_a_real_on_disk_store(
    durable, signer, external_anchor
):
    """Contract section 1 names five levels. All five, observed on a live store.

    ANCHORED had only ever been seen on MemoryBackend, and on SQLite it was unreachable by
    construction while ``capabilities()`` asserted ``append_only_enforced`` instead of
    probing for it. This walks one signed ledger down the lattice: anchor it, remove the
    guard, then rewrite a row.
    """
    observed = []

    unsigned = durable()
    with Ledger(unsigned.backend) as ledger:
        ledger.append({"event": "unsigned"})
        observed.append(ledger.assurance_level())

    guarded = durable()
    with Ledger(
        guarded.backend, enable_signatures=True, signer=signer, anchor_sink=external_anchor
    ) as ledger:
        ledger.append({"event": "signed"})
        observed.append(ledger.assurance_level())
        ledger.checkpoint()
        observed.append(ledger.assurance_level())
        guarded.disarm()
        observed.append(ledger.assurance_level())
        guarded.rewrite(0, '{"event": "forged"}')
        observed.append(ledger.assurance_level())

    assert observed == [
        AssuranceLevel.CHAIN_ONLY,
        AssuranceLevel.SIGNED,
        AssuranceLevel.WORM,
        AssuranceLevel.ANCHORED,
        AssuranceLevel.NONE,
    ]


def test_a_ledger_whose_resolver_lacks_the_signing_key_reports_unknown_key_on_a_real_store(
    durable, signer, stranger_resolver
):
    """Key rotation that dropped the retired public key, end to end through the API an
    operator actually calls."""
    store = durable()

    with Ledger(
        store.backend, enable_signatures=True, signer=signer, key_resolver=stranger_resolver
    ) as ledger:
        ledger.append({"event": "opened"})
        report = ledger.check_integrity()

    assert kinds(report) == [FailureKind.UNKNOWN_KEY]


def test_a_ledger_whose_resolver_lacks_the_signing_key_is_not_ok_on_a_real_store(
    durable, signer, stranger_resolver
):
    store = durable()

    with Ledger(
        store.backend, enable_signatures=True, signer=signer, key_resolver=stranger_resolver
    ) as ledger:
        ledger.append({"event": "opened"})
        report = ledger.check_integrity()

    assert report.ok is False
