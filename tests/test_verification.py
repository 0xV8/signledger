"""Chain verification (architecture contract, section 5).

Verification never raises for integrity reasons - it returns a report, and the report
carries **every** failure, because an operator repairing a damaged ledger needs the whole
damage list rather than whichever break happened to come first.

Nothing here mocks a backend. Damage is inflicted on a **real** store, the same way an
adversary with write access would: the row is overwritten or removed underneath the
ledger. A verifier tested against a fake store proves nothing about a real one.
"""

from datetime import datetime, timezone

import pytest

from signledger.backends.memory import MemoryBackend
from signledger.core.verification import AssuranceLevel, FailureKind, verify_chain

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

    def _append(target, count=4):
        appended = []
        for index in range(count):

            def build(tip, index=index):
                sealed = build_next_entry(
                    tip,
                    ledger_id=ledger_id,
                    created_at=ledger_created_at,
                    data={"event": "signed", "index": index},
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


def checkpoint_over(backend, ledger_id):
    """A checkpoint describing the store exactly as it stands right now."""
    from signledger.anchor.checkpoint import Checkpoint

    head = backend.get_head()
    return Checkpoint(
        ledger_id=ledger_id,
        sequence=head.sequence,
        head_hash=head.hash,
        entry_count=backend.count(),
        merkle_root="0" * 64,
        timestamp=BASE_INSTANT,
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
    store, signed_entries, chain, genesis, resolver, ledger_id
):
    signed_entries(store, 3)
    cp = checkpoint_over(store, ledger_id)

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
    store, signed_entries, chain, genesis, resolver, ledger_id
):
    signed_entries(store, 3)
    cp = checkpoint_over(store, ledger_id)

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
