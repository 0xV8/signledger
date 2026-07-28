"""The property that makes "tamper-evident" a claim rather than a slogan.

For an **arbitrary** ledger and an **arbitrary** change to **any** hashed field of **any**
entry, verification must notice, and the report must say which record moved. The threat
model file proves specific attacks are caught; this file proves there is no unexamined
corner where a change slips through.

Both directions matter. A detector that always shouts "tampered" would pass the first
property and be useless, so the untouched-ledger property is tested just as hard.
"""

from datetime import datetime, timedelta, timezone

from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from signledger.backends.memory import MemoryBackend
from signledger.core.ledger import Ledger
from signledger.core.verification import FailureKind, verify_chain
from signledger.crypto.canonical import entry_preimage

# --------------------------------------------------------------------------------------
# strategies
# --------------------------------------------------------------------------------------

#: Comfortably inside the +/-2**53 microsecond window SLCS/1 rule 6 imposes on the
#: integer-microsecond timestamp encoding.
SAFE_INSTANTS = st.datetimes(
    min_value=datetime(1970, 1, 1, 1),
    max_value=datetime(2200, 1, 1),
).map(lambda moment: moment.replace(tzinfo=timezone.utc))

JSON_SCALARS = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-(2**40), max_value=2**40),
    st.floats(min_value=-1e9, max_value=1e9, allow_nan=False, allow_infinity=False),
    st.text(max_size=16),
)

PAYLOADS = st.dictionaries(
    st.text(min_size=1, max_size=8), JSON_SCALARS, min_size=1, max_size=4
)

HEX_DIGESTS = st.text(alphabet="0123456789abcdef", min_size=64, max_size=64)

#: Every field contract section 3 marks as covered by the preimage.
HASHED_FIELDS = (
    "format_version",
    "ledger_id",
    "sequence",
    "entry_id",
    "timestamp",
    "data",
    "metadata",
    "previous_hash",
    "key_id",
    "signature_algorithm",
)

FIELD_STRATEGIES = {
    "format_version": st.integers(min_value=1, max_value=64),
    "ledger_id": st.text(min_size=1, max_size=24),
    "sequence": st.integers(min_value=0, max_value=64),
    "entry_id": st.text(min_size=1, max_size=24),
    "timestamp": SAFE_INSTANTS,
    "data": PAYLOADS,
    "metadata": st.dictionaries(
        st.text(min_size=1, max_size=8), JSON_SCALARS, max_size=3
    ),
    "previous_hash": HEX_DIGESTS,
    "key_id": st.one_of(st.none(), st.text(min_size=1, max_size=16)),
    "signature_algorithm": st.one_of(st.none(), st.text(min_size=1, max_size=16)),
}

BASE_INSTANT = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)

SETTINGS = settings(
    max_examples=250,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)


# --------------------------------------------------------------------------------------
# helpers - a real store, really tampered with
# --------------------------------------------------------------------------------------


def build_ledger(backend, size):
    """A healthy chain of ``size`` entries in a real store."""
    ledger = Ledger(backend, max_cache_entries=0)
    for index in range(size):
        ledger.append(
            {"event": "generated", "index": index},
            timestamp=BASE_INSTANT + timedelta(seconds=index),
        )
    return ledger


def overwrite(backend, position, field, value):
    """Replace one stored row, leaving its digest untouched - a raw write to the store."""
    original = backend._entries[position]
    tampered = original.model_copy(update={field: value})
    backend._entries[position] = tampered
    backend._by_entry_id.pop(original.entry_id, None)
    backend._by_entry_id[tampered.entry_id] = tampered
    return original, tampered


def report_for(ledger, backend):
    return verify_chain(backend, chain=ledger._chain, genesis=ledger._genesis())


# --------------------------------------------------------------------------------------
# no false negatives: any change to any hashed field is detected
# --------------------------------------------------------------------------------------


@SETTINGS
@given(
    size=st.integers(min_value=1, max_value=6),
    field=st.sampled_from(HASHED_FIELDS),
    data=st.data(),
)
def test_changing_any_hashed_field_of_any_entry_is_detected(size, field, data):
    position = data.draw(st.integers(min_value=0, max_value=size - 1), label="position")
    value = data.draw(FIELD_STRATEGIES[field], label="value")

    backend = MemoryBackend()
    try:
        ledger = build_ledger(backend, size)
        original, tampered = overwrite(backend, position, field, value)
        # If the preimage is unchanged the record is byte-identical for hashing purposes,
        # so there is genuinely nothing to detect. `1` and `1.0` are one JSON number.
        assume(entry_preimage(original) != entry_preimage(tampered))

        assert report_for(ledger, backend).ok is False
    finally:
        backend.close()


@SETTINGS
@given(
    size=st.integers(min_value=1, max_value=6),
    field=st.sampled_from(HASHED_FIELDS),
    data=st.data(),
)
def test_the_failure_names_the_record_that_was_changed(size, field, data):
    """``entry_id`` identifies the record even when the mutated field is the sequence."""
    position = data.draw(st.integers(min_value=0, max_value=size - 1), label="position")
    value = data.draw(FIELD_STRATEGIES[field], label="value")

    backend = MemoryBackend()
    try:
        ledger = build_ledger(backend, size)
        original, tampered = overwrite(backend, position, field, value)
        assume(entry_preimage(original) != entry_preimage(tampered))

        named = {failure.entry_id for failure in report_for(ledger, backend).failures}

        assert tampered.entry_id in named
    finally:
        backend.close()


@SETTINGS
@given(
    size=st.integers(min_value=1, max_value=6),
    field=st.sampled_from([f for f in HASHED_FIELDS if f != "sequence"]),
    data=st.data(),
)
def test_the_failure_names_the_sequence_that_was_changed(size, field, data):
    """When the sequence itself is untouched, the report must point straight at it."""
    position = data.draw(st.integers(min_value=0, max_value=size - 1), label="position")
    value = data.draw(FIELD_STRATEGIES[field], label="value")

    backend = MemoryBackend()
    try:
        ledger = build_ledger(backend, size)
        original, tampered = overwrite(backend, position, field, value)
        assume(entry_preimage(original) != entry_preimage(tampered))

        named = {failure.sequence for failure in report_for(ledger, backend).failures}

        assert position in named
    finally:
        backend.close()


@SETTINGS
@given(size=st.integers(min_value=1, max_value=6), data=st.data())
def test_renumbering_an_entry_is_detected_and_located(size, data):
    position = data.draw(st.integers(min_value=0, max_value=size - 1), label="position")
    value = data.draw(st.integers(min_value=0, max_value=64), label="sequence")

    backend = MemoryBackend()
    try:
        ledger = build_ledger(backend, size)
        original, tampered = overwrite(backend, position, "sequence", value)
        assume(entry_preimage(original) != entry_preimage(tampered))

        report = report_for(ledger, backend)
        named = {failure.sequence for failure in report.failures}

        assert report.ok is False
        assert named & {original.sequence, tampered.sequence}
    finally:
        backend.close()


# --------------------------------------------------------------------------------------
# deletion
# --------------------------------------------------------------------------------------


@SETTINGS
@given(size=st.integers(min_value=2, max_value=8), data=st.data())
def test_removing_any_entry_but_the_last_is_detected_by_the_chain_alone(size, data):
    position = data.draw(st.integers(min_value=0, max_value=size - 2), label="position")

    backend = MemoryBackend()
    try:
        ledger = build_ledger(backend, size)
        removed = backend._entries.pop(position)
        del backend._by_entry_id[removed.entry_id]

        assert report_for(ledger, backend).ok is False
    finally:
        backend.close()


@SETTINGS
@given(size=st.integers(min_value=2, max_value=8), keep=st.integers(min_value=1, max_value=7))
def test_removing_the_tail_is_invisible_to_the_chain(size, keep):
    """Not a bug - the reason checkpoints exist. Every surviving link still matches and
    every digest still recomputes, so there is nothing inside the chain to find. Hypothesis
    discovered this by disproving a stronger property that was simply not true."""
    assume(keep < size)
    backend = MemoryBackend()
    try:
        ledger = build_ledger(backend, size)
        for entry in backend._entries[keep:]:
            del backend._by_entry_id[entry.entry_id]
        del backend._entries[keep:]

        assert report_for(ledger, backend).ok is True
    finally:
        backend.close()


@SETTINGS
@given(size=st.integers(min_value=2, max_value=8), keep=st.integers(min_value=1, max_value=7))
def test_removing_the_tail_is_always_caught_once_a_checkpoint_exists(size, keep):
    assume(keep < size)
    backend = MemoryBackend()
    try:
        ledger = build_ledger(backend, size)
        checkpoint = ledger.checkpoint()
        for entry in backend._entries[keep:]:
            del backend._by_entry_id[entry.entry_id]
        del backend._entries[keep:]

        report = verify_chain(
            backend,
            chain=ledger._chain,
            genesis=ledger._genesis(),
            checkpoint=checkpoint,
        )

        assert report.ok is False
    finally:
        backend.close()


@SETTINGS
@given(size=st.integers(min_value=2, max_value=8), keep=st.integers(min_value=1, max_value=7))
def test_a_truncated_tail_always_reports_both_a_head_and_a_count_mismatch(size, keep):
    assume(keep < size)
    backend = MemoryBackend()
    try:
        ledger = build_ledger(backend, size)
        checkpoint = ledger.checkpoint()
        for entry in backend._entries[keep:]:
            del backend._by_entry_id[entry.entry_id]
        del backend._entries[keep:]

        report = verify_chain(
            backend, chain=ledger._chain, genesis=ledger._genesis(), checkpoint=checkpoint
        )
        kinds = {failure.kind for failure in report.failures}

        assert {FailureKind.HEAD_MISMATCH, FailureKind.COUNT_MISMATCH} <= kinds
    finally:
        backend.close()


# --------------------------------------------------------------------------------------
# no false positives: an untouched ledger always verifies
# --------------------------------------------------------------------------------------


@SETTINGS
@given(payloads=st.lists(PAYLOADS, min_size=1, max_size=8))
def test_an_untouched_ledger_of_any_shape_verifies(payloads):
    """The other half of the property. A detector that cried wolf would satisfy every
    test above and be worthless."""
    backend = MemoryBackend()
    try:
        ledger = Ledger(backend, max_cache_entries=0)
        for index, payload in enumerate(payloads):
            ledger.append(payload, timestamp=BASE_INSTANT + timedelta(seconds=index))

        assert report_for(ledger, backend).ok is True
    finally:
        backend.close()


@SETTINGS
@given(
    payloads=st.lists(PAYLOADS, min_size=1, max_size=6),
    metadata=st.lists(
        st.dictionaries(st.text(min_size=1, max_size=6), JSON_SCALARS, max_size=3)
    ),
    instants=st.lists(SAFE_INSTANTS, min_size=1, max_size=6),
)
def test_an_untouched_ledger_verifies_whatever_its_timestamps_do(payloads, metadata, instants):
    """Timestamps that jump backwards must not disturb verification: ``sequence`` is the
    ordering authority, and v1's wall-clock ordering broke pristine ledgers on an NTP step."""
    backend = MemoryBackend()
    try:
        ledger = Ledger(backend, max_cache_entries=0)
        for index, payload in enumerate(payloads):
            ledger.append(
                payload,
                metadata[index] if index < len(metadata) else None,
                timestamp=instants[index % len(instants)],
            )

        assert report_for(ledger, backend).ok is True
    finally:
        backend.close()


@SETTINGS
@given(payloads=st.lists(PAYLOADS, min_size=1, max_size=6), start=st.integers(0, 5))
def test_incremental_verification_of_an_untouched_ledger_always_passes(payloads, start):
    """v1's incremental path raised on every healthy ledger, so nobody could use it."""
    assume(start < len(payloads))
    backend = MemoryBackend()
    try:
        ledger = Ledger(backend, max_cache_entries=0)
        for index, payload in enumerate(payloads):
            ledger.append(payload, timestamp=BASE_INSTANT + timedelta(seconds=index))

        assert ledger.check_integrity(from_sequence=start).ok is True
    finally:
        backend.close()


# --------------------------------------------------------------------------------------
# verification never raises, whatever it is handed
# --------------------------------------------------------------------------------------


@SETTINGS
@given(
    size=st.integers(min_value=1, max_value=5),
    field=st.sampled_from(HASHED_FIELDS),
    data=st.data(),
)
def test_verification_answers_rather_than_raising_however_the_row_is_mangled(size, field, data):
    """Contract section 5: verification never raises for integrity reasons. A verifier an
    attacker can crash is a denial of service dressed as a security control."""
    position = data.draw(st.integers(min_value=0, max_value=size - 1), label="position")
    value = data.draw(FIELD_STRATEGIES[field], label="value")

    backend = MemoryBackend()
    try:
        ledger = build_ledger(backend, size)
        overwrite(backend, position, field, value)

        report = report_for(ledger, backend)

        assert isinstance(report.ok, bool)
    finally:
        backend.close()


@SETTINGS
@given(
    size=st.integers(min_value=1, max_value=4),
    digest=st.text(max_size=80),
    data=st.data(),
)
def test_an_arbitrary_string_in_the_hash_column_never_crashes_verification(size, digest, data):
    """Including non-ASCII, which is what makes ``hmac.compare_digest`` raise on ``str``."""
    position = data.draw(st.integers(min_value=0, max_value=size - 1), label="position")

    backend = MemoryBackend()
    try:
        ledger = build_ledger(backend, size)
        overwrite(backend, position, "hash", digest)

        assert report_for(ledger, backend).ok is False
    finally:
        backend.close()
