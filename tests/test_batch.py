"""Tests for ``signledger.core.batch`` - the bulk append path."""

import inspect
import threading
import time
from datetime import datetime, timezone
from decimal import Decimal
from typing import List

import pytest

from signledger.backends.memory import MemoryBackend
from signledger.core import batch as batch_module
from signledger.core.batch import (
    BatchItem,
    BatchProcessor,
    BatchResult,
    DeadLetter,
    PartialBatchError,
)
from signledger.core.exceptions import (
    ConfigurationError,
    SignatureError,
    StorageError,
    ValidationError,
)
from signledger.core.ledger import Ledger
from signledger.crypto.canonical import MAX_ENTRY_BYTES
from signledger.crypto.signatures import Ed25519Signer


def test_a_batch_appends_every_item_as_its_own_entry() -> None:
    ledger = Ledger(MemoryBackend())
    processor = BatchProcessor(ledger)

    processor.append_batch([{"event": "a"}, {"event": "b"}, {"event": "c"}])

    assert [e.data["event"] for e in ledger.iter_entries()] == ["a", "b", "c"]


def test_a_batch_result_carries_the_entries_it_committed() -> None:
    ledger = Ledger(MemoryBackend())
    processor = BatchProcessor(ledger)

    result = processor.append_batch([{"event": "a"}, {"event": "b"}])

    assert [e.sequence for e in result.entries] == [0, 1]


def test_a_failure_at_item_three_of_five_commits_nothing() -> None:
    ledger = Ledger(MemoryBackend())
    processor = BatchProcessor(ledger)
    items = [
        {"event": "a"},
        {"event": "b"},
        {"event": "c", "amount": Decimal("1.00")},
        {"event": "d"},
        {"event": "e"},
    ]

    processor.append_batch(items)

    assert ledger.count() == 0


def test_a_refused_item_is_reported_as_a_dead_letter_naming_its_index() -> None:
    ledger = Ledger(MemoryBackend())
    processor = BatchProcessor(ledger)

    result = processor.append_batch(
        [{"event": "a"}, {"event": "b", "amount": Decimal("1.00")}, {"event": "c"}]
    )

    assert [(d.index, d.error) for d in result.dead_letters] == [
        (1, "CanonicalizationError")
    ]


def test_a_fully_committed_batch_is_ok() -> None:
    ledger = Ledger(MemoryBackend())

    result = BatchProcessor(ledger).append_batch([{"event": "a"}, {"event": "b"}])

    assert result.ok is True


def test_a_batch_that_refused_an_item_is_not_ok() -> None:
    ledger = Ledger(MemoryBackend())

    result = BatchProcessor(ledger).append_batch([{"event": "a"}, {"bad": Decimal("1")}])

    assert result.ok is False


def test_a_result_reports_how_many_items_were_submitted() -> None:
    ledger = Ledger(MemoryBackend())

    result = BatchProcessor(ledger).append_batch([{"n": 1}, {"n": 2}, {"n": 3}])

    assert result.submitted == 3


def test_a_non_atomic_batch_commits_the_items_that_survived_preparation() -> None:
    ledger = Ledger(MemoryBackend())
    processor = BatchProcessor(ledger, atomic=False)

    processor.append_batch([{"event": "a"}, {"bad": Decimal("1")}, {"event": "c"}])

    assert [e.data["event"] for e in ledger.iter_entries()] == ["a", "c"]


def test_an_empty_batch_is_refused_rather_than_reported_as_a_success() -> None:
    """A bulk call handed nothing did nothing. Answering ``ok`` would be a success tick
    over an empty range - contract section 5 rule 1 in a different costume."""
    ledger = Ledger(MemoryBackend())

    with pytest.raises(ValidationError):
        BatchProcessor(ledger).append_batch([])


def test_a_batch_larger_than_the_configured_maximum_is_refused() -> None:
    """The whole chain is built in memory before anything is written, so an unbounded
    batch is an unbounded allocation."""
    ledger = Ledger(MemoryBackend())
    processor = BatchProcessor(ledger, max_batch_size=2)

    with pytest.raises(ValidationError):
        processor.append_batch([{"n": 1}, {"n": 2}, {"n": 3}])


def test_a_batch_exactly_at_the_maximum_is_accepted() -> None:
    ledger = Ledger(MemoryBackend())
    processor = BatchProcessor(ledger, max_batch_size=2)

    assert processor.append_batch([{"n": 1}, {"n": 2}]).committed == 2


def test_a_batch_item_carries_metadata_onto_the_entry() -> None:
    ledger = Ledger(MemoryBackend())

    BatchProcessor(ledger).append_batch(
        [BatchItem(data={"event": "a"}, metadata={"actor": "svc"})]
    )

    assert ledger.get_by_sequence(0).metadata == {"actor": "svc"}


def test_a_bare_dict_is_the_entry_data_and_leaves_metadata_empty() -> None:
    """``{"data": ...}`` is a payload with a key called ``data``, never an item spec.
    Guessing between the two forms is how a caller silently loses a field."""
    ledger = Ledger(MemoryBackend())

    BatchProcessor(ledger).append_batch([{"data": "not-a-spec"}])

    stored = ledger.get_by_sequence(0)
    assert (stored.data, stored.metadata) == ({"data": "not-a-spec"}, {})


def test_a_batch_item_can_pin_its_own_timestamp() -> None:
    ledger = Ledger(MemoryBackend())
    moment = datetime(2024, 3, 4, 5, 6, 7, tzinfo=timezone.utc)

    BatchProcessor(ledger).append_batch([BatchItem(data={"n": 1}, timestamp=moment)])

    assert ledger.get_by_sequence(0).timestamp == moment


def test_a_batch_item_can_supply_its_own_entry_id() -> None:
    ledger = Ledger(MemoryBackend())

    BatchProcessor(ledger).append_batch([BatchItem(data={"n": 1}, entry_id="invoice-77")])

    assert ledger.get_entry("invoice-77").data == {"n": 1}


def test_items_without_a_timestamp_share_one_instant_stamped_once() -> None:
    """The batch is one event. Stamping each item separately would spread a single
    logical operation across a range of instants for no reason - and ``sequence``, not
    the clock, is the ordering authority."""
    ledger = Ledger(MemoryBackend())

    BatchProcessor(ledger).append_batch([{"n": 1}, {"n": 2}, {"n": 3}])

    assert len({e.timestamp for e in ledger.iter_entries()}) == 1


def test_an_entry_id_repeated_inside_one_batch_is_caught_before_anything_is_written() -> None:
    """UNIQUE(entry_id) would catch it, but only after the earlier entries had already
    been committed - and the backend would rebuild the same duplicate five times before
    giving up. Catching it in preparation keeps the batch abandonable whole."""
    ledger = Ledger(MemoryBackend())
    processor = BatchProcessor(ledger)

    result = processor.append_batch(
        [
            BatchItem(data={"n": 1}, entry_id="same"),
            BatchItem(data={"n": 2}, entry_id="same"),
        ]
    )

    assert ledger.count() == 0
    assert [d.index for d in result.dead_letters] == [1]


def test_a_batch_on_a_signing_ledger_produces_entries_that_verify() -> None:
    signer = Ed25519Signer.generate()
    ledger = Ledger(MemoryBackend(), enable_signatures=True, signer=signer)

    BatchProcessor(ledger).append_batch([{"n": 1}, {"n": 2}, {"n": 3}])

    report = ledger.check_integrity()
    assert report.ok, [f.message for f in report.failures]


class FlakySigner:
    """A caller-supplied signer that stops working part-way through a batch.

    Real signers fail like this: an HSM session expiring, a KMS quota, a smartcard
    removed. Everything but the failure is a genuine Ed25519 signer.
    """

    def __init__(self, fail_on_call: int) -> None:
        self._inner = Ed25519Signer.generate()
        self._fail_on_call = fail_on_call
        self.calls = 0
        self.key_id = self._inner.key_id
        self.algorithm = self._inner.algorithm

    def sign(self, payload: bytes) -> str:
        self.calls += 1
        if self.calls == self._fail_on_call:
            raise SignatureError("the signing device went away")
        return self._inner.sign(payload)

    def public_key_pem(self) -> str:
        return self._inner.public_key_pem()


def test_a_signer_that_fails_at_item_three_of_five_commits_nothing() -> None:
    signer = FlakySigner(fail_on_call=3)
    ledger = Ledger(MemoryBackend(), enable_signatures=True, signer=signer)
    processor = BatchProcessor(ledger)

    result = processor.append_batch([{"n": n} for n in range(5)])

    assert ledger.count() == 0
    assert [d.index for d in result.dead_letters] == [2]


class InterferingBackend:
    """A real :class:`MemoryBackend` with a competing writer wired into it.

    Nothing is faked: the delegate is a live backend and the interfering append is a
    genuine one. It just happens at an instant a thread race would only reach sometimes,
    which is what makes the re-link path testable rather than hopeful.
    """

    def __init__(self, inner, interfere_before: int, interfere) -> None:
        self._inner = inner
        self._interfere_before = interfere_before
        self._interfere = interfere
        self.appends = 0

    def append_atomic(self, build):
        self.appends += 1
        if self.appends == self._interfere_before:
            self._interfere(self._inner)
        return self._inner.append_atomic(build)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def test_a_writer_that_moves_the_tip_mid_batch_does_not_break_the_chain(
    ledger_id: str,
) -> None:
    inner = MemoryBackend()
    intruder = Ledger(inner, ledger_id=ledger_id, max_cache_entries=0)
    backend = InterferingBackend(inner, 2, lambda _: intruder.append({"who": "intruder"}))
    ledger = Ledger(backend, ledger_id=ledger_id)

    BatchProcessor(ledger).append_batch([{"n": 1}, {"n": 2}, {"n": 3}])

    report = ledger.check_integrity()
    assert report.ok, [f.message for f in report.failures]
    assert [e.sequence for e in ledger.iter_entries()] == [0, 1, 2, 3]


def test_batched_entries_reach_the_ledger_subscribers() -> None:
    """A subscriber that watches appends must not go blind the moment a caller switches
    to the bulk path - silently dropping notifications is data loss in an audit library."""
    ledger = Ledger(MemoryBackend())
    seen: List[str] = []
    delivered = threading.Event()

    def watch(entry) -> None:
        seen.append(entry.data["event"])
        if len(seen) == 3:
            delivered.set()

    ledger.subscribe(watch)

    BatchProcessor(ledger).append_batch([{"event": "a"}, {"event": "b"}, {"event": "c"}])

    assert delivered.wait(timeout=5)
    assert seen == ["a", "b", "c"]


class BreakingBackend:
    """A real backend whose store gives out part-way through a batch.

    Stands in for a disk filling or a connection dropping - the failures no append-only
    protocol can undo, because the earlier rows are already committed and there is no
    DELETE to take them back with.
    """

    def __init__(self, inner, break_on: int) -> None:
        self._inner = inner
        self._break_on = break_on
        self.appends = 0

    def append_atomic(self, build):
        self.appends += 1
        if self.appends == self._break_on:
            raise StorageError("the store went away")
        return self._inner.append_atomic(build)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def test_a_store_failure_mid_batch_is_reported_as_a_partial_batch() -> None:
    ledger = Ledger(BreakingBackend(MemoryBackend(), break_on=3))
    processor = BatchProcessor(ledger)

    with pytest.raises(PartialBatchError) as caught:
        processor.append_batch([{"n": 1}, {"n": 2}, {"n": 3}, {"n": 4}])

    assert [e.sequence for e in caught.value.committed] == [0, 1]
    assert ledger.count() == 2


def test_a_store_failure_on_the_very_first_entry_propagates_untouched() -> None:
    """Nothing was committed, so nothing is partial. Wrapping the error would tell an
    operator to go looking for a half-written chain that does not exist."""
    ledger = Ledger(BreakingBackend(MemoryBackend(), break_on=1))
    processor = BatchProcessor(ledger)

    with pytest.raises(StorageError) as caught:
        processor.append_batch([{"n": 1}, {"n": 2}])

    assert not isinstance(caught.value, PartialBatchError)


def test_a_max_batch_size_below_one_is_refused_at_construction() -> None:
    ledger = Ledger(MemoryBackend())

    with pytest.raises(ConfigurationError):
        BatchProcessor(ledger, max_batch_size=0)


def test_a_result_reports_how_long_the_batch_took() -> None:
    ledger = Ledger(MemoryBackend())

    result = BatchProcessor(ledger).append_batch([{"n": 1}])

    assert result.duration_seconds > 0


def test_a_result_counts_the_items_it_refused() -> None:
    ledger = Ledger(MemoryBackend())

    result = BatchProcessor(ledger).append_batch([{"a": Decimal("1")}, {"b": Decimal("2")}])

    assert result.failed == 2


def _time_once(work) -> float:
    """One timed run against a stock ledger, excluding setup and teardown."""
    ledger = Ledger(MemoryBackend())
    started = time.perf_counter()
    work(ledger)
    elapsed = time.perf_counter() - started
    ledger.close()
    return elapsed


def _race(left, right, rounds: int = 5):
    """Time two workloads alternately and return the best of each.

    Alternating matters more than the number of rounds: a machine that gets busy part-way
    through then slows both sides equally instead of handing the win to whichever ran
    first. The minimum is the least noisy statistic available, because scheduling and GC
    can only ever make a run slower.
    """
    left_times, right_times = [], []
    for _ in range(rounds):
        left_times.append(_time_once(left))
        right_times.append(_time_once(right))
    return min(left_times), min(right_times)


BENCH_PAYLOADS = [
    {
        "event": "payment.settled",
        "index": index,
        "amount": "149.99",
        "currency": "USD",
        "actor": {"id": f"user-{index}", "roles": ["teller", "auditor"]},
        "tags": ["settled", "reconciled", "eu"],
    }
    for index in range(400)
]


def test_a_batch_is_faster_than_the_same_appends_in_a_loop() -> None:
    """v1's BatchProcessor measured 0.89x a plain loop and its parallel mode 0.67x, so
    "bulk" meant "slower".

    Both sides run a stock ``Ledger``, which is what a caller actually gets. The saving is
    structural and it is made of two things: each payload is canonicalized, validated and
    deep-copied exactly once instead of being rebuilt inside the write transaction, and a
    batch does not push every entry through the ledger's read cache - measuring the byte
    cost of a payload means canonicalizing it twice more, and a bulk load has no business
    evicting a hot cache with ten thousand rows nobody asked for.
    """

    def by_loop(ledger) -> None:
        for payload in BENCH_PAYLOADS:
            ledger.append(payload)

    def by_batch(ledger) -> None:
        BatchProcessor(ledger).append_batch(BENCH_PAYLOADS)

    _time_once(by_loop)  # warm both paths before either is measured
    _time_once(by_batch)
    loop_seconds, batch_seconds = _race(by_loop, by_batch)

    assert batch_seconds < loop_seconds, (
        f"batch {batch_seconds:.4f}s was not faster than the loop {loop_seconds:.4f}s"
    )


def test_a_batch_joins_the_chain_the_store_already_holds() -> None:
    """Two workers that both start against an empty store each mint their own identity
    and neither can detect the clash at construction.

    This test used to assert the loser REFUSED, matching ``Ledger.append`` at the time.
    Both were wrong, and measurably so: the refusal was permanent - the id was frozen at
    construction with no way back - so three of four gunicorn workers returned 500 on
    every request for the life of the process. ``Ledger._adopt_or_refuse`` fixed that at
    the single-append level; this is the same correction on the bulk path.

    There is no splicing here. The loser invented a name, wrote nothing under it, and
    gives it up for the one the store actually holds. Only one chain ever exists. The
    genuine protections are asserted separately: a ledger that has already written refuses
    a foreign tip, and a caller who named the id explicitly is never overridden.
    """
    store = MemoryBackend()
    winner = Ledger(store)
    loser = Ledger(store)
    winner.append({"who": "winner"})

    result = BatchProcessor(loser).append_batch([{"who": "loser"}])

    assert result.ok is True
    assert store.count() == 2
    assert loser.ledger_id == winner.ledger_id
    assert loser.check_integrity().ok is True


def test_an_aborted_batch_leaves_an_existing_chain_verifying() -> None:
    """The other half of the atomicity requirement: not just "nothing new was written"
    but "what was already there is still sound"."""
    ledger = Ledger(MemoryBackend())
    ledger.append({"event": "before"})
    ledger.append({"event": "also-before"})

    BatchProcessor(ledger).append_batch(
        [{"event": "a"}, {"event": "b"}, {"bad": Decimal("1")}, {"event": "d"}, {"e": 5}]
    )

    assert ledger.count() == 2
    report = ledger.check_integrity()
    assert report.ok, [f.message for f in report.failures]


def test_a_batch_continues_an_existing_chain() -> None:
    ledger = Ledger(MemoryBackend())
    ledger.append({"event": "first"})

    result = BatchProcessor(ledger).append_batch([{"event": "second"}, {"event": "third"}])

    assert [e.sequence for e in result.entries] == [1, 2]
    assert ledger.check_integrity().ok


def test_a_batched_chain_verifies_from_genesis() -> None:
    """Entry zero's genesis anchor has to be derived exactly as ``Ledger.append`` derives
    it, or a pristine batched ledger reports GENESIS_MISMATCH the first time it reopens."""
    store = MemoryBackend()
    ledger = Ledger(store)
    BatchProcessor(ledger).append_batch([{"n": n} for n in range(4)])

    reopened = Ledger(store)
    report = reopened.check_integrity()

    assert report.ok, [f.message for f in report.failures]


def test_a_batch_produces_entries_indistinguishable_from_single_appends() -> None:
    """A ledger written half by ``append`` and half by ``append_batch`` is one chain."""
    ledger = Ledger(MemoryBackend())
    ledger.append({"via": "append"})
    BatchProcessor(ledger).append_batch([{"via": "batch"}, {"via": "batch"}])
    ledger.append({"via": "append"})

    assert ledger.count() == 4
    assert ledger.check_integrity().ok


def test_the_batch_path_and_the_append_path_agree_on_the_signature_fields() -> None:
    """The signing identity is inside the preimage, and both writers must put it there.

    ``Ledger.append`` and ``BatchProcessor`` build their own ``Entry`` each, and each
    stamps ``key_id`` and ``signature_algorithm`` from the signer. Dropping either from
    one of the two still yields a chain that verifies perfectly: the digest is recomputed
    over whatever was actually written, so verification cannot see the difference. Only
    holding the two paths against each other can - so the same signer writes through
    both, and the entries are compared field by field.
    """
    signer = Ed25519Signer.generate()
    ledger = Ledger(MemoryBackend(), enable_signatures=True, signer=signer)

    direct = ledger.append({"n": 1})
    batched = BatchProcessor(ledger).append_batch([{"n": 2}]).entries[0]

    assert direct.signature_algorithm == signer.algorithm
    assert direct.key_id == signer.key_id
    assert batched.signature_algorithm == direct.signature_algorithm
    assert batched.key_id == direct.key_id
    # And as stored, not merely as returned.
    stored = list(ledger.iter_entries())
    assert [e.signature_algorithm for e in stored] == [signer.algorithm] * 2
    assert [e.key_id for e in stored] == [signer.key_id] * 2
    assert ledger.check_integrity().ok


@pytest.mark.parametrize(
    "batch_first", [True, False], ids=["batch-writes-genesis", "append-writes-genesis"]
)
@pytest.mark.parametrize(("algorithm", "digest_length"), [("sha256", 64), ("sha512", 128)])
def test_a_ledger_on_a_non_default_algorithm_writes_checkpoints_and_verifies(
    algorithm: str, digest_length: int, batch_first: bool
) -> None:
    """Every ``algorithm=`` handed down the write path, exercised off the default.

    Nothing in this suite ever built a ``Ledger`` on anything but sha256, so a hop that
    quietly dropped the keyword - the genesis anchor, the entry digest, the Merkle tree
    the checkpoint publishes - fell back to the default on *both* sides of every
    comparison, and the chain still verified against itself.

    Digest *length* is what makes the fallback visible: sha256 and sha512 disagree about
    it, so a component still hashing with the default is measurable rather than merely
    self-consistent.

    The genesis anchor is derived by whichever writer lands the first entry, so both
    orderings are run: ``append`` and the batch each own their own copy of that line.
    """
    ledger = Ledger(MemoryBackend(), hash_algorithm=algorithm)
    if batch_first:
        BatchProcessor(ledger).append_batch([{"n": 0}, {"n": 1}, {"n": 2}])
        ledger.append({"n": 3})
    else:
        ledger.append({"n": 0})
        BatchProcessor(ledger).append_batch([{"n": 1}, {"n": 2}, {"n": 3}])

    checkpoint = ledger.checkpoint()

    entries = list(ledger.iter_entries())
    assert [len(e.hash) for e in entries] == [digest_length] * 4
    # Entry 0's is the genesis anchor; the rest are the digest of the entry before.
    assert [len(e.previous_hash) for e in entries] == [digest_length] * 4
    assert len(checkpoint.head_hash) == digest_length
    assert len(checkpoint.merkle_root) == digest_length
    assert ledger.check_integrity().ok


def test_an_item_with_a_naive_timestamp_is_refused() -> None:
    ledger = Ledger(MemoryBackend())

    result = BatchProcessor(ledger).append_batch(
        [BatchItem(data={"n": 1}, timestamp=datetime(2024, 1, 1))]
    )

    assert result.dead_letters[0].error == "ValidationError"
    assert ledger.count() == 0


def test_an_item_that_is_not_a_dict_or_a_batch_item_is_refused() -> None:
    ledger = Ledger(MemoryBackend())

    result = BatchProcessor(ledger).append_batch([{"n": 1}, 42])

    assert [d.index for d in result.dead_letters] == [1]


def test_an_item_with_empty_data_is_refused() -> None:
    ledger = Ledger(MemoryBackend())

    result = BatchProcessor(ledger).append_batch([{}])

    assert result.dead_letters[0].error == "ValidationError"


def test_an_item_over_the_entry_size_limit_is_refused() -> None:
    ledger = Ledger(MemoryBackend())
    oversized = {"blob": "x" * (MAX_ENTRY_BYTES + 1)}

    result = BatchProcessor(ledger).append_batch([oversized])

    assert result.dead_letters[0].error == "CanonicalizationError"
    assert ledger.count() == 0


def test_a_dead_letter_message_names_the_path_but_never_the_value() -> None:
    """Contract section 10: no exception message may carry an entry payload. A dead
    letter is the most tempting place to leak one, because it is about a payload."""
    ledger = Ledger(MemoryBackend())
    secret = "4111111111111111"

    result = BatchProcessor(ledger).append_batch([{"card": Decimal(secret)}])

    message = result.dead_letters[0].message
    assert "data.card" in message
    assert secret not in message


def test_the_store_is_detached_from_the_dict_the_caller_handed_over() -> None:
    ledger = Ledger(MemoryBackend())
    payload = {"event": "x", "nested": {"n": 1}}

    BatchProcessor(ledger).append_batch([payload])
    payload["nested"]["n"] = 999

    assert ledger.get_by_sequence(0).data == {"event": "x", "nested": {"n": 1}}


def test_a_non_atomic_batch_survives_a_signer_that_fails_once() -> None:
    signer = FlakySigner(fail_on_call=2)
    ledger = Ledger(MemoryBackend(), enable_signatures=True, signer=signer)

    result = BatchProcessor(ledger, atomic=False).append_batch([{"n": n} for n in range(4)])

    assert [d.index for d in result.dead_letters] == [1]
    assert result.committed == 3
    assert ledger.check_integrity().ok


def test_batch_works_on_every_backend(backend) -> None:
    ledger = Ledger(backend)

    result = BatchProcessor(ledger).append_batch([{"n": n} for n in range(5)])

    assert [e.sequence for e in result.entries] == [0, 1, 2, 3, 4]
    assert ledger.check_integrity().ok


def _parameters(member) -> List[str]:
    """Every parameter name on a callable, including its public methods."""
    names: List[str] = []
    for candidate in [member, *(getattr(member, attribute) for attribute in dir(member)
                                if not attribute.startswith("_")
                                and callable(getattr(member, attribute, None)))]:
        try:
            names.extend(inspect.signature(candidate).parameters)
        except (TypeError, ValueError):  # pragma: no cover - builtins have no signature
            continue
    return names


def test_the_module_offers_no_parallel_processing_switch() -> None:
    """A hash chain is serial at the tip, so threads there can only contend for it. v1
    shipped ``parallel_processing=True`` anyway and it measured 0.67x a plain loop. This
    is a regression guard, not a discovery: it exists so the flag cannot come back."""
    offenders = [
        f"{name}.{parameter}"
        for name in batch_module.__all__
        for member in [getattr(batch_module, name)]
        if callable(member)
        for parameter in _parameters(member)
        if "parallel" in parameter or "thread" in parameter or "worker" in parameter
    ]

    assert offenders == []


def test_the_module_offers_no_transactional_batch() -> None:
    """v1's ``TransactionalBatch.commit()`` raised AttributeError on the first line of its
    body and its ``_rollback()`` only logged, so it was a rollback that rolled nothing
    back. Atomicity lives in the prepare phase now; there is no second class to get it
    wrong twice."""
    assert not hasattr(batch_module, "TransactionalBatch")


def test_the_module_exports_exactly_its_public_surface() -> None:
    assert sorted(batch_module.__all__) == [
        "BatchItem",
        "BatchProcessor",
        "BatchResult",
        "DEFAULT_MAX_BATCH_SIZE",
        "DeadLetter",
        "PartialBatchError",
    ]


def test_a_result_holding_dead_letters_is_never_ok_however_it_was_built() -> None:
    """``BatchResult`` is public and constructible, so ``ok`` has to answer no whenever
    something was refused rather than relying on the counts happening to disagree."""
    result = BatchResult(entries=(), dead_letters=(DeadLetter(0, "X", "y"),), submitted=0)

    assert result.ok is False


# ------------------------------------------------------------------------------------
# Cold-start identity on the batch path.
#
# ``Ledger.append`` learned to adopt the store's identity when it minted its own and has
# not yet written (see ``Ledger._adopt_or_refuse``). ``BatchProcessor`` did not, and it is
# the README's own bulk example - so a multi-worker deployment using it stayed broken, and
# a processor built before an adoption disagreed with the very ledger it wraps.
# ------------------------------------------------------------------------------------


class TestBatchAdoptsTheStoresIdentity:
    """The batch path must answer identity exactly the way ``Ledger.append`` does."""

    def test_a_worker_that_lost_the_race_for_entry_zero_can_still_batch(self) -> None:
        """Two workers cold-start on one empty store; the loser must not be dead forever.

        This is the defect that served 145 of 300 requests under gunicorn -w 4.
        """
        backend = MemoryBackend()
        winner = Ledger(backend)
        loser = Ledger(backend)

        BatchProcessor(winner).append_batch([{"event": "winner"}])
        result = BatchProcessor(loser).append_batch([{"event": "loser"}])

        assert result.ok is True
        assert loser.ledger_id == winner.ledger_id
        assert loser.check_integrity().ok is True

    def test_an_adopted_entry_carries_the_adopted_identity_not_the_minted_one(self) -> None:
        """Re-linking against a moved tip must re-stamp the id, not just the hash.

        ``_reseal`` rebuilt sequence, previous_hash, hash and signature. If it leaves the
        minted ``ledger_id`` in place, the row written is stamped with an identity the
        chain does not belong to and verification reports it forever.
        """
        backend = MemoryBackend()
        winner = Ledger(backend)
        loser = Ledger(backend)
        minted = loser.ledger_id

        BatchProcessor(winner).append_batch([{"event": "winner"}])
        BatchProcessor(loser).append_batch([{"event": "loser"}])

        stored = list(winner.iter_entries())
        assert {entry.ledger_id for entry in stored} == {winner.ledger_id}
        assert minted not in {entry.ledger_id for entry in stored}

    def test_a_processor_built_before_adoption_agrees_with_the_ledger_after_it(self) -> None:
        """One ledger must not answer two different identities through two of its methods.

        ``BatchProcessor.__init__`` snapshotted ``ledger.ledger_id``. Once the ledger
        adopted a different one via ``append``, the processor held a stale copy - so the
        same object succeeded through ``.append()`` and refused through the processor.
        """
        backend = MemoryBackend()
        winner = Ledger(backend)
        loser = Ledger(backend)
        processor_built_early = BatchProcessor(loser)

        winner.append({"event": "winner"})
        loser.append({"event": "adopts here"})

        assert processor_built_early.ledger_id == loser.ledger_id
        assert processor_built_early.append_batch([{"event": "still works"}]).ok is True

    def test_a_batch_that_has_written_refuses_a_foreign_tip(self) -> None:
        """Adoption is a cold-start allowance, not a standing licence.

        A processor that keeps adopting would splice two independent logs together and
        every hash in the result would verify. Once this ledger has committed anything,
        a tip belonging to someone else is a fork and has to stay loud.
        """
        backend = MemoryBackend()
        ledger = Ledger(backend)
        BatchProcessor(ledger).append_batch([{"event": "ours"}])

        stored = backend._entries[-1]
        backend._entries[-1] = stored.model_copy(update={"ledger_id": "someone-else"})

        with pytest.raises(ConfigurationError, match="different ledger"):
            BatchProcessor(ledger).append_batch([{"event": "theirs"}])

    def test_an_explicitly_named_ledger_still_refuses_through_the_batch_path(self) -> None:
        """A caller who named the id meant it. Adopting would splice logs kept apart.

        Both ledgers are constructed while the store is still empty, so construction
        cannot catch the clash - there is no tip to compare against yet. The refusal has
        to come from the batch path itself, inside the transaction that finally sees a
        tip. This is the case ``_adopt_or_refuse`` must not wave through.
        """
        backend = MemoryBackend()
        alpha = Ledger(backend, ledger_id="alpha")
        beta = Ledger(backend, ledger_id="beta")
        alpha.append({"event": "alpha"})

        with pytest.raises(ConfigurationError, match="different ledger"):
            BatchProcessor(beta).append_batch([{"event": "beta"}])

    def test_the_batch_refusal_names_the_remedy(self) -> None:
        """The 2.0.0 message told an operator what was wrong and not what to do."""
        backend = MemoryBackend()
        alpha = Ledger(backend, ledger_id="alpha")
        beta = Ledger(backend, ledger_id="beta")
        alpha.append({"event": "alpha"})

        with pytest.raises(ConfigurationError) as caught:
            BatchProcessor(beta).append_batch([{"event": "beta"}])

        assert "ledger_id" in str(caught.value)
