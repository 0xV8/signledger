"""Concurrent appends (architecture contract, section 4).

``append_atomic`` is the entire concurrency story: the tip read and the insert happen in
one transaction. v1 did a read-modify-write under a *process-local* lock, which looks fine
in a single-threaded test and forks the chain permanently the moment a second gunicorn
worker exists.

So the interesting test is not the threaded one - it is the two-process one at the bottom
of this file. A process-local lock passes every threaded test ever written and still loses
the chain in production.
"""

import multiprocessing
import os
import threading
import uuid

import pytest

from signledger.backends.memory import MemoryBackend
from signledger.backends.sqlite import SQLiteBackend
from signledger.core.ledger import Ledger

THREADS = 16
PER_THREAD = 200

#: A fixed identity, so every writer - thread or process - continues one chain rather than
#: minting its own. This is the production practice too: a ledger's id belongs to the
#: deployment, not to whichever worker happened to start first.
SHARED_LEDGER_ID = "11111111-2222-3333-4444-555555555555"


def append_in_this_process(path, ledger_id, count, marker):
    """Append ``count`` entries to the SQLite ledger at ``path``. Used by both tests."""
    backend = SQLiteBackend(path, timeout=60.0)
    ledger = Ledger(backend, ledger_id=ledger_id, max_cache_entries=0)
    try:
        for index in range(count):
            ledger.append({"marker": marker, "index": index, "pid": os.getpid()})
    finally:
        ledger.close()
    return count


def create_store(path):
    """Create the schema and put the database into WAL mode before any worker opens it.

    A deployment does this once - at migration time, or in the first worker to start.
    It also works around a live defect in ``signledger/backends/sqlite.py``: see
    :func:`test_two_processes_opening_a_brand_new_database_at_once`.
    """
    backend = SQLiteBackend(path, timeout=60.0)
    backend.close()
    return path


def assert_one_unbroken_chain(entries, expected_count):
    """Every property a forked or gap-ridden chain would violate."""
    assert len(entries) == expected_count

    sequences = [entry.sequence for entry in entries]
    assert sequences == list(range(expected_count)), "sequence gaps or duplicates"

    assert len({entry.entry_id for entry in entries}) == expected_count, "duplicate entry_id"
    assert len({entry.hash for entry in entries}) == expected_count, "duplicate digest"

    # A fork is two entries claiming the same parent. Every previous_hash must be unique
    # *and* must be the digest of the entry immediately before it.
    assert len({entry.previous_hash for entry in entries}) == expected_count, "forked chain"
    for earlier, later in zip(entries, entries[1:]):
        assert later.previous_hash == earlier.hash


# --------------------------------------------------------------------------------------
# threads
# --------------------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("store", ["memory", "sqlite"])
def test_sixteen_threads_appending_produce_one_unbroken_chain(store, tmp_path):
    backend = (
        MemoryBackend()
        if store == "memory"
        else SQLiteBackend(str(tmp_path / "threads.db"), timeout=60.0)
    )
    ledger = Ledger(backend, ledger_id=SHARED_LEDGER_ID, max_cache_entries=0)
    failures = []

    def worker(number):
        try:
            for index in range(PER_THREAD):
                ledger.append({"thread": number, "index": index})
        except BaseException as err:  # noqa: BLE001 - re-raised on the main thread below
            failures.append(err)

    try:
        threads = [threading.Thread(target=worker, args=(number,)) for number in range(THREADS)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=300)

        assert failures == []
        assert_one_unbroken_chain(list(ledger.iter_entries()), THREADS * PER_THREAD)
    finally:
        ledger.close()


@pytest.mark.slow
@pytest.mark.parametrize("store", ["memory", "sqlite"])
def test_a_ledger_written_by_sixteen_threads_still_verifies(store, tmp_path):
    backend = (
        MemoryBackend()
        if store == "memory"
        else SQLiteBackend(str(tmp_path / "threads-verify.db"), timeout=60.0)
    )
    ledger = Ledger(backend, ledger_id=SHARED_LEDGER_ID, max_cache_entries=0)

    def worker(number):
        for index in range(PER_THREAD):
            ledger.append({"thread": number, "index": index})

    try:
        threads = [threading.Thread(target=worker, args=(number,)) for number in range(THREADS)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=300)

        report = ledger.check_integrity()

        assert report.ok is True
        assert report.entries_verified == THREADS * PER_THREAD
    finally:
        ledger.close()


@pytest.mark.slow
def test_concurrent_readers_never_see_a_torn_entry(tmp_path):
    """A reader racing the writer must see either an entry or nothing - never a row whose
    digest does not match its own contents."""
    from signledger.crypto.hashing import HashChain

    backend = SQLiteBackend(str(tmp_path / "readers.db"), timeout=60.0)
    ledger = Ledger(backend, ledger_id=SHARED_LEDGER_ID, max_cache_entries=0)
    stop = threading.Event()
    bad = []

    def writer():
        for index in range(400):
            ledger.append({"index": index})
        stop.set()

    def reader():
        chain = HashChain()
        while not stop.is_set():
            for entry in ledger.iter_entries():
                if not chain.verify_hash(entry):
                    bad.append(entry.sequence)

    try:
        workers = [threading.Thread(target=writer)] + [
            threading.Thread(target=reader) for _ in range(3)
        ]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(timeout=300)

        assert bad == []
    finally:
        ledger.close()


# --------------------------------------------------------------------------------------
# processes - the scenario that permanently forked v1 under gunicorn
# --------------------------------------------------------------------------------------


@pytest.mark.slow
def test_two_processes_appending_to_one_sqlite_file_do_not_fork_the_chain(tmp_path):
    """The test v1 could never have passed.

    Two genuinely separate OS processes, one database file. A ``threading.Lock`` in the
    writer's address space is invisible to the other process, so if anything but the
    database transaction is providing the atomicity, this forks.
    """
    path = create_store(str(tmp_path / "shared.db"))
    per_process = 150
    context = multiprocessing.get_context("spawn")

    with context.Pool(2) as pool:
        written = pool.starmap(
            append_in_this_process,
            [(path, SHARED_LEDGER_ID, per_process, "alpha"),
             (path, SHARED_LEDGER_ID, per_process, "beta")],
        )

    assert written == [per_process, per_process]

    backend = SQLiteBackend(path, timeout=60.0)
    ledger = Ledger(backend, ledger_id=SHARED_LEDGER_ID, max_cache_entries=0)
    try:
        assert_one_unbroken_chain(list(ledger.iter_entries()), per_process * 2)
        assert ledger.check_integrity().ok is True
    finally:
        ledger.close()


@pytest.mark.slow
def test_both_processes_actually_got_entries_in(tmp_path):
    """Guards the test above: it would also pass if one process silently wrote nothing."""
    path = create_store(str(tmp_path / "shared-markers.db"))
    per_process = 100
    context = multiprocessing.get_context("spawn")

    with context.Pool(2) as pool:
        pool.starmap(
            append_in_this_process,
            [(path, SHARED_LEDGER_ID, per_process, "alpha"),
             (path, SHARED_LEDGER_ID, per_process, "beta")],
        )

    backend = SQLiteBackend(path, timeout=60.0)
    try:
        markers = {entry.data["marker"] for entry in backend.iter_entries()}
        assert markers == {"alpha", "beta"}
    finally:
        backend.close()


@pytest.mark.slow
def test_a_second_process_continues_the_chain_rather_than_restarting_it(tmp_path):
    """Sequential, not concurrent: proves a fresh process picks up the existing tip
    instead of writing a second entry zero."""
    path = create_store(str(tmp_path / "sequential.db"))
    context = multiprocessing.get_context("spawn")

    for _ in range(2):
        with context.Pool(1) as pool:
            pool.starmap(append_in_this_process, [(path, SHARED_LEDGER_ID, 25, "run")])

    backend = SQLiteBackend(path, timeout=60.0)
    try:
        assert_one_unbroken_chain(list(backend.iter_entries()), 50)
    finally:
        backend.close()


# --------------------------------------------------------------------------------------
# identity safety under a construction race
# --------------------------------------------------------------------------------------


def test_a_worker_that_lost_the_entry_zero_race_refuses_to_splice_in_its_own_identity():
    """The gunicorn cold start. Two workers both construct against an empty store, so each
    mints its own ledger_id and neither can detect the clash yet. One wins the race to
    entry zero; the loser must refuse rather than continue someone else's chain under its
    own identity - which would splice two ledgers together, every hash still correct."""
    from signledger.core.exceptions import ConfigurationError

    backend = MemoryBackend()
    try:
        first = Ledger(backend, ledger_id="ledger-one")
        second = Ledger(backend, ledger_id="ledger-two")

        first.append({"n": 0})

        with pytest.raises(ConfigurationError):
            second.append({"n": 1})
    finally:
        backend.close()


def test_the_winner_of_that_race_carries_on_normally():
    backend = MemoryBackend()
    try:
        first = Ledger(backend, ledger_id="ledger-one")
        # Held, not discarded: an abandoned Ledger is finalized, and finalizing closes the
        # backend it was handed. Both workers are alive in the race this models.
        loser = Ledger(backend, ledger_id="ledger-two")
        first.append({"n": 0})

        assert first.append({"n": 1}).sequence == 1
        assert loser.ledger_id == "ledger-two"
    finally:
        backend.close()


def test_the_uniqueness_of_entry_ids_holds_under_thread_pressure():
    """Not a chain property - a birthday-collision guard on uuid4 under real contention."""
    backend = MemoryBackend()
    ledger = Ledger(backend, ledger_id=SHARED_LEDGER_ID, max_cache_entries=0)

    def worker():
        for _ in range(50):
            ledger.append({"x": 1})

    try:
        threads = [threading.Thread(target=worker) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=120)

        identifiers = {entry.entry_id for entry in ledger.iter_entries()}
        assert len(identifiers) == 400
    finally:
        ledger.close()


def test_a_generated_ledger_id_really_is_a_uuid():
    instance = Ledger()
    try:
        assert uuid.UUID(instance.ledger_id).version == 4
    finally:
        instance.close()
