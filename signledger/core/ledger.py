"""``Ledger`` - the public surface of a tamper-evident log (contract section 9).

Every rule enforced here is a v1 defect written down:

* ``auto_verify`` defaults to **off**, and turning it on *requires* a failure handler.
  v1 defaulted it on and swallowed the ``IntegrityError`` it raised into a log line.
* ``enable_signatures=True`` without a signer fails at **construction**, not on the first
  append months later.
* ``get_stats()`` reports what was measured. v1 hardcoded ``integrity_verified = True``.
* Subscribers run on a notifier thread, never under the write lock. v1 called them while
  holding a non-reentrant lock, so a subscriber that appended deadlocked the ledger
  permanently.
* The cache re-checks an entry's digest before handing it back, and verification never
  reads the cache at all.
* A worker that cold starts against an empty store, mints an identity and then loses the
  race to entry zero adopts the identity the store ended up with, once. 2.0.0 froze the
  minted id at construction and had no way back, so three of four ``gunicorn`` workers
  raised ``ConfigurationError`` on every append for the life of the process.
* A construction that fails closes the backend it adopted. ``close()`` owns it, so the
  one path that raised must not also be the one path that leaked it.
"""

import queue
import threading
import uuid
import weakref
from collections import OrderedDict
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from signledger.anchor.checkpoint import Checkpoint
from signledger.anchor.sinks import checkpoint_signing_payload
from signledger.backends.memory import MemoryBackend
from signledger.core.entry import CURRENT_FORMAT_VERSION, Entry
from signledger.core.exceptions import ConfigurationError, IntegrityError, ValidationError
from signledger.core.verification import AssuranceLevel, VerificationReport, verify_chain
from signledger.crypto.canonical import canonicalize
from signledger.crypto.hashing import DEFAULT_ALGORITHM, HashChain, genesis_hash
from signledger.crypto.merkle import MerkleTree

__all__ = ["Entry", "Ledger"]

#: Sentinel that tells the notifier thread to finish.
_STOP = object()

#: Default depth of the notification queue. Bounded so a slow subscriber sheds load
#: instead of stalling every writer behind it.
DEFAULT_NOTIFY_QUEUE_SIZE = 1000

#: Appended to every refusal to write a second chain into one store. A message that names
#: the fault but not the fix leaves an operator staring at a 500 with nowhere to go, which
#: is what 2.0.0 shipped. It names no identifier and no payload (contract section 10).
_IDENTITY_REMEDY = (
    "give every process that writes this store the same explicit ledger_id=, or point "
    "this ledger at a store of its own"
)


def _utcnow() -> datetime:
    """The current instant, always timezone-aware."""
    return datetime.now(timezone.utc)


def _entry_bytes(entry: Entry) -> int:
    """Roughly what holding this entry in memory costs.

    Measured from the entry itself rather than guessed at, so the byte budget is a
    real bound and not a decorative constant.
    """
    scalars = (
        entry.ledger_id,
        entry.entry_id,
        entry.previous_hash,
        entry.hash,
        entry.key_id or "",
        entry.signature_algorithm or "",
        entry.signature or "",
    )
    return (
        sum(len(text) for text in scalars)
        + len(canonicalize(entry.data))
        + len(canonicalize(entry.metadata))
    )


class Ledger:
    """An append-only, hash-chained log over a :class:`~signledger.backends.base.StorageBackend`."""

    def __init__(
        self,
        backend: Any = None,
        *,
        ledger_id: Optional[str] = None,
        hash_algorithm: str = DEFAULT_ALGORITHM,
        enable_signatures: bool = False,
        signer: Any = None,
        key_resolver: Any = None,
        auto_verify: bool = False,
        verify_interval: float = 3600,
        on_integrity_failure: Any = None,
        anchor_sink: Any = None,
        max_cache_entries: int = 1000,
        max_cache_bytes: Optional[int] = None,
        notify_queue_size: int = DEFAULT_NOTIFY_QUEUE_SIZE,
    ) -> None:
        # Adopted first, and released again by every path out of here that raises.
        # ``close()`` owns the backend whether the caller supplied it or not, so a
        # constructor that fails must not be the one path that leaves it open: 1000
        # refused constructions held 227 descriptors between them, and every one of
        # them belonged to a ledger object that no longer existed.
        self.backend = MemoryBackend() if backend is None else backend
        try:
            self._configure(
                ledger_id=ledger_id,
                hash_algorithm=hash_algorithm,
                enable_signatures=enable_signatures,
                signer=signer,
                key_resolver=key_resolver,
                auto_verify=auto_verify,
                verify_interval=verify_interval,
                on_integrity_failure=on_integrity_failure,
                anchor_sink=anchor_sink,
                max_cache_entries=max_cache_entries,
                max_cache_bytes=max_cache_bytes,
                notify_queue_size=notify_queue_size,
            )
        except BaseException:
            # Nothing after the finalizer is registered can fail except starting the
            # verifier thread, and a thread that failed to start is not running - so
            # releasing the store is the whole of the cleanup.
            self.backend.close()
            raise

    def _configure(
        self,
        *,
        ledger_id: Optional[str],
        hash_algorithm: str,
        enable_signatures: bool,
        signer: Any,
        key_resolver: Any,
        auto_verify: bool,
        verify_interval: float,
        on_integrity_failure: Any,
        anchor_sink: Any,
        max_cache_entries: int,
        max_cache_bytes: Optional[int],
        notify_queue_size: int,
    ) -> None:
        """The body of ``__init__``, split out so one ``except`` can guard all of it."""
        # Every check below runs before anything is opened or started, so a
        # misconfigured ledger never reaches the point of writing a single entry.
        if enable_signatures and signer is None:
            raise ConfigurationError(
                "enable_signatures=True requires a signer; a ledger that cannot sign "
                "must not pretend it is signing"
            )
        if auto_verify and on_integrity_failure is None:
            raise ConfigurationError(
                "auto_verify=True requires on_integrity_failure; a verifier whose findings "
                "go nowhere is worse than no verifier"
            )
        if max_cache_entries < 0:
            raise ConfigurationError(
                f"max_cache_entries must be zero or greater, got {max_cache_entries}"
            )
        if max_cache_bytes is not None and max_cache_bytes < 0:
            raise ConfigurationError(
                f"max_cache_bytes must be zero or greater, got {max_cache_bytes}"
            )
        if notify_queue_size < 1:
            raise ConfigurationError(
                f"notify_queue_size must be at least 1, got {notify_queue_size}"
            )
        if verify_interval <= 0:
            raise ConfigurationError(
                f"verify_interval must be greater than zero, got {verify_interval}"
            )

        # Raises ConfigurationError for an unknown algorithm.
        self._chain = HashChain(hash_algorithm)
        self.auto_verify = auto_verify
        self._signer = signer
        self._enable_signatures = enable_signatures
        self._key_resolver = key_resolver
        self._anchor_sink = anchor_sink
        self._max_cache_entries = max_cache_entries
        self._max_cache_bytes = max_cache_bytes
        self._on_integrity_failure = on_integrity_failure
        self._verify_interval = verify_interval

        self._cache_lock = threading.RLock()
        self._cache: OrderedDict[int, Entry] = OrderedDict()
        self._cache_ids: Dict[str, int] = {}
        self._cache_bytes = 0
        self._subscriber_lock = threading.RLock()
        self._subscribers: List[Callable[[Entry], None]] = []
        self._events: queue.Queue[Any] = queue.Queue(maxsize=notify_queue_size)
        self._notifier: Optional[threading.Thread] = None
        self._verifier: Optional[threading.Thread] = None
        self._verifier_stop = threading.Event()
        self._dropped = 0
        self._subscriber_errors = 0
        self._last_subscriber_error: Optional[str] = None
        self._last_verify_error: Optional[str] = None
        self._derived_resolver: Any = None
        self._last_report: Optional[VerificationReport] = None
        self._last_verified_at: Optional[datetime] = None

        #: True when this process invented the identity rather than being given one or
        #: reading one out of the store. Only a minted identity may be given up again.
        self._minted_identity = False
        #: True once this ledger has itself written an entry. After that a tip belonging
        #: to someone else is a forked chain, not a cold start.
        self._has_appended = False

        self.ledger_id = self._resolve_identity(ledger_id)
        self._load_cache()

        # weakref.finalize runs its callback at most once, which is what makes close()
        # idempotent, and it must not capture ``self`` or the ledger could never be
        # collected.
        self._resources = _Resources(self.backend)
        self._finalizer = weakref.finalize(self, _shutdown, self._resources)

        if self.auto_verify:
            self._start_verifier()

    def _resolve_identity(self, ledger_id: Optional[str]) -> str:
        """Adopt the identity already written in the store, or mint a new one.

        A store that already holds entries owns its identity: inventing a fresh one would
        make entry 0 fail its genesis check forever, on a ledger nobody had touched.

        An **empty** store has no identity to adopt, so a caller who named none gets a
        minted one - and that is recorded, because a minted identity is the only kind
        this ledger may later give up (see :meth:`_adopt_or_refuse`).
        """
        oldest = self.backend.get_oldest()
        if oldest is None:
            if ledger_id is None:
                self._minted_identity = True
                return str(uuid.uuid4())
            return ledger_id
        if ledger_id is not None and ledger_id != oldest.ledger_id:
            raise ConfigurationError(
                "the requested ledger_id is not the one this store was written with; "
                f"refusing to append a second chain into it - {_IDENTITY_REMEDY}"
            )
        return str(oldest.ledger_id)

    def _adopt_or_refuse(self, tip: Entry) -> None:
        """Take over the identity the store already holds, or refuse to fork the chain.

        Called from inside the backend transaction, with the tip that transaction read.

        Against an empty store there is nothing to adopt, so every process mints its own
        uuid4 and exactly one of them wins entry zero. 2.0.0 froze that minted id at
        construction and gave ``_seal`` no way back, so each loser raised
        ``ConfigurationError`` on *every* append for the rest of its life: measured,
        Flask + ``gunicorn -w 4`` + SQLite served 129 of 300 requests and FastAPI +
        ``uvicorn --workers 4`` + PostgreSQL served exactly 75 - one live worker out of
        four, permanently. Nothing was wrong with the three that died; the store had
        simply acquired a name while they were not looking.

        So a ledger that minted its own identity and has not yet written anything under
        it gives it up and adopts the tip's. ``tip`` came out of the store inside this
        transaction, which makes it the store's own answer to who owns the chain - and
        asking the backend again from in here would re-enter a connection that is
        mid-transaction, or take a second one from a pool that may only have the one.

        The refusal stays for the two cases where the disagreement is real:

        * the identity was **chosen by the caller** - a genuine misconfiguration, and
          adopting would silently splice two logs an operator deliberately kept apart;
        * this ledger has **already written entries** of its own, so the chain forked
          underneath it, which is an integrity event and must stay loud.
        """
        if self._minted_identity and not self._has_appended:
            self.ledger_id = tip.ledger_id
            return
        raise ConfigurationError(
            "the chain in this store belongs to a different ledger; refusing to "
            f"continue it under this ledger's identity - {_IDENTITY_REMEDY}"
        )

    # -- append ------------------------------------------------------------------------

    def append(
        self,
        data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        *,
        timestamp: Optional[datetime] = None,
        entry_id: Optional[str] = None,
    ) -> Entry:
        """Seal ``data`` into the next entry and store it.

        The tip is read and the row inserted inside **one** backend transaction. The
        ledger never reads the head itself and then appends: that is a read-modify-write,
        and it is exactly how two v1 workers forked a chain within seconds of each other,
        permanently and with no repair path.
        """
        moment = _utcnow() if timestamp is None else timestamp
        if not isinstance(moment, datetime):
            # Entry would accept an ISO string, but the genesis preimage is built from
            # this value directly, so a string would anchor entry 0 to a digest a reopened
            # ledger could never reproduce.
            raise ValidationError(f"timestamp must be a datetime, got {type(moment).__name__}")

        def build(tip: Optional[Entry]) -> Entry:
            return self._seal(tip, data, metadata, moment, entry_id)

        entry: Entry = self.backend.append_atomic(build)
        # Set only once the store has actually taken the entry. From here on this ledger
        # has a stake in the chain, and a tip belonging to anyone else is a fork.
        self._has_appended = True
        self._remember(entry)
        # Outside every lock this method took, and handed to a different thread.
        self._publish_event(entry)
        return entry

    def _seal(
        self,
        tip: Optional[Entry],
        data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]],
        moment: datetime,
        entry_id: Optional[str],
    ) -> Entry:
        """Build, hash and sign the entry that follows ``tip``.

        Called *inside* the backend transaction and re-invoked on a retry, so it must stay
        free of side effects - with one exception, :meth:`_adopt_or_refuse`, which is
        idempotent: the second call sees an identity that already matches the tip.
        """
        if tip is None:
            sequence = 0
            # Entry 0's own timestamp is the ledger's creation instant. Both it and
            # ledger_id are inside the preimage, so a reopened ledger can recompute this
            # anchor from the store and an attacker cannot move it without breaking
            # entry 0's digest - and therefore every link after it.
            previous_hash = genesis_hash(
                self.ledger_id,
                moment,
                CURRENT_FORMAT_VERSION,
                algorithm=self._chain.algorithm,
            )
        else:
            if tip.ledger_id != self.ledger_id:
                self._adopt_or_refuse(tip)
            sequence = tip.sequence + 1
            previous_hash = tip.hash

        signer = self._signer if self._enable_signatures else None
        draft = Entry(
            ledger_id=self.ledger_id,
            sequence=sequence,
            entry_id=str(uuid.uuid4()) if entry_id is None else entry_id,
            timestamp=moment,
            data=data,
            metadata={} if metadata is None else metadata,
            previous_hash=previous_hash,
            # Contract section 6: the signing identity is in the preimage, so it cannot be
            # swapped for another key without invalidating the hash.
            key_id=None if signer is None else signer.key_id,
            signature_algorithm=None if signer is None else signer.algorithm,
        )
        sealed = draft.model_copy(update={"hash": self._chain.calculate_hash(draft)})
        if signer is None:
            return sealed
        # Imported here, not at module scope. ``signledger.crypto.signatures`` imports
        # ``cryptography``, which is the optional ``signatures`` extra - a module-scope
        # import would make every `import signledger` load it, including for the large
        # majority of users who never sign anything. This line only runs when a signer
        # was actually configured, by which point the extra is necessarily installed.
        from signledger.crypto.signatures import signing_payload

        signature = signer.sign(signing_payload(sealed.ledger_id, sealed.hash))
        return sealed.model_copy(update={"signature": signature})

    # -- reads -------------------------------------------------------------------------

    def count(self) -> int:
        """The number of entries in the store."""
        return int(self.backend.count())

    def head(self) -> Optional[Entry]:
        """The newest entry, or ``None`` if the ledger is empty."""
        return self._remember(self.backend.get_head())

    def get_by_sequence(self, sequence: int) -> Optional[Entry]:
        """The entry at ``sequence``, from the cache when it can be trusted."""
        cached = self._cached(sequence)
        if cached is not None:
            return cached
        return self._remember(self.backend.get_by_sequence(sequence))

    def get_entry(self, entry_id: str) -> Optional[Entry]:
        """The entry with ``entry_id``, from the cache when it can be trusted."""
        with self._cache_lock:
            sequence = self._cache_ids.get(entry_id)
        if sequence is not None:
            cached = self._cached(sequence)
            if cached is not None and cached.entry_id == entry_id:
                return cached
        return self._remember(self.backend.get_entry(entry_id))

    def iter_entries(self, **window: Any) -> Iterator[Entry]:
        """Iterate the store directly. Windowed reads are never served from the cache."""
        return self.backend.iter_entries(**window)

    # -- cache -------------------------------------------------------------------------

    def _cached(self, sequence: int) -> Optional[Entry]:
        """Return the cached entry at ``sequence`` **only if it still hashes correctly**.

        A cached entry is a copy of something that was once valid. ``Entry`` is only
        shallow-frozen, so ``entry.data[...] = x`` still works and would leave a cache
        holding a record that no longer matches its own digest. Re-checking on the way out
        is what stops the cache from becoming a way to launder a corrupted entry - v1
        served the cache unconditionally.
        """
        with self._cache_lock:
            entry = self._cache.get(sequence)
            if entry is None:
                return None
            try:
                intact = self._chain.verify_hash(entry)
            except ValidationError:
                # An entry this build cannot hash cannot be vouched for either.
                intact = False
            if not intact:
                self._forget(sequence)
                return None
            self._cache.move_to_end(sequence)
            return entry

    def _remember(self, entry: Optional[Entry]) -> Optional[Entry]:
        """Add ``entry`` to the cache and return it unchanged."""
        if entry is None or self._max_cache_entries == 0:
            return entry
        with self._cache_lock:
            if entry.sequence in self._cache:
                self._forget(entry.sequence)
            self._cache[entry.sequence] = entry
            self._cache_ids[entry.entry_id] = entry.sequence
            self._cache_bytes += _entry_bytes(entry)
            self._evict()
        return entry

    def _forget(self, sequence: int) -> None:
        """Drop one entry from the cache. The caller holds the lock."""
        entry = self._cache.pop(sequence)
        self._cache_ids.pop(entry.entry_id, None)
        self._cache_bytes -= _entry_bytes(entry)

    def _evict(self) -> None:
        """Enforce both budgets, oldest first. The caller holds the lock."""
        while len(self._cache) > self._max_cache_entries:
            self._forget(next(iter(self._cache)))
        while (
            self._max_cache_bytes is not None
            and self._cache
            and self._cache_bytes > self._max_cache_bytes
        ):
            self._forget(next(iter(self._cache)))

    def _load_cache(self) -> None:
        """Warm the cache with the **newest** entries.

        v1 fetched the oldest N, so the cache reliably held the entries nobody was asking
        for while every read of a recent entry went to disk.
        """
        if self._max_cache_entries == 0:
            return
        total = self.backend.count()
        if total == 0:
            return
        for entry in self.backend.iter_entries(start_seq=max(0, total - self._max_cache_entries)):
            self._remember(entry)

    # -- checkpoints -------------------------------------------------------------------

    def checkpoint(self) -> Checkpoint:
        """Record the current head, sign it, store it and publish it to the anchor sink.

        The published record is what makes a **truncated tail** detectable. Deleting the
        last N entries leaves a chain that verifies perfectly against itself; only a
        statement made earlier, and kept somewhere the adversary does not control, can
        contradict it.
        """
        head = self.backend.get_head()
        if head is None:
            raise ValidationError("cannot checkpoint an empty ledger; there is no head to record")

        entries = list(self.backend.iter_entries())
        tree = MerkleTree(
            (bytes.fromhex(entry.hash) for entry in entries), algorithm=self._chain.algorithm
        )
        signer = self._signer if self._enable_signatures else None
        checkpoint = Checkpoint(
            ledger_id=self.ledger_id,
            sequence=head.sequence,
            head_hash=head.hash,
            entry_count=len(entries),
            # ``root``, not ``merkle_tree_hash``: the root binds the leaf count, so it
            # cannot be replayed as the root of a shorter chain.
            merkle_root=tree.root,
            timestamp=_utcnow(),
            signature=None,
            key_id=None if signer is None else signer.key_id,
            format_version=CURRENT_FORMAT_VERSION,
        )
        if signer is not None:
            checkpoint = replace(
                checkpoint, signature=signer.sign(checkpoint_signing_payload(checkpoint))
            )

        self.backend.save_checkpoint(checkpoint)
        if self._anchor_sink is not None:
            self._anchor_sink.publish(checkpoint)
        return checkpoint

    # -- integrity ---------------------------------------------------------------------

    def check_integrity(self, *, from_sequence: Optional[int] = None) -> VerificationReport:
        """Verify the chain and return a report. **Never raises for integrity reasons.**

        Reads go straight to the backend. The cache is not consulted, at all: v1 answered
        verification queries out of an unvalidated cache, so on-disk tampering was
        invisible to the one function whose entire job was to find it.
        """
        checkpoint, anchored = self._newest_checkpoint()
        report = verify_chain(
            self.backend,
            chain=self._chain,
            genesis=self._genesis(),
            from_sequence=from_sequence,
            enable_signatures=self._enable_signatures,
            key_resolver=self._resolver(),
            checkpoint=checkpoint,
            anchored=anchored,
            capabilities=self.backend.capabilities(),
            ledger_id=self.ledger_id,
        )
        self._last_report = report
        self._last_verified_at = _utcnow()
        return report

    def verify_integrity(self, *, from_sequence: Optional[int] = None) -> bool:
        """``check_integrity`` as a boolean, raising :class:`IntegrityError` on failure."""
        report = self.check_integrity(from_sequence=from_sequence)
        if not report.ok:
            # Failure messages carry sequences, field names and digests only - never a
            # payload (contract section 10), so the whole list is safe to surface.
            raise IntegrityError(
                f"{len(report.failures)} integrity failure(s): "
                + "; ".join(failure.message for failure in report.failures)
            )
        return True

    def assurance_level(self) -> AssuranceLevel:
        """What this deployment's evidence is actually worth, measured right now.

        Computed by running a real verification. v1 hardcoded the answer, which meant the
        number reported to an auditor had never been checked against anything.
        """
        return self.check_integrity().assurance

    def _genesis(self) -> str:
        """The digest entry 0 must link back to.

        Derived from entry 0's own ``ledger_id`` and ``timestamp``, both of which are
        inside its preimage. Moving either breaks entry 0's hash, and therefore every link
        that follows it.
        """
        oldest = self.backend.get_oldest()
        if oldest is None:
            # Nothing to anchor. The report will refuse an empty range regardless.
            return ""
        return genesis_hash(
            oldest.ledger_id,
            oldest.timestamp,
            oldest.format_version,
            algorithm=self._chain.algorithm,
        )

    def _newest_checkpoint(self) -> Tuple[Optional[Any], bool]:
        """The checkpoint to verify against, and whether it came from outside this store.

        The anchored copy used to win unconditionally, on the reasoning that the local one
        sits in the same database as the entries and so is controlled by anyone who can
        truncate them. That is true, and it is only half the picture: preferring the anchor
        no matter what hands the same power to anyone who can reach the anchor *file*, who
        needs only to roll it back to an older genuine checkpoint to have the store's own
        contradicting evidence discarded unread. Measured: the local table still said
        sequence 7 while a rolled-back anchor said 3, and verification answered with the
        anchor's story.

        So both are consulted and the **newer** wins, which makes a rollback cost the
        attacker *both* copies rather than whichever one they happen to reach.

        An anchor that is behind, or that holds nothing at all, returns ``anchored=False``
        rather than a failure. It is not damage to the chain: the commonest cause by far is
        a publish that did not land, and an integrity report is the wrong place to raise an
        operational alarm. What it does cost is the ``ANCHORED``/``WORM`` label, which is
        the honest answer - nobody outside this process is holding a copy of this head, so
        the evidence is worth exactly what a local signature is worth.
        """
        local = self.backend.get_latest_checkpoint()
        if self._anchor_sink is None:
            return local, False

        external = self._anchor_sink.fetch_latest(self.ledger_id)
        if external is None:
            return local, False
        if local is not None and local.sequence > external.sequence:
            return local, False
        return external, True

    def _resolver(self) -> Any:
        """The key resolver verification should use.

        With signatures on and no resolver supplied, the ledger trusts its own signer's
        public key. That is safe against the adversary signatures exist to stop - someone
        with write access to the *store* - and it keeps ``Ledger(enable_signatures=True,
        signer=...)`` from reporting every entry as UNKNOWN_KEY.
        """
        if self._key_resolver is not None:
            return self._key_resolver
        if not self._enable_signatures:
            return None
        if self._derived_resolver is None:
            from signledger.crypto.keys import InMemoryKeyResolver

            derived = InMemoryKeyResolver()
            derived.add_pem(self._signer.public_key_pem())
            self._derived_resolver = derived
        return self._derived_resolver

    # -- subscribers -------------------------------------------------------------------

    def subscribe(self, callback: Callable[[Entry], None]) -> None:
        """Call ``callback`` with each appended entry, from the notifier thread.

        Never from the appending thread and never under a lock. v1 invoked subscribers
        while holding a non-reentrant write lock, so the first subscriber that appended
        deadlocked the ledger forever.
        """
        with self._subscriber_lock:
            self._subscribers.append(callback)
        self._start_notifier()

    def unsubscribe(self, callback: Callable[[Entry], None]) -> None:
        """Stop delivering entries to ``callback``. Unknown callbacks are ignored."""
        with self._subscriber_lock:
            if callback in self._subscribers:
                self._subscribers.remove(callback)

    def _start_notifier(self) -> None:
        """Start the notifier thread once, on the first subscription."""
        with self._subscriber_lock:
            if self._notifier is not None:
                return
            worker = threading.Thread(
                target=self._notify_loop, name=f"signledger-notifier-{self.ledger_id[:8]}",
                daemon=True,
            )
            self._notifier = worker
            self._resources.events = self._events
            self._resources.notifier = worker
        worker.start()

    def _publish_event(self, entry: Entry) -> None:
        """Hand ``entry`` to the notifier queue without ever blocking the writer."""
        with self._subscriber_lock:
            if not self._subscribers:
                return
        try:
            self._events.put_nowait(entry)
        except queue.Full:
            # The queue is bounded on purpose. Blocking here would stall an append behind
            # a slow subscriber, which is a worse failure than a missed notification - and
            # the drop is counted rather than hidden.
            self._dropped += 1

    def _notify_loop(self) -> None:
        while True:
            item = self._events.get()
            if item is _STOP:
                return
            with self._subscriber_lock:
                targets = list(self._subscribers)
            for callback in targets:
                try:
                    callback(item)
                except Exception as err:  # noqa: BLE001 - a subscriber bug is not ours
                    # Broad on purpose: third-party callback code must not be able to tear
                    # down the notification thread for every other subscriber. The failure
                    # is counted and surfaced by get_stats(), never swallowed silently.
                    self._subscriber_errors += 1
                    self._last_subscriber_error = type(err).__name__

    # -- automatic verification --------------------------------------------------------

    def _start_verifier(self) -> None:
        worker = threading.Thread(
            target=self._verify_loop, name=f"signledger-verifier-{self.ledger_id[:8]}",
            daemon=True,
        )
        self._verifier = worker
        self._resources.verifier_stop = self._verifier_stop
        self._resources.verifier = worker
        worker.start()

    def _verify_loop(self) -> None:
        while not self._verifier_stop.wait(self._verify_interval):
            try:
                report = self.check_integrity()
            except Exception as err:  # noqa: BLE001 - a dead verifier is a silent failure
                self._last_verify_error = type(err).__name__
                continue
            if report.ok:
                continue
            try:
                self._on_integrity_failure(report)
            except Exception as err:  # noqa: BLE001
                # A handler that throws must not stop the ledger being watched - that is
                # exactly how v1 ended up monitoring nothing while looking healthy.
                self._last_verify_error = type(err).__name__

    # -- statistics --------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Report real, measured state.

        Backend errors are **propagated**. v1 caught them and returned healthy-looking
        zeros, so a ledger whose store had gone away still reported itself fine.
        """
        report = self.check_integrity()
        stats = self.get_stats_snapshot()
        stats.update(
            {
                "entry_count": self.backend.count(),
                "size_bytes": self.backend.size_bytes(),
                "capabilities": self.backend.capabilities(),
                "integrity_ok": report.ok,
                "entries_verified": report.entries_verified,
                "last_verification_time": self._last_verified_at,
                "last_failure": report.failures[0].message if report.failures else None,
                "assurance_level": report.assurance,
            }
        )
        return stats

    def get_stats_snapshot(self) -> Dict[str, Any]:
        """The cheap half of :meth:`get_stats` - process state only, no backend reads."""
        return {
            "ledger_id": self.ledger_id,
            "hash_algorithm": self._chain.algorithm,
            "signatures_enabled": self._enable_signatures,
            "auto_verify": self.auto_verify,
            "anchor_sink": None if self._anchor_sink is None else self._anchor_sink.name,
            "cached_entries": len(self._cache),
            "cached_bytes": self._cache_bytes,
            "dropped_notifications": self._dropped,
            "subscriber_errors": self._subscriber_errors,
            "last_subscriber_error": self._last_subscriber_error,
            "last_verify_error": self._last_verify_error,
        }

    # -- lifecycle ---------------------------------------------------------------------

    def close(self) -> None:
        """Stop the notifier and the verifier, then close the backend. Idempotent.

        Idempotency comes from :func:`weakref.finalize`, which guarantees the callback
        runs at most once however many times it is invoked.
        """
        self._finalizer()

    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()


class _Resources:
    """The things ``close()`` has to release, held apart from the ``Ledger`` itself.

    ``weakref.finalize`` must not reference the object it is finalizing, so the backend
    and the two worker threads live here instead.
    """

    def __init__(self, backend: Any) -> None:
        self.backend = backend
        self.events: Optional[queue.Queue[Any]] = None
        self.notifier: Optional[threading.Thread] = None
        self.verifier_stop: Optional[threading.Event] = None
        self.verifier: Optional[threading.Thread] = None


def _shutdown(resources: _Resources) -> None:
    """Stop the workers, then close the store. Runs exactly once per ledger."""
    if resources.notifier is not None and resources.events is not None:
        resources.events.put(_STOP)
        resources.notifier.join(timeout=5)
    if resources.verifier is not None and resources.verifier_stop is not None:
        resources.verifier_stop.set()
        resources.verifier.join(timeout=5)
    resources.backend.close()
