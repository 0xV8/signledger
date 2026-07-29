"""The ``signledger`` command line.

This is a module, never a package. v1 shipped an empty ``signledger/cli/`` directory
alongside it, which shadowed this file and broke the console script outright.

What the v1 CLI could do was open an in-memory backend, which meant ``signledger verify``
could not be pointed at a real database at all - and, run with no arguments, it printed a
success tick against an empty ledger. Both are fixed here: every command takes a backend
URI, and an empty ledger is never reported as valid.

Exit codes are the contract with a shell script:

* ``0`` - the ledger (or bundle) verified.
* ``1`` - it did not. Tampering, truncation, a broken link, or nothing to verify.
* ``2`` - the command could not run: a bad URI, a missing driver, an unreadable file.

Two properties every command here has, because four consumer projects found out what
happens without them:

* **A command names the store the library named.** The database is the mongodb URI's own
  path component, and the table (the *collection*, on MongoDB) is ``--entries-table`` /
  ``--checkpoints-table``. Without both, ``signledger verify`` answered from whatever the
  backend defaults happened to be - which is a success report about a store nobody asked
  about, and there is no worse answer this tool can give.
* **A command that reads does not write.** ``verify``, ``stats``, ``inspect``, ``export``
  and ``anchor`` open the store with ``create=False``, and every backend honours it: no
  ``CREATE TABLE``, no trigger, no index, no validator, and on SQLite not even
  ``PRAGMA journal_mode = WAL``, which is itself a write to the database header. An
  integrity check that provisions the thing it is checking cannot be run from a read-only
  role or against a hot standby, which is exactly where a careful operator runs it - and
  a mistyped DSN answered "your ledger is empty" (exit 1, a verdict about integrity)
  where the truth was "that store does not exist" (exit 2, an operational error). Being
  told your audit log is gone when you only mistyped its name is the second-worst answer
  this tool can give.
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, cast
from urllib.parse import parse_qsl, quote, unquote, urlsplit

from signledger import __version__
from signledger.core.exceptions import ConfigurationError

EXIT_OK = 0
EXIT_INVALID = 1
EXIT_ERROR = 2

#: The on-disk format ``export --bundle`` writes and ``verify-bundle`` reads.
BUNDLE_FORMAT = "signledger/bundle/1"

#: Every field a bundle must carry.
_BUNDLE_FIELDS = (
    "format",
    "ledger_id",
    "hash_algorithm",
    "genesis_hash",
    "entry_count",
    "merkle_root",
    "entries",
    "checkpoint",
    "public_keys",
)

#: The format ``migrate --record`` writes.
MIGRATION_FORMAT = "signledger/migration/1"

#: Every field of a migration record, in the order they are documented.
_MIGRATION_FIELDS = (
    "format",
    "ledger_id",
    "source_scheme",
    "target_scheme",
    "source_head_hash",
    "source_entry_count",
    "source_format_versions",
    "hash_algorithm",
    "migrated_at",
    "key_id",
    "signature",
)

#: Why a bundle's own keys cannot raise its assurance level.
EMBEDDED_KEY_NOTE = (
    "the signatures were checked against keys carried inside the bundle, which proves "
    "internal consistency only; supply --key with a key obtained separately to learn "
    "who signed"
)

#: What each assurance level is actually worth, in words. Printed next to the level so a
#: reader cannot mistake CHAIN_ONLY for "tamper-proof" (contract section 1).
ASSURANCE_NOTES = {
    "none": "nothing has been established",
    "chain_only": (
        "detects accidental corruption and naive edits only - anyone with write access "
        "to the store can rewrite history and re-derive every digest"
    ),
    "signed": "every entry carries a signature that verified against a resolved key",
    "anchored": "signed, and a checkpoint has been published outside this store",
    "worm": "anchored, and the store itself refuses updates and deletes",
}

__all__ = [
    "ASSURANCE_NOTES",
    "BUNDLE_FORMAT",
    "EMBEDDED_KEY_NOTE",
    "EXIT_ERROR",
    "EXIT_INVALID",
    "EXIT_OK",
    "MIGRATION_FORMAT",
    "build_parser",
    "find_forks",
    "main",
    "migration_signing_payload",
    "open_backend",
]


def _utcnow() -> datetime:
    """The current instant, always timezone-aware."""
    return datetime.now(timezone.utc)


@dataclass
class Result:
    """What a subcommand decided: an exit code, a JSON payload and human lines."""

    code: int
    payload: Dict[str, Any]
    lines: Tuple[str, ...] = field(default_factory=tuple)


# --------------------------------------------------------------------------------------
# backend URIs
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Store:
    """What the operator asked for beyond the URI itself.

    ``entries`` and ``checkpoints`` are the table names - collection names, on MongoDB.
    ``create`` is False for a command that only reads.
    """

    entries: Optional[str] = None
    checkpoints: Optional[str] = None
    create: bool = True


#: Option names that belong to signledger rather than to a driver. Written into a URI
#: query string they reach the driver instead: libpq refuses the whole DSN (which the
#: backend reports as ``connect failed: ProgrammingError``, naming neither the option nor
#: the fix), and pymongo warns, drops it, and reads the default collection anyway.
_STORE_OPTION_KEYS = frozenset(
    {"entries_table", "entries_collection", "checkpoints_table", "checkpoints_collection"}
)


def open_backend(
    uri: str,
    *,
    entries_table: Optional[str] = None,
    checkpoints_table: Optional[str] = None,
    create: bool = True,
) -> Any:
    """Open the storage backend named by ``uri``.

    ``memory://``, ``sqlite:<path>``, ``postgresql://…``, ``mongodb://…/<database>``.

    ``entries_table`` and ``checkpoints_table`` name the table the ledger lives in - the
    *collection*, on MongoDB. Both are constructor arguments of every real backend, and a
    deployment that used them had no CLI at all until they were plumbed through here.

    ``create=False`` opens a store that already exists and refuses to bring one into
    being. Every read-only command passes it, and all three real backends honour it in
    their constructors - the only place that could, since that is where each of them used
    to provision. A store that is not there raises ``StorageError`` naming what is
    missing, which reaches the shell as exit 2 rather than as a verdict about a ledger.

    No error raised here ever quotes the URI: a DSN routinely carries a password, and
    contract section 10 forbids a credential reaching a message (or a CI log).
    """
    scheme = urlsplit(uri).scheme
    if not scheme:
        raise ConfigurationError(
            "a backend URI needs a scheme: memory://, sqlite:<path>, "
            "postgresql://… or mongodb://…"
        )
    opener = _OPENERS.get(scheme)
    if opener is None:
        raise ConfigurationError(
            f"unsupported backend scheme '{scheme}'; expected one of: "
            + ", ".join(sorted(_OPENERS))
        )
    _refuse_store_options_in_uri(uri)
    return opener(uri, _Store(entries_table, checkpoints_table, create))


def _refuse_store_options_in_uri(uri: str) -> None:
    """Refuse a table name written into the URI's query string, by name.

    Only the option *name* reaches the message; the URI itself never does.
    """
    named = sorted(
        {
            key
            for key, _value in parse_qsl(urlsplit(uri).query, keep_blank_values=True)
            if key.strip().lower().replace("-", "_") in _STORE_OPTION_KEYS
        }
    )
    if named:
        raise ConfigurationError(
            "the backend URI carries a signledger option in its query string ("
            + ", ".join(named)
            + "), where only the driver will see it; name a non-default store with "
            "--entries-table / --checkpoints-table (--entries-collection / "
            "--checkpoints-collection reads better on MongoDB, and means the same thing)"
        )


def _named_tables(store: _Store, entries: str, checkpoints: str) -> Dict[str, Any]:
    """The constructor keywords for whichever names were given.

    Passed only when the operator actually named one, so each backend keeps its own
    documented default rather than having it restated here.
    """
    named: Dict[str, Any] = {}
    if store.entries is not None:
        named[entries] = store.entries
    if store.checkpoints is not None:
        named[checkpoints] = store.checkpoints
    return named


def _open_memory(uri: str, store: _Store) -> Any:
    if store.entries is not None or store.checkpoints is not None:
        # Accepting the flag here would silently ignore it, which is how an operator
        # comes to believe they are looking at a store they are not looking at.
        raise ConfigurationError(
            "the memory backend holds no tables to name, so --entries-table and "
            "--checkpoints-table do not apply to memory://"
        )
    from signledger.backends.memory import MemoryBackend

    # The backend owns the refusal, exactly as it does for ``sqlite::memory:``. A
    # read-only command against memory:// used to report "no entries were verified"
    # (exit 1, a verdict about integrity) for a store that had just been created empty;
    # it is now an operational error (exit 2), which is what naming nothing deserves.
    return MemoryBackend(create=store.create)


def _open_sqlite(uri: str, store: _Store) -> Any:
    from signledger.backends.sqlite import SQLiteBackend

    # The backend owns the refusal, because the backend is what would otherwise create
    # the file: ``sqlite3.connect`` on a path that is not there conjures a database, and
    # ``PRAGMA journal_mode = WAL`` rewrites the header of one that is.
    return SQLiteBackend(
        _sqlite_path(uri),
        create=store.create,
        **_named_tables(store, "entries_table", "checkpoints_table"),
    )


def _sqlite_path(uri: str) -> str:
    """The filesystem path a ``sqlite://`` URI names.

    ``sqlite:///var/lib/x.db`` is the absolute path ``/var/lib/x.db`` - the plain URL
    reading, rather than SQLAlchemy's extra-slash convention, because an operator typing a
    path into a shell means the path they typed. ``sqlite:x.db`` is relative to the
    working directory, and ``sqlite://`` on its own is a throwaway in-memory database.
    """
    parts = urlsplit(uri)
    if parts.netloc:
        raise ConfigurationError(
            "a sqlite URI takes a path, not a host; write sqlite:///absolute/path.db "
            "or sqlite:relative/path.db"
        )
    path = parts.path
    if path in ("", "/", "/:memory:", ":memory:"):
        return ":memory:"
    return path


def _open_postgres(uri: str, store: _Store) -> Any:
    try:
        from signledger.backends.postgresql import PostgresBackend
    except ImportError as err:
        raise ConfigurationError(
            "the postgresql backend needs psycopg: pip install 'signledger[postgresql]'"
        ) from err

    _refuse_unknown_connection_options(uri)
    # ``create=False`` runs no DDL and takes no advisory lock, so a read-only command
    # works from a role holding SELECT and nothing else - and a database that holds no
    # ledger is reported as absent instead of being provisioned and called empty.
    return PostgresBackend(
        uri,
        create=store.create,
        **_named_tables(store, "entries_table", "checkpoints_table"),
    )


def _refuse_unknown_connection_options(uri: str) -> None:
    """Name the query parameter libpq will not accept, rather than let it die opaque.

    A rejected DSN reaches the operator as ``StorageError: connect failed:
    ProgrammingError`` - which names neither the option nor the fix, and a mistyped
    connection option is one of the likeliest things to be wrong about a DSN.

    Every key is probed on a URI built here, so nothing from the operator's own DSN but
    the parameter *name* can reach the message (contract section 10).
    """
    from psycopg import ProgrammingError
    from psycopg.conninfo import conninfo_to_dict

    refused = []
    for key, _value in parse_qsl(urlsplit(uri).query, keep_blank_values=True):
        try:
            conninfo_to_dict("postgresql:///?" + quote(key, safe="") + "=probe")
        except ProgrammingError:
            refused.append(key)
    if refused:
        raise ConfigurationError(
            "the backend URI carries connection options PostgreSQL does not accept: "
            + ", ".join(sorted(set(refused)))
            + "; a non-default table is named with --entries-table / --checkpoints-table"
        )


def _open_mongo(uri: str, store: _Store) -> Any:
    try:
        from signledger.backends.mongodb import DEFAULT_DATABASE, MongoBackend
    except ImportError as err:
        raise ConfigurationError(
            "the mongodb backend needs pymongo: pip install 'signledger[mongodb]'"
        ) from err

    named = _named_tables(store, "entries_collection", "checkpoints_collection")
    # Always explicit: the database is the one the URI names, and the backend's own
    # default only when the URI names none. Leaving it to the default unconditionally is
    # what discarded the URI's database in the first place.
    named["database"] = _mongo_database(uri) or DEFAULT_DATABASE
    # ``create=False`` creates no collection, validator or index. MongoDB brings a
    # collection into being on first use, so without it a mistyped database name is
    # answered with an empty ledger rather than with "that store does not exist".
    named["create"] = store.create
    return MongoBackend(uri, **named)


def _mongo_database(uri: str) -> Optional[str]:
    """The database a mongodb URI names, or ``None`` when it names none.

    It is the path component, exactly as every other MongoDB tool reads it. Dropping it
    is how ``verify -b mongodb://host/ledgers`` came to print VERIFIED and exit 0 about
    the contents of ``signledger`` - the backend default, and a store the operator never
    asked about. A verification tool reporting success about the wrong store is the worst
    failure this project has.
    """
    return unquote(urlsplit(uri).path.lstrip("/")) or None


_OPENERS: Dict[str, Callable[[str, _Store], Any]] = {
    "memory": _open_memory,
    "sqlite": _open_sqlite,
    "postgres": _open_postgres,
    "postgresql": _open_postgres,
    "mongodb": _open_mongo,
    "mongodb+srv": _open_mongo,
}


def _open_store(args: argparse.Namespace, *, create: bool) -> Any:
    """Open the store ``--backend`` names, under the table names the command line gave."""
    return open_backend(
        args.backend,
        entries_table=args.entries_table,
        checkpoints_table=args.checkpoints_table,
        create=create,
    )


# --------------------------------------------------------------------------------------
# shared pieces
# --------------------------------------------------------------------------------------


def _ledger_id_of(backend: Any) -> Optional[str]:
    """The identity written into this store, or ``None`` when it is empty."""
    oldest = backend.get_oldest()
    return None if oldest is None else str(oldest.ledger_id)


def _genesis_of(backend: Any, algorithm: str) -> str:
    """The digest entry 0 must link back to.

    Derived from entry 0's own ``ledger_id`` and ``timestamp``, exactly as
    :class:`~signledger.core.ledger.Ledger` derives it - both are inside entry 0's
    preimage, so moving either breaks its digest and every link after it.
    """
    from signledger.crypto.hashing import genesis_hash

    oldest = backend.get_oldest()
    if oldest is None:
        return ""
    return genesis_hash(
        oldest.ledger_id, oldest.timestamp, oldest.format_version, algorithm=algorithm
    )


def _key_resolver(paths: Optional[List[str]]) -> Any:
    """Build a resolver over every PEM named by ``--key``, or ``None`` if none were."""
    if not paths:
        return None
    from signledger.crypto.keys import InMemoryKeyResolver

    resolver = InMemoryKeyResolver()
    for path in paths:
        pems = _pem_files(path)
        if not pems:
            # An empty resolver answers UNKNOWN_KEY for every entry, which reads exactly
            # like tampering. Refusing here keeps a typo from being reported as an attack.
            raise ConfigurationError(f"no .pem files were found in the key directory {path}")
        for pem_path in pems:
            with open(pem_path, encoding="utf-8") as handle:
                resolver.add_pem(handle.read())
    return resolver


def _pem_files(path: str) -> List[str]:
    """The PEM files ``path`` names: itself, or every ``.pem`` in it."""
    if not os.path.isdir(path):
        return [path]
    return sorted(
        os.path.join(path, name) for name in os.listdir(path) if name.endswith(".pem")
    )


def _signer_from_pem(path: str, password_env: Optional[str]) -> Any:
    """Load a private key and wrap it in the signer that matches its type.

    The password comes from the environment and never from ``argv``: a command-line
    argument is visible in ``ps`` output to every user on the machine, and it is written
    to shell history.
    """
    from signledger.crypto.signatures import (
        ECDSASigner,
        Ed25519Signer,
        RSASigner,
        load_pem_private_key,
    )

    password: Optional[bytes] = None
    if password_env is not None:
        raw = os.environ.get(password_env)
        if raw is None:
            raise ConfigurationError(
                f"the signing key password environment variable {password_env} is not set"
            )
        if raw == "":
            # `export SL_PASS=$(cat /secrets/missing)` sets the variable to an empty
            # string, not to nothing. Passing b"" through as a real passphrase produced
            # the generic "the password may be wrong", sending the operator to check a
            # passphrase when the actual fault is an empty variable. Different causes
            # need different messages; the value itself is never echoed either way.
            raise ConfigurationError(
                f"the signing key password environment variable {password_env} is set but "
                f"empty; if the key is unencrypted, omit --password-env"
            )
        password = raw.encode("utf-8")

    with open(path, "rb") as handle:
        private_key = load_pem_private_key(handle.read(), password)

    # Dispatched on what the key *is*, never on a flag the operator has to keep in sync.
    # Every signer re-checks the type it was handed and raises ConfigurationError on a
    # mismatch, so an unsupported key kind ends in a clear refusal rather than a cast.
    curve = getattr(private_key, "curve", None)
    if curve is not None:
        return ECDSASigner(cast(Any, private_key), curve=curve.name)
    if hasattr(private_key, "key_size"):
        return RSASigner(cast(Any, private_key))
    return Ed25519Signer(cast(Any, private_key))


def _anchor_sink(path: Optional[str]) -> Any:
    """A :class:`FileAnchorSink` over ``path``, or ``None``."""
    if path is None:
        return None
    from signledger.anchor.sinks import FileAnchorSink

    return FileAnchorSink(path)


def _newest_checkpoint(backend: Any, sink: Any, ledger_id: Optional[str]) -> Tuple[Any, bool]:
    """The checkpoint to verify against, and whether it came from outside this store.

    An externally anchored checkpoint wins over a locally stored one even when the local
    one is newer: the local copy lives in the database the adversary already controls.
    """
    if sink is not None and ledger_id is not None:
        external = sink.fetch_latest(ledger_id)
        if external is not None:
            return external, True
    return backend.get_latest_checkpoint(), False


def _verify(
    backend: Any,
    *,
    algorithm: str,
    from_sequence: Optional[int] = None,
    resolver: Any = None,
    sink: Any = None,
) -> Tuple[Any, bool]:
    """Run a full verification against the store and report what it found."""
    from signledger.core.verification import verify_chain
    from signledger.crypto.hashing import HashChain

    ledger_id = _ledger_id_of(backend)
    checkpoint, anchored = _newest_checkpoint(backend, sink, ledger_id)
    report = verify_chain(
        backend,
        chain=HashChain(algorithm),
        genesis=_genesis_of(backend, algorithm),
        from_sequence=from_sequence,
        enable_signatures=resolver is not None,
        key_resolver=resolver,
        checkpoint=checkpoint,
        anchored=anchored,
        capabilities=backend.capabilities(),
    )
    return report, anchored


def _failure_payload(report: Any) -> List[Dict[str, Any]]:
    """Every failure, as JSON. Digests and sequences only - never a payload."""
    return [
        {
            "kind": failure.kind.value,
            "sequence": failure.sequence,
            "entry_id": failure.entry_id,
            "expected": failure.expected,
            "actual": failure.actual,
            "message": failure.message,
        }
        for failure in report.failures
    ]


def _report_lines(
    report: Any,
    *,
    ledger_id: Optional[str],
    assurance: Optional[str] = None,
    ok: Optional[bool] = None,
) -> List[str]:
    """The human rendering of a verification report."""
    level = report.assurance.value if assurance is None else assurance
    verified = report.ok if ok is None else ok
    # "entries 3 verified" printed two lines above "result FAILED" reads as reassurance
    # next to a failure. Entries that were read and found wrong were checked, not
    # verified, and only the passing report gets to use the word.
    counted = "verified" if verified else "checked"
    span = (
        f"{report.entries_verified} {counted}"
        if report.first_sequence is None
        else f"{report.entries_verified} {counted} "
        f"(sequence {report.first_sequence}..{report.last_sequence})"
    )
    lines = [
        f"ledger       {'(none)' if ledger_id is None else ledger_id}",
        f"entries      {span}",
        f"assurance    {level} - {ASSURANCE_NOTES[level]}",
    ]
    if verified:
        lines.append("result       VERIFIED")
    else:
        # No tick, ever, when something is wrong - and never one for an empty range.
        lines.append("result       FAILED")
        lines.extend(f"  [{f.kind.value}] {f.message}" for f in report.failures)
    return lines


def _verification_result(
    command: str,
    report: Any,
    *,
    ledger_id: Optional[str],
    anchored: bool,
    capabilities: Any,
    extra_failures: Optional[List[Dict[str, Any]]] = None,
) -> Result:
    """One report, rendered both ways.

    ``extra_failures`` carries checks made outside :func:`verify_chain` - the bundle's
    own declared summary - so a swapped merkle root is as fatal as a broken link.
    """
    extra = list(extra_failures or [])
    failures = _failure_payload(report) + extra
    ok = report.ok and not extra
    level = "none" if not ok else report.assurance.value
    lines = _report_lines(report, ledger_id=ledger_id, assurance=level, ok=ok)
    lines.extend(f"  [{item['kind']}] {item['message']}" for item in extra)
    return Result(
        code=EXIT_OK if ok else EXIT_INVALID,
        payload={
            "command": command,
            "ok": ok,
            "exit_code": EXIT_OK if ok else EXIT_INVALID,
            "ledger_id": ledger_id,
            "entries_verified": report.entries_verified,
            "first_sequence": report.first_sequence,
            "last_sequence": report.last_sequence,
            "assurance": level,
            "assurance_note": ASSURANCE_NOTES[level],
            "anchored": anchored,
            "capabilities": sorted(capabilities),
            "failures": failures,
        },
        lines=tuple(lines),
    )


# --------------------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------------------


def find_forks(entries: Iterable[Any]) -> List[Dict[str, Any]]:
    """Every shape in ``entries`` that a single append-only chain cannot have.

    A fork is what v1 produced when two gunicorn workers each read the head, built the
    next entry from it and wrote: two chains in one store, permanently, with no repair
    path. It is worth looking for separately from verification because the individual
    entries of a forked ledger are all perfectly well formed.

    Reported anomalies: ``duplicate_sequence``, ``duplicate_entry_id``,
    ``duplicate_hash``, ``branch`` (two entries claiming the same parent), ``orphan`` (an
    entry whose parent is not in the store) and ``multiple_heads``.
    """
    records = list(entries)
    by_sequence: Dict[int, List[Any]] = {}
    by_entry_id: Dict[str, List[Any]] = {}
    by_hash: Dict[str, List[Any]] = {}
    by_previous: Dict[str, List[Any]] = {}
    for entry in records:
        by_sequence.setdefault(entry.sequence, []).append(entry)
        by_entry_id.setdefault(entry.entry_id, []).append(entry)
        by_hash.setdefault(entry.hash, []).append(entry)
        by_previous.setdefault(entry.previous_hash, []).append(entry)

    anomalies: List[Dict[str, Any]] = []
    for sequence, group in by_sequence.items():
        if len(group) > 1:
            anomalies.append(
                _anomaly(
                    "duplicate_sequence",
                    [sequence],
                    f"sequence {sequence} is stored {len(group)} times",
                )
            )
    for group in by_entry_id.values():
        if len(group) > 1:
            anomalies.append(
                _anomaly(
                    "duplicate_entry_id",
                    _sequences(group),
                    "one entry id is shared by sequences " + _listed(group),
                )
            )
    for group in by_hash.values():
        if len(group) > 1:
            anomalies.append(
                _anomaly(
                    "duplicate_hash",
                    _sequences(group),
                    "one digest is shared by sequences " + _listed(group),
                )
            )
    for group in by_previous.values():
        if len(group) > 1:
            anomalies.append(
                _anomaly(
                    "branch",
                    _sequences(group),
                    f"sequences {_listed(group)} all claim the same parent; "
                    f"the chain forks here",
                )
            )
    for entry in records:
        # Entry 0 links to the genesis digest, which is deliberately not the hash of any
        # stored entry, so it is never an orphan.
        if entry.sequence != 0 and entry.previous_hash not in by_hash:
            anomalies.append(
                _anomaly(
                    "orphan",
                    [entry.sequence],
                    f"sequence {entry.sequence} links to an entry this store does not hold",
                )
            )
    heads = sorted(entry.sequence for entry in records if entry.hash not in by_previous)
    if len(heads) > 1:
        anomalies.append(
            _anomaly(
                "multiple_heads",
                heads,
                f"{len(heads)} entries are followed by nothing; a chain has exactly one head",
            )
        )
    # Sorted so two runs over the same store produce identical output.
    return sorted(anomalies, key=lambda anomaly: (anomaly["kind"], anomaly["sequences"]))


def _anomaly(kind: str, sequences: List[int], message: str) -> Dict[str, Any]:
    return {"kind": kind, "sequences": sequences, "message": message}


def _sequences(group: List[Any]) -> List[int]:
    return sorted(entry.sequence for entry in group)


def _listed(group: List[Any]) -> str:
    return ", ".join(str(sequence) for sequence in _sequences(group))


def _cmd_verify(args: argparse.Namespace) -> Result:
    backend = _open_store(args, create=False)
    try:
        report, anchored = _verify(
            backend,
            algorithm=args.hash_algorithm,
            from_sequence=args.from_sequence,
            resolver=_key_resolver(args.key),
            sink=_anchor_sink(args.anchor),
        )
        return _verification_result(
            "verify",
            report,
            ledger_id=_ledger_id_of(backend),
            anchored=anchored,
            capabilities=backend.capabilities(),
        )
    finally:
        backend.close()


def _checkpoint_payload(checkpoint: Any) -> Dict[str, Any]:
    """A checkpoint as JSON. Field-for-field what an anchor record carries."""
    return {
        "ledger_id": checkpoint.ledger_id,
        "sequence": checkpoint.sequence,
        "head_hash": checkpoint.head_hash,
        "entry_count": checkpoint.entry_count,
        "merkle_root": checkpoint.merkle_root,
        "timestamp": checkpoint.timestamp.isoformat(),
        "signature": checkpoint.signature,
        "key_id": checkpoint.key_id,
        "format_version": checkpoint.format_version,
    }


def _checkpoint_lines(checkpoint: Any) -> List[str]:
    signed = (
        "no"
        if checkpoint.signature is None
        else f"yes (key {(checkpoint.key_id or '')[:16]}...)"
    )
    return [
        f"ledger       {checkpoint.ledger_id}",
        f"sequence     {checkpoint.sequence}",
        f"head         {checkpoint.head_hash}",
        f"entries      {checkpoint.entry_count}",
        f"merkle root  {checkpoint.merkle_root}",
        f"signed       {signed}",
    ]


def _cmd_checkpoint(args: argparse.Namespace) -> Result:
    from signledger.core.exceptions import ValidationError
    from signledger.core.ledger import Ledger

    # The one command whose whole purpose is to write to the store, so it may create it.
    backend = _open_store(args, create=True)
    closed = False
    try:
        if backend.count() == 0:
            # Not "invalid" - there is simply nothing to record. Reported separately from
            # the verification gate below so the message says what is actually wrong.
            raise ValidationError(
                "cannot checkpoint an empty ledger; there is no head to record"
            )
        if not args.force:
            report, _ = _verify(backend, algorithm=args.hash_algorithm)
            if not report.ok:
                # Publishing a checkpoint over a chain that does not verify would launder
                # the damage into the one record an auditor is meant to be able to trust.
                lines = ["refusing to checkpoint a ledger that does not verify"]
                lines.extend(_report_lines(report, ledger_id=_ledger_id_of(backend)))
                lines.append("pass --force to checkpoint it anyway")
                return Result(
                    code=EXIT_INVALID,
                    payload={
                        "command": "checkpoint",
                        "ok": False,
                        "exit_code": EXIT_INVALID,
                        "error": "the ledger does not verify; refusing to checkpoint it",
                        "failures": _failure_payload(report),
                    },
                    lines=tuple(lines),
                )

        signer = (
            None
            if args.signing_key is None
            else _signer_from_pem(args.signing_key, args.password_env)
        )
        ledger = Ledger(
            backend,
            hash_algorithm=args.hash_algorithm,
            enable_signatures=signer is not None,
            signer=signer,
            anchor_sink=_anchor_sink(args.anchor),
        )
        try:
            checkpoint = ledger.checkpoint()
        finally:
            # Closing the ledger closes the backend it was handed.
            ledger.close()
            closed = True

        payload = _checkpoint_payload(checkpoint)
        payload.update({"command": "checkpoint", "ok": True, "exit_code": EXIT_OK})
        payload["anchored_to"] = args.anchor
        lines = _checkpoint_lines(checkpoint)
        lines.append(f"anchored     {args.anchor or '(not published)'}")
        return Result(code=EXIT_OK, payload=payload, lines=tuple(lines))
    finally:
        if not closed:
            backend.close()


def _cmd_anchor(args: argparse.Namespace) -> Result:
    from signledger.core.exceptions import ValidationError

    # Reads the store and writes only to the sink, whether it is publishing or showing.
    backend = _open_store(args, create=False)
    try:
        sink = _anchor_sink(args.anchor)
        ledger_id = _ledger_id_of(backend)
        if args.show:
            published = None if ledger_id is None else sink.fetch_latest(ledger_id)
            if published is None:
                # An unanchored ledger is not evidence of anything, so this is never a
                # zero exit.
                return Result(
                    code=EXIT_INVALID,
                    payload={
                        "command": "anchor",
                        "ok": False,
                        "exit_code": EXIT_INVALID,
                        "ledger_id": ledger_id,
                        "checkpoint": None,
                    },
                    lines=("no checkpoint has been anchored for this ledger",),
                )
            payload = _checkpoint_payload(published)
            payload.update({"command": "anchor", "ok": True, "exit_code": EXIT_OK})
            return Result(
                code=EXIT_OK, payload=payload, lines=tuple(_checkpoint_lines(published))
            )

        checkpoint = backend.get_latest_checkpoint()
        if checkpoint is None:
            raise ValidationError(
                "this store holds no checkpoint to publish; run `signledger checkpoint` first"
            )
        receipt = sink.publish(checkpoint)
        payload = _checkpoint_payload(checkpoint)
        payload.update(
            {
                "command": "anchor",
                "ok": True,
                "exit_code": EXIT_OK,
                "sink": receipt.sink,
                "location": receipt.location,
                "published_at": receipt.published_at.isoformat(),
            }
        )
        lines = _checkpoint_lines(checkpoint)
        lines.append(f"published    {receipt.sink} -> {receipt.location}")
        return Result(code=EXIT_OK, payload=payload, lines=tuple(lines))
    finally:
        backend.close()


def _public_key_pems(paths: Optional[List[str]]) -> List[str]:
    """The PEM text of every key named by ``--key``, verbatim.

    Refuses an empty key directory for the same reason ``_key_resolver`` does, and it has
    to be the same answer: the identical typo used to be an operational error on ``verify``
    and a silent success on ``export``, which archived a signed ledger into a bundle
    holding no keys and reported it as done. The operator found out whenever someone next
    tried to verify the archive.
    """
    pems: List[str] = []
    for path in paths or []:
        found = _pem_files(path)
        if not found:
            raise ConfigurationError(f"no .pem files were found in the key directory {path}")
        for pem_path in found:
            with open(pem_path, encoding="utf-8") as handle:
                pems.append(handle.read())
    return pems


def _cmd_export(args: argparse.Namespace) -> Result:
    from signledger.core.exceptions import ValidationError
    from signledger.crypto.merkle import MerkleTree

    backend = _open_store(args, create=False)
    try:
        entries = list(backend.iter_entries())
        if not entries:
            raise ValidationError(
                "cannot export an empty ledger; a bundle of nothing is not evidence"
            )
        if not args.force:
            report, _ = _verify(backend, algorithm=args.hash_algorithm)
            if not report.ok:
                # A bundle is the artefact handed to someone who cannot see the store, so
                # exporting a chain that does not verify and exiting 0 tells an operator
                # scripting an archive step that the archive is sound. `checkpoint` has
                # always refused this; `export` printed a merkle root and exited 0, and
                # the only thing that caught it was whoever verified the bundle later.
                lines = ["refusing to export a ledger that does not verify"]
                lines.extend(_report_lines(report, ledger_id=entries[0].ledger_id))
                lines.append("pass --force to export it anyway")
                return Result(
                    code=EXIT_INVALID,
                    payload={
                        "command": "export",
                        "ok": False,
                        "exit_code": EXIT_INVALID,
                        "error": "the ledger does not verify; refusing to export it",
                        "failures": _failure_payload(report),
                    },
                    lines=tuple(lines),
                )
        checkpoint = backend.get_latest_checkpoint()
        bundle = {
            "format": BUNDLE_FORMAT,
            "ledger_id": entries[0].ledger_id,
            "hash_algorithm": args.hash_algorithm,
            "genesis_hash": _genesis_of(backend, args.hash_algorithm),
            "entry_count": len(entries),
            "merkle_root": MerkleTree(
                (bytes.fromhex(entry.hash) for entry in entries),
                algorithm=args.hash_algorithm,
            ).root,
            "exported_at": _utcnow().isoformat(),
            # ``mode="json"`` renders the timestamp as an offset-bearing ISO string, which
            # ``Entry.from_dict`` reads back to the same instant - so a re-hash on the far
            # side is byte-exact.
            "entries": [entry.model_dump(mode="json") for entry in entries],
            "checkpoint": None if checkpoint is None else _checkpoint_payload(checkpoint),
            "public_keys": _public_key_pems(args.key),
        }
        with open(args.bundle, "w", encoding="utf-8") as handle:
            json.dump(bundle, handle, indent=2, sort_keys=True)
            handle.write("\n")
        return Result(
            code=EXIT_OK,
            payload={
                "command": "export",
                "ok": True,
                "exit_code": EXIT_OK,
                "bundle": args.bundle,
                "ledger_id": bundle["ledger_id"],
                "entry_count": bundle["entry_count"],
                "merkle_root": bundle["merkle_root"],
                "checkpoint_included": bundle["checkpoint"] is not None,
                "public_keys_included": len(bundle["public_keys"]),
            },
            lines=(
                f"ledger       {bundle['ledger_id']}",
                f"entries      {bundle['entry_count']}",
                f"merkle root  {bundle['merkle_root']}",
                f"checkpoint   {'included' if bundle['checkpoint'] else 'none stored'}",
                f"public keys  {len(bundle['public_keys'])}",
                f"written      {args.bundle}",
            ),
        )
    finally:
        backend.close()


class _BundleView:
    """The three reads verification makes, served from a list held in memory.

    This is what makes ``verify-bundle`` genuinely offline: no driver is imported, no
    connection is opened, and no file other than the bundle is touched. A third party
    checking the evidence should not have to install a database to do it.
    """

    def __init__(self, entries: List[Any]) -> None:
        self._entries = entries

    def iter_entries(self, *, start_seq: Optional[int] = None) -> Any:
        if start_seq is None:
            return iter(list(self._entries))
        return iter([e for e in self._entries if e.sequence >= start_seq])

    def get_by_sequence(self, sequence: int) -> Any:
        return next((e for e in self._entries if e.sequence == sequence), None)

    def count(self) -> int:
        return len(self._entries)


def _decode_bundle(raw: str) -> Dict[str, Any]:
    """Parse and structurally check a bundle. Messages name fields, never values."""
    try:
        decoded = json.loads(raw)
    except ValueError as err:
        raise ConfigurationError("the bundle is not valid JSON") from err
    if not isinstance(decoded, dict):
        raise ConfigurationError(
            f"the bundle is a {type(decoded).__name__}, not a JSON object"
        )
    if decoded.get("format") != BUNDLE_FORMAT:
        raise ConfigurationError(
            f"unsupported bundle format {decoded.get('format')!r}; expected {BUNDLE_FORMAT!r}"
        )
    missing = [field for field in _BUNDLE_FIELDS if field not in decoded]
    if missing:
        raise ConfigurationError("the bundle is missing: " + ", ".join(sorted(missing)))
    if not isinstance(decoded["entries"], list):
        raise ConfigurationError(
            f"the bundle's entries must be a list, got {type(decoded['entries']).__name__}"
        )
    return decoded


def _bundle_entries(bundle: Dict[str, Any]) -> List[Any]:
    """Rebuild every entry. A record that cannot be read is an error, never a skip."""
    from pydantic import ValidationError as PydanticValidationError

    from signledger.core.entry import Entry
    from signledger.core.exceptions import SignLedgerError

    entries = []
    for index, record in enumerate(bundle["entries"]):
        try:
            entries.append(Entry.from_dict(record))
        except (SignLedgerError, PydanticValidationError, TypeError, ValueError) as err:
            # The type name only: a rejected record is user data (contract section 10).
            raise ConfigurationError(
                f"bundle entry {index} could not be read: {type(err).__name__}"
            ) from err
    return entries


def _bundle_checkpoint(bundle: Dict[str, Any]) -> Any:
    from signledger.anchor.checkpoint import Checkpoint

    record = bundle["checkpoint"]
    if record is None:
        return None
    try:
        return Checkpoint(
            ledger_id=record["ledger_id"],
            sequence=record["sequence"],
            head_hash=record["head_hash"],
            entry_count=record["entry_count"],
            merkle_root=record["merkle_root"],
            timestamp=datetime.fromisoformat(record["timestamp"]),
            signature=record.get("signature"),
            key_id=record.get("key_id"),
            format_version=record.get("format_version", 2),
        )
    except (KeyError, TypeError, ValueError) as err:
        raise ConfigurationError(
            f"the bundle's checkpoint could not be read: {type(err).__name__}"
        ) from err


def _bundle_failures(bundle: Dict[str, Any], entries: List[Any]) -> List[Dict[str, Any]]:
    """The checks that are about the bundle's own claims rather than the chain.

    Each one catches an edit to the *summary* a reader would otherwise take on trust: a
    swapped merkle root, a swapped genesis, a count that no longer matches the entries.
    """
    from signledger.crypto.hashing import genesis_hash
    from signledger.crypto.merkle import MerkleTree

    failures: List[Dict[str, Any]] = []
    if bundle["entry_count"] != len(entries):
        failures.append(
            _bundle_failure(
                "bundle_count_mismatch",
                expected=str(bundle["entry_count"]),
                actual=str(len(entries)),
                message=(
                    f"the bundle declares {bundle['entry_count']} entries but carries "
                    f"{len(entries)}"
                ),
            )
        )
    if entries:
        recomputed = MerkleTree(
            (bytes.fromhex(entry.hash) for entry in entries),
            algorithm=bundle["hash_algorithm"],
        ).root
        if recomputed != bundle["merkle_root"]:
            failures.append(
                _bundle_failure(
                    "bundle_merkle_mismatch",
                    expected=bundle["merkle_root"],
                    actual=recomputed,
                    message="the bundle's merkle root is not the root of its own entries",
                )
            )
    first = next((entry for entry in entries if entry.sequence == 0), None)
    if first is not None:
        anchor = genesis_hash(
            first.ledger_id,
            first.timestamp,
            first.format_version,
            algorithm=bundle["hash_algorithm"],
        )
        if anchor != bundle["genesis_hash"]:
            failures.append(
                _bundle_failure(
                    "bundle_genesis_mismatch",
                    expected=bundle["genesis_hash"],
                    actual=anchor,
                    message=(
                        "the bundle's genesis digest is not the one entry 0's own "
                        "identity and timestamp produce"
                    ),
                )
            )
    return failures


def _bundle_failure(kind: str, *, expected: str, actual: str, message: str) -> Dict[str, Any]:
    return {
        "kind": kind,
        "sequence": None,
        "entry_id": None,
        "expected": expected,
        "actual": actual,
        "message": message,
    }


def _cmd_verify_bundle(args: argparse.Namespace) -> Result:
    from signledger.core.verification import verify_chain
    from signledger.crypto.hashing import HashChain

    with open(args.bundle, encoding="utf-8") as handle:
        bundle = _decode_bundle(handle.read())

    entries = _bundle_entries(bundle)
    resolver = _key_resolver(args.key)
    embedded = False
    if resolver is None and args.trust_embedded_keys:
        resolver = _embedded_resolver(bundle)
        embedded = True

    report = verify_chain(
        _BundleView(entries),
        chain=HashChain(bundle["hash_algorithm"]),
        genesis=bundle["genesis_hash"],
        from_sequence=args.from_sequence,
        enable_signatures=resolver is not None,
        key_resolver=resolver,
        checkpoint=_bundle_checkpoint(bundle),
        anchored=False,
        capabilities=frozenset(),
    )
    extra = _bundle_failures(bundle, entries)
    result = _verification_result(
        "verify-bundle",
        report,
        ledger_id=bundle["ledger_id"],
        anchored=False,
        capabilities=frozenset(),
        extra_failures=extra,
    )
    if embedded and result.payload["ok"]:
        # Keys carried inside the thing they authenticate cannot raise assurance: whoever
        # rewrote the entries could re-sign them and swap the key in the same edit. The
        # signature check still runs, so a broken signature is still reported.
        #
        # Only ever applied to a bundle that *passed*. This is a ceiling on what a passing
        # check is worth, never a floor: a bundle that failed has established nothing
        # (contract section 1, NONE = "chain broken or unverified"), and stamping
        # CHAIN_ONLY on it would have printed "the signatures were checked ... proves
        # internal consistency" directly above "sequence 1 has a signature that does not
        # verify".
        result.payload["assurance"] = "chain_only"
        result.payload["assurance_note"] = EMBEDDED_KEY_NOTE
        result.lines = tuple(
            f"assurance    chain_only - {EMBEDDED_KEY_NOTE}"
            if line.startswith("assurance ")
            else line
            for line in result.lines
        )
    return result


def _embedded_resolver(bundle: Dict[str, Any]) -> Any:
    from signledger.crypto.keys import InMemoryKeyResolver

    pems = bundle["public_keys"]
    if not pems:
        raise ConfigurationError(
            "this bundle carries no public keys, so --trust-embedded-keys has nothing "
            "to check the signatures against"
        )
    resolver = InMemoryKeyResolver()
    for pem in pems:
        resolver.add_pem(pem)
    return resolver


def migration_signing_payload(record: Dict[str, Any]) -> bytes:
    """The canonical bytes a migration record's signature covers.

    Every field except ``signature``. ``key_id`` is inside it for the same reason it is
    inside an entry preimage: otherwise an attacker who can rewrite the record just
    re-points it at a key they hold.
    """
    from signledger.crypto.canonical import canonicalize

    return canonicalize(
        {field: record[field] for field in _MIGRATION_FIELDS if field != "signature"}
    )


def _cmd_migrate(args: argparse.Namespace) -> Result:
    from signledger.core.exceptions import ValidationError

    # The source is only ever read; the target is the store this command exists to fill.
    source = _open_store(args, create=False)
    target = None
    try:
        if source.count() == 0:
            raise ValidationError("cannot migrate an empty ledger; there is nothing to move")

        report, _ = _verify(source, algorithm=args.hash_algorithm)
        if not report.ok:
            # Copying a chain that does not verify into a fresh append-only store would
            # launder it: the new store is pristine and holds the tampered history.
            lines = ["refusing to migrate a ledger that does not verify"]
            lines.extend(_report_lines(report, ledger_id=_ledger_id_of(source)))
            return Result(
                code=EXIT_INVALID,
                payload={
                    "command": "migrate",
                    "ok": False,
                    "exit_code": EXIT_INVALID,
                    "error": "the source ledger does not verify; refusing to migrate it",
                    "failures": _failure_payload(report),
                },
                lines=tuple(lines),
            )

        # Unlocked *before* anything is written. The signing key is the operator input
        # most likely to be wrong - a mistyped passphrase, an unset variable, the wrong
        # PEM - and it is only needed at the very end, to sign the record. Loading it
        # there left a fully populated target behind a command that exited 2, and a
        # populated target can never be migrated into again ("must be empty"); on an
        # append-only store it cannot be emptied either, so the typo was permanent.
        signer = (
            None
            if args.signing_key is None
            else _signer_from_pem(args.signing_key, args.password_env)
        )

        target = open_backend(
            args.target,
            entries_table=args.target_entries_table,
            checkpoints_table=args.target_checkpoints_table,
            create=True,
        )
        if target.count() != 0:
            raise ConfigurationError(
                "the migration target must be empty; refusing to splice one chain into "
                "another"
            )

        entries = list(source.iter_entries())
        for entry in entries:
            # Copied verbatim, digests and all, so the migrated store verifies against the
            # same genesis the original did.
            target.append_atomic(lambda _tip, sealed=entry: sealed)
        for checkpoint in source.iter_checkpoints():
            target.save_checkpoint(checkpoint)

        moved, _ = _verify(target, algorithm=args.hash_algorithm)
        record = _migration_record(args, entries, signer=signer)
        if args.record is not None:
            with open(args.record, "w", encoding="utf-8") as handle:
                json.dump(record, handle, indent=2, sort_keys=True)
                handle.write("\n")

        payload = {
            "command": "migrate",
            "ok": moved.ok,
            "exit_code": EXIT_OK if moved.ok else EXIT_INVALID,
            "ledger_id": record["ledger_id"],
            "entries_migrated": len(entries),
            "source_scheme": record["source_scheme"],
            "target_scheme": record["target_scheme"],
            "source_head_hash": record["source_head_hash"],
            "record": args.record,
            "target_verified": moved.ok,
            "failures": _failure_payload(moved),
        }
        lines = [
            f"ledger       {record['ledger_id']}",
            f"entries      {len(entries)} migrated",
            f"from -> to   {record['source_scheme']} -> {record['target_scheme']}",
            f"head         {record['source_head_hash']}",
            f"record       {args.record or '(not written)'}",
        ]
        lines.extend(_verdict_lines(moved, label="target     "))
        return Result(
            code=EXIT_OK if moved.ok else EXIT_INVALID, payload=payload, lines=tuple(lines)
        )
    finally:
        source.close()
        if target is not None:
            target.close()


def _migration_record(
    args: argparse.Namespace, entries: List[Any], *, signer: Any
) -> Dict[str, Any]:
    """The record binding what was migrated, signed when ``signer`` is not ``None``.

    Only the *scheme* of each store is recorded. A DSN carries credentials, and this
    record is the one artefact of a migration meant to be handed to an auditor.

    The signer arrives already unlocked: a key that cannot be loaded must stop the
    migration before it copies anything, not after.
    """
    head = entries[-1]
    record = {
        "format": MIGRATION_FORMAT,
        "ledger_id": head.ledger_id,
        "source_scheme": urlsplit(args.backend).scheme,
        "target_scheme": urlsplit(args.target).scheme,
        "source_head_hash": head.hash,
        "source_entry_count": len(entries),
        "source_format_versions": sorted({entry.format_version for entry in entries}),
        "hash_algorithm": args.hash_algorithm,
        "migrated_at": _utcnow().isoformat(),
        "key_id": None,
        "signature": None,
    }
    if signer is not None:
        record["key_id"] = signer.key_id
        record["signature"] = signer.sign(migration_signing_payload(record))
    return record


def _cmd_stats(args: argparse.Namespace) -> Result:
    from signledger.observability import LedgerHealth

    backend = _open_store(args, create=False)
    try:
        report, anchored = _verify(
            backend,
            algorithm=args.hash_algorithm,
            resolver=_key_resolver(args.key),
            sink=_anchor_sink(args.anchor),
        )
        ledger_id = _ledger_id_of(backend)
        capabilities = backend.capabilities()
        head = backend.get_head()
        # Measured, never assumed. v1 hardcoded integrity_verified = True here.
        health = LedgerHealth.from_report(
            report, ledger_id=ledger_id or "", capabilities=capabilities
        )
        payload = dict(health.to_dict())
        payload.update(
            {
                "command": "stats",
                "exit_code": EXIT_OK if report.ok else EXIT_INVALID,
                "ledger_id": ledger_id,
                "entry_count": backend.count(),
                "size_bytes": backend.size_bytes(),
                "head_sequence": None if head is None else head.sequence,
                "head_hash": None if head is None else head.hash,
                "anchored": anchored,
                "failures": _failure_payload(report),
            }
        )
        level = report.assurance.value
        lines = [
            f"ledger       {'(none)' if ledger_id is None else ledger_id}",
            f"entries      {payload['entry_count']}",
            f"size         {payload['size_bytes']} bytes",
            f"head         {_head_text(head)}",
            f"capabilities {', '.join(sorted(capabilities)) or '(none)'}",
            f"assurance    {level} - {ASSURANCE_NOTES[level]}",
        ]
        lines.extend(_verdict_lines(report, label="integrity  "))
        return Result(
            code=EXIT_OK if report.ok else EXIT_INVALID, payload=payload, lines=tuple(lines)
        )
    finally:
        backend.close()


def _cmd_inspect(args: argparse.Namespace) -> Result:
    backend = _open_store(args, create=False)
    try:
        ledger_id = _ledger_id_of(backend)
        oldest = backend.get_oldest()
        head = backend.get_head()
        count = backend.count()
        capabilities = backend.capabilities()
        forks = find_forks(backend.iter_entries()) if args.forks else None

        lines = [
            f"ledger       {'(none)' if ledger_id is None else ledger_id}",
            f"entries      {count}",
            f"oldest       {_head_text(oldest)}",
            f"head         {_head_text(head)}",
            f"capabilities {', '.join(sorted(capabilities)) or '(none)'}",
        ]
        code = EXIT_OK
        if forks is not None:
            if count == 0:
                # Nothing to inspect is not the same answer as nothing wrong.
                code = EXIT_INVALID
                lines.append("forks        no entries; an empty store proves nothing")
            elif forks:
                code = EXIT_INVALID
                lines.append(f"forks        {len(forks)} anomaly(ies)")
                lines.extend(f"  [{item['kind']}] {item['message']}" for item in forks)
            else:
                lines.append("forks        none detected")
        return Result(
            code=code,
            payload={
                "command": "inspect",
                "ok": code == EXIT_OK,
                "exit_code": code,
                "ledger_id": ledger_id,
                "entry_count": count,
                "oldest_sequence": None if oldest is None else oldest.sequence,
                "head_sequence": None if head is None else head.sequence,
                "head_hash": None if head is None else head.hash,
                "capabilities": sorted(capabilities),
                "forks": forks,
            },
            lines=tuple(lines),
        )
    finally:
        backend.close()


def _head_text(entry: Any) -> str:
    """``sequence N (abcdef12…)``, or ``(none)`` for an empty store."""
    if entry is None:
        return "(none)"
    return f"sequence {entry.sequence} ({entry.hash[:12]}...)"


def _verdict_lines(report: Any, *, label: str) -> List[str]:
    if report.ok:
        return [f"{label}  OK"]
    lines = [f"{label}  FAILED - {len(report.failures)} failure(s)"]
    lines.extend(f"  [{f.kind.value}] {f.message}" for f in report.failures)
    return lines


# --------------------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """The full command line. Every subcommand names the store it works on."""
    parser = argparse.ArgumentParser(
        prog="signledger",
        description="Inspect, verify and anchor a tamper-evident SignLedger log.",
    )
    parser.set_defaults(command=None)
    parser.add_argument(
        "--version",
        action="version",
        version=f"signledger {__version__}",
        help="print the version and exit",
    )

    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument(
        "--json", action="store_true", help="emit machine-readable JSON on stdout"
    )

    store = argparse.ArgumentParser(add_help=False, parents=[shared])
    store.add_argument(
        "-b",
        "--backend",
        required=True,
        metavar="URI",
        help=(
            "memory://, sqlite:<path>, postgresql://… or mongodb://…/<database> "
            "(the mongodb database is the URI's path)"
        ),
    )
    store.add_argument(
        "--entries-table",
        "--entries-collection",
        dest="entries_table",
        metavar="NAME",
        help="the table the entries are in - the collection, on MongoDB "
        "(default: ledger_entries)",
    )
    store.add_argument(
        "--checkpoints-table",
        "--checkpoints-collection",
        dest="checkpoints_table",
        metavar="NAME",
        help="the table the checkpoints are in (default: ledger_checkpoints)",
    )
    store.add_argument(
        "--hash-algorithm",
        default="sha256",
        metavar="ALG",
        help="the algorithm this ledger was written with (default: sha256)",
    )

    subcommands = parser.add_subparsers(dest="command", metavar="COMMAND")

    verify = subcommands.add_parser(
        "verify", parents=[store], help="verify a ledger's chain, signatures and anchor"
    )
    verify.add_argument(
        "--from-sequence", type=int, metavar="N", help="verify from this sequence onwards"
    )
    verify.add_argument(
        "--key",
        action="append",
        metavar="PATH",
        help="a public key PEM, or a directory of them; enables signature checking",
    )
    verify.add_argument(
        "--anchor", metavar="PATH", help="an anchor file to compare the head against"
    )
    verify.set_defaults(handler=_cmd_verify)

    stats = subcommands.add_parser(
        "stats", parents=[store], help="size, capabilities and measured integrity"
    )
    stats.add_argument("--key", action="append", metavar="PATH", help="public key PEM(s)")
    stats.add_argument("--anchor", metavar="PATH", help="an anchor file to compare against")
    stats.set_defaults(handler=_cmd_stats)

    inspect = subcommands.add_parser(
        "inspect", parents=[store], help="describe a store, optionally hunting for forks"
    )
    inspect.add_argument(
        "--forks",
        action="store_true",
        help="look for two chains sharing one store (duplicate, branched or orphaned rows)",
    )
    inspect.set_defaults(handler=_cmd_inspect)

    checkpoint = subcommands.add_parser(
        "checkpoint",
        parents=[store],
        help="record, sign and publish the current head",
    )
    checkpoint.add_argument(
        "--signing-key", metavar="PATH", help="a private key PEM to sign the checkpoint with"
    )
    checkpoint.add_argument(
        "--password-env",
        metavar="VAR",
        help="environment variable holding the signing key's password",
    )
    checkpoint.add_argument(
        "--anchor", metavar="PATH", help="also publish the checkpoint to this anchor file"
    )
    checkpoint.add_argument(
        "--force",
        action="store_true",
        help="checkpoint even if the ledger does not verify (refused by default)",
    )
    checkpoint.set_defaults(handler=_cmd_checkpoint)

    anchor = subcommands.add_parser(
        "anchor", parents=[store], help="publish or read a checkpoint outside this store"
    )
    anchor.add_argument("--anchor", required=True, metavar="PATH", help="the anchor file")
    anchor.add_argument(
        "--show",
        action="store_true",
        help="read the newest anchored checkpoint instead of publishing one",
    )
    anchor.set_defaults(handler=_cmd_anchor)

    export = subcommands.add_parser(
        "export", parents=[store], help="write a self-contained, offline-verifiable bundle"
    )
    export.add_argument(
        "-o",
        "--bundle",
        required=True,
        metavar="PATH",
        help="where to write the bundle",
    )
    export.add_argument(
        "--key",
        action="append",
        metavar="PATH",
        help="public key PEM(s) to embed alongside the entries",
    )
    export.add_argument(
        "--force",
        action="store_true",
        help="export even if the ledger does not verify (refused by default)",
    )
    export.set_defaults(handler=_cmd_export)

    verify_bundle = subcommands.add_parser(
        "verify-bundle",
        parents=[shared],
        help="verify an exported bundle with no database access at all",
    )
    verify_bundle.add_argument("bundle", metavar="PATH", help="the bundle to verify")
    verify_bundle.add_argument(
        "--from-sequence", type=int, metavar="N", help="verify from this sequence onwards"
    )
    verify_bundle.add_argument(
        "--key",
        action="append",
        metavar="PATH",
        help="a public key PEM obtained separately from the bundle",
    )
    verify_bundle.add_argument(
        "--trust-embedded-keys",
        action="store_true",
        help="check signatures against the bundle's own keys (internal consistency only)",
    )
    verify_bundle.set_defaults(handler=_cmd_verify_bundle)

    migrate = subcommands.add_parser(
        "migrate", parents=[store], help="copy a verified ledger into another store"
    )
    migrate.add_argument(
        "--target", required=True, metavar="URI", help="the store to migrate into"
    )
    migrate.add_argument(
        "--target-entries-table",
        "--target-entries-collection",
        dest="target_entries_table",
        metavar="NAME",
        help="the entries table (collection) to create in the target",
    )
    migrate.add_argument(
        "--target-checkpoints-table",
        "--target-checkpoints-collection",
        dest="target_checkpoints_table",
        metavar="NAME",
        help="the checkpoints table (collection) to create in the target",
    )
    migrate.add_argument(
        "--record", metavar="PATH", help="write a migration record to this file"
    )
    migrate.add_argument(
        "--signing-key", metavar="PATH", help="a private key PEM to sign the record with"
    )
    migrate.add_argument(
        "--password-env",
        metavar="VAR",
        help="environment variable holding the signing key's password",
    )
    migrate.set_defaults(handler=_cmd_migrate)

    return parser


def _error_result(command: Optional[str], err: BaseException) -> Result:
    message = f"{type(err).__name__}: {err}"
    return Result(
        code=EXIT_ERROR,
        payload={
            "command": command,
            "ok": False,
            "exit_code": EXIT_ERROR,
            "error": message,
        },
        lines=(f"error: {message}",),
    )


def _unreadable_result(command: Optional[str], err: BaseException) -> Result:
    """A fault from below the ledger's own vocabulary, reported without its message.

    Only the exception *type* is quoted. A driver or a validator routinely puts the
    offending value in its message, and the offending value here is a row of the ledger -
    contract section 10 - which is why this cannot go through :func:`_error_result`.

    Before this existed, a store holding a column of the wrong type escaped ``main`` as a
    raw ``TypeError`` traceback and a process exit of 1: the code that means "the ledger
    did not verify", for a store nothing could read.
    """
    message = (
        f"the store could not be read: {type(err).__name__}. A stored row holds a value "
        f"this build cannot interpret - a column of the wrong type is the usual cause. "
        f"The underlying message is withheld: it can quote the row itself."
    )
    return Result(
        code=EXIT_ERROR,
        payload={
            "command": command,
            "ok": False,
            "exit_code": EXIT_ERROR,
            "error": message,
        },
        lines=(f"error: {message}",),
    )


def _render(result: Result, *, json_mode: bool) -> None:
    if json_mode:
        # Everything, success or failure, on stdout: a pipe reading JSON should not have
        # to merge two streams to find out what happened.
        print(json.dumps(result.payload, indent=2, sort_keys=True))
        return
    stream = sys.stderr if result.code == EXIT_ERROR else sys.stdout
    for line in result.lines:
        print(line, file=stream)


def main(argv: Optional[List[str]] = None) -> int:
    """Run one command. Returns the process exit code; never raises for a ledger fault."""
    from signledger.core.exceptions import SignLedgerError

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return EXIT_ERROR

    json_mode = bool(args.json)
    result: Result
    try:
        result = args.handler(args)
    except (SignLedgerError, OSError, ImportError) as err:
        # An operational failure - a bad URI, an unreadable file, a store that will not
        # open, an optional extra that is not installed. Distinct from exit 1, which
        # means the ledger itself did not verify.
        result = _error_result(args.command, err)
    except Exception as err:
        # Anything else is a store this build cannot read. It is still an operational
        # failure and it is still exit 2 - what it must never be is a traceback.
        result = _unreadable_result(args.command, err)
    _render(result, json_mode=json_mode)
    return result.code
