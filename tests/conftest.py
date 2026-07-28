"""Shared fixtures for every SignLedger test module.

The centrepiece is the parametrized :func:`backend` fixture. It yields a live instance of
**every** storage backend in turn, so a test written once is proof about all four. That is
the whole point of contract section 4: the backends must be interchangeable, and the only
way to know they are is to run one set of assertions against all of them.

``postgres`` and ``mongo`` are marked (see the ``postgres`` / ``mongo`` markers declared in
``pyproject.toml``) and **skip** - never fail - when the driver, the module or a live
server is missing. Nothing here mocks a database: a mocked backend proves nothing about
the SQL or BSON a real server actually executes.
"""

import os
import socket
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterator, List, Optional, Tuple
from urllib.parse import urlsplit

import pytest

from signledger.backends.base import build_next_entry
from signledger.core.entry import Entry
from signledger.crypto.hashing import HashChain, genesis_hash

# --------------------------------------------------------------------------------------
# ledger identity
# --------------------------------------------------------------------------------------

#: Fixed creation instant, so a genesis hash is reproducible across a test session.
LEDGER_CREATED_AT = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

#: Fixed base instant for generated entry timestamps.
BASE_TIMESTAMP = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def ledger_id() -> str:
    """A unique ledger id per test, so nothing leaks between them."""
    return str(uuid.uuid4())


@pytest.fixture
def ledger_created_at() -> datetime:
    return LEDGER_CREATED_AT


@pytest.fixture
def chain() -> HashChain:
    return HashChain("sha256")


@pytest.fixture
def genesis(ledger_id: str, ledger_created_at: datetime) -> str:
    """The digest entry 0 of this test's ledger must link back to."""
    return genesis_hash(ledger_id, ledger_created_at)


# --------------------------------------------------------------------------------------
# backends
# --------------------------------------------------------------------------------------


#: Override either of these to point the suite at your own server.
POSTGRES_DSN_ENV = "SIGNLEDGER_TEST_POSTGRES_DSN"
MONGO_URI_ENV = "SIGNLEDGER_TEST_MONGO_URI"


def _postgres_dsn() -> str:
    """Default matches a CI service container and a ``initdb -U postgres`` cluster.

    Without an explicit user libpq falls back to the OS account, which does not exist as
    a role in a freshly initialised cluster - a confusing skip on a server that is up.
    """
    return os.environ.get(
        POSTGRES_DSN_ENV, "postgresql://postgres@localhost:5432/signledger_test"
    )


def _mongo_uri() -> str:
    return os.environ.get(MONGO_URI_ENV, "mongodb://localhost:27017")


#: One shared database, per-test *collections*. A database per test leaves hundreds of
#: `signledger_test_*` databases on the server after a full run and slows it to a crawl.
MONGO_TEST_DATABASE = "signledger_test"


def _endpoint(url: str, default_port: int) -> Tuple[str, int]:
    """Host and port of a DSN or URI, for a fast pre-flight check."""
    parts = urlsplit(url)
    return (parts.hostname or "localhost", parts.port or default_port)


def _skip_unless_reachable(host: str, port: int, label: str) -> None:
    """Skip unless a TCP listener answers.

    A developer without the server running gets a skip, not a red suite - but a *fast*
    one: the driver's own connect can sit in a multi-second retry loop before giving up.
    """
    try:
        with socket.create_connection((host, port), timeout=1.0):
            return
    except OSError as err:
        pytest.skip(f"no live {label} server at {host}:{port} ({type(err).__name__})")


def _make_memory_backend(tmp_path: Any, table: str) -> Any:
    from signledger.backends.memory import MemoryBackend

    return MemoryBackend()


def _make_sqlite_backend(tmp_path: Any, table: str) -> Any:
    from signledger.backends.sqlite import SQLiteBackend

    return SQLiteBackend(str(tmp_path / "conformance.db"))


def _make_postgres_backend(tmp_path: Any, table: str) -> Any:
    pytest.importorskip("psycopg", reason="psycopg (v3) is not installed")
    import psycopg

    from signledger.backends.postgresql import PostgresBackend

    dsn = _postgres_dsn()
    host, port = _endpoint(dsn, 5432)
    _skip_unless_reachable(host, port, "PostgreSQL")
    try:
        psycopg.connect(dsn, connect_timeout=2).close()
    except Exception as err:  # pragma: no cover - depends on the developer's machine
        pytest.skip(
            f"cannot connect to PostgreSQL at {host}:{port} ({type(err).__name__}); "
            f"set {POSTGRES_DSN_ENV} to override"
        )
    return PostgresBackend(
        dsn, entries_table=f"{table}_entries", checkpoints_table=f"{table}_checkpoints"
    )


def _make_mongo_backend(tmp_path: Any, table: str) -> Any:
    pytest.importorskip("pymongo", reason="pymongo is not installed")
    import pymongo

    from signledger.backends.mongodb import MongoBackend

    uri = _mongo_uri()
    host, port = _endpoint(uri, 27017)
    _skip_unless_reachable(host, port, "MongoDB")
    try:
        # The client MUST be closed. pyproject sets filterwarnings=["error"], so the
        # ResourceWarning MongoClient.__del__ emits for a live client becomes a hard error
        # on the *next* test to run - reported as an unrelated failure in an unrelated file.
        with pymongo.MongoClient(uri, serverSelectionTimeoutMS=1500) as client:
            client.admin.command("ping")
    except Exception as err:  # pragma: no cover - depends on the developer's machine
        pytest.skip(
            f"cannot connect to MongoDB at {host}:{port} ({type(err).__name__}); "
            f"set {MONGO_URI_ENV} to override"
        )
    return MongoBackend(
        uri,
        database=MONGO_TEST_DATABASE,
        entries_collection=f"{table}_entries",
        checkpoints_collection=f"{table}_checkpoints",
    )


def _drop_memory(table: str) -> None:
    """Nothing to drop: the heap goes away with the instance."""


def _drop_sqlite(table: str) -> None:
    """Nothing to drop: the file lives under pytest's per-test ``tmp_path``."""


def _drop_postgres(table: str) -> None:
    """Drop the two tables this parametrization created.

    Without it a full run leaves ~130 ``sl_*`` tables in the test database, and the next
    run's catalog queries get slower with every run that came before it.
    """
    import psycopg

    with psycopg.connect(_postgres_dsn(), connect_timeout=5) as connection:
        with connection.cursor() as cursor:
            for name in (f"{table}_entries", f"{table}_checkpoints", f"{table}_entries_state"):
                cursor.execute(f'DROP TABLE IF EXISTS "{name}" CASCADE')  # noqa: S608
        connection.commit()


def _drop_mongo(table: str) -> None:
    """Drop the three collections this parametrization created."""
    import pymongo

    with pymongo.MongoClient(_mongo_uri(), serverSelectionTimeoutMS=1500) as client:
        database = client[MONGO_TEST_DATABASE]
        for name in database.list_collection_names():
            if name.startswith(table):
                database.drop_collection(name)


_BACKEND_FACTORIES = {
    "memory": _make_memory_backend,
    "sqlite": _make_sqlite_backend,
    "postgres": _make_postgres_backend,
    "mongo": _make_mongo_backend,
}

_BACKEND_CLEANUP = {
    "memory": _drop_memory,
    "sqlite": _drop_sqlite,
    "postgres": _drop_postgres,
    "mongo": _drop_mongo,
}


@pytest.fixture(
    params=[
        pytest.param("memory", id="memory"),
        pytest.param("sqlite", id="sqlite"),
        pytest.param("postgres", id="postgres", marks=pytest.mark.postgres),
        pytest.param("mongo", id="mongo", marks=pytest.mark.mongo),
    ]
)
def backend(request: Any, tmp_path: Any) -> Iterator[Any]:
    """A live, empty backend of every supported kind, one per parametrization."""
    table = "sl_" + uuid.uuid4().hex[:12]
    instance = _BACKEND_FACTORIES[request.param](tmp_path, table)
    try:
        yield instance
    finally:
        instance.close()
        _BACKEND_CLEANUP[request.param](table)


@pytest.fixture
def backend_name(request: Any) -> str:
    """The parametrization id of the active :func:`backend`, for capability assertions."""
    return str(request.node.callspec.params["backend"])


@pytest.fixture
def tmp_ledger(tmp_path: Any) -> Any:
    """A throwaway on-disk SQLite ledger store, closed automatically."""
    from signledger.backends.sqlite import SQLiteBackend

    instance = SQLiteBackend(str(tmp_path / "ledger.db"))
    try:
        yield instance
    finally:
        instance.close()


# --------------------------------------------------------------------------------------
# entry construction
# --------------------------------------------------------------------------------------


@pytest.fixture
def append(ledger_id: str, ledger_created_at: datetime, chain: HashChain) -> Any:
    """Factory: append one correctly linked entry to ``backend`` and return it.

    Sequence, ``previous_hash`` and ``hash`` are derived from the tip the backend hands
    the build callback, so the produced chain is always valid.
    """

    def _append(target: Any, data: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Entry:
        payload = {"event": "test"} if data is None else data
        return target.append_atomic(
            lambda tip: build_next_entry(
                tip,
                ledger_id=ledger_id,
                created_at=ledger_created_at,
                data=payload,
                chain=chain,
                **kwargs,
            )
        )

    return _append


@pytest.fixture
def sample_entries(append: Any) -> Any:
    """Factory: append ``count`` sequential entries and return them in sequence order."""

    def _sample(target: Any, count: int = 5, **kwargs: Any) -> List[Entry]:
        return [
            append(
                target,
                {"event": "sample", "index": index},
                timestamp=BASE_TIMESTAMP + timedelta(seconds=index),
                **kwargs,
            )
            for index in range(count)
        ]

    return _sample


# --------------------------------------------------------------------------------------
# signing material
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="session")
def signing_keypair() -> Any:
    """A real Ed25519 keypair plus its ``key_id`` fingerprint.

    Built straight from ``cryptography`` rather than from ``signledger.crypto.signatures``
    so that backend tests do not depend on the signature wave landing first.
    """
    import base64
    import hashlib
    from types import SimpleNamespace

    serialization = pytest.importorskip(
        "cryptography.hazmat.primitives.serialization", reason="cryptography is not installed"
    )
    ed25519 = pytest.importorskip("cryptography.hazmat.primitives.asymmetric.ed25519")

    private_key = ed25519.Ed25519PrivateKey.generate()
    public_key = private_key.public_key()
    public_der = public_key.public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    public_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode("ascii")

    return SimpleNamespace(
        private_key=private_key,
        public_key=public_key,
        public_pem=public_pem,
        key_id=hashlib.sha256(public_der).hexdigest(),
        algorithm="ed25519",
        sign=lambda payload: base64.b64encode(private_key.sign(payload)).decode("ascii"),
    )
