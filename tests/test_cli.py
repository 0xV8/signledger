"""Tests for ``signledger.cli`` and the ``signledger`` console script."""

import base64
import contextlib
import json
import os
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List

import pytest
from cryptography.hazmat.primitives.asymmetric import ec, ed25519, rsa

from signledger.anchor.sinks import FileAnchorSink
from signledger.backends.base import entry_to_row
from signledger.backends.memory import MemoryBackend
from signledger.backends.sqlite import SQLiteBackend
from signledger.cli import (
    BUNDLE_FORMAT,
    EXIT_ERROR,
    EXIT_INVALID,
    EXIT_OK,
    MIGRATION_FORMAT,
    build_parser,
    find_forks,
    main,
    migration_signing_payload,
    open_backend,
)
from signledger.core.entry import Entry
from signledger.core.exceptions import ConfigurationError, SignLedgerError
from signledger.core.ledger import Ledger
from signledger.crypto.merkle import MerkleTree
from signledger.crypto.signatures import Ed25519Signer, get_private_key_pem, verify

REPO_ROOT = Path(__file__).resolve().parent.parent


def sqlite_uri(path: Any) -> str:
    """The URI naming an absolute on-disk SQLite file."""
    return "sqlite://" + str(path)


def run(argv: List[str], capsys: Any) -> Any:
    """Invoke ``main`` and return ``(exit_code, stdout, stderr)``."""
    code = main(argv)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


# ======================================================================================
# backend URIs
# ======================================================================================


def test_a_memory_uri_opens_an_in_process_backend() -> None:
    backend = open_backend("memory://")
    try:
        assert isinstance(backend, MemoryBackend)
    finally:
        backend.close()


def test_a_sqlite_uri_opens_the_file_it_names(tmp_path: Any) -> None:
    path = tmp_path / "ledger.db"
    backend = open_backend(sqlite_uri(path))
    try:
        assert isinstance(backend, SQLiteBackend)
        assert backend.path == str(path)
    finally:
        backend.close()


def test_a_sqlite_uri_actually_creates_the_database_file(tmp_path: Any) -> None:
    path = tmp_path / "ledger.db"
    backend = open_backend(sqlite_uri(path))
    try:
        backend.count()
    finally:
        backend.close()

    assert path.exists()


def test_a_bare_sqlite_uri_opens_a_throwaway_in_memory_database() -> None:
    backend = open_backend("sqlite://")
    try:
        assert backend.path == ":memory:"
    finally:
        backend.close()


def test_an_explicit_sqlite_memory_uri_is_honoured() -> None:
    backend = open_backend("sqlite:///:memory:")
    try:
        assert backend.path == ":memory:"
    finally:
        backend.close()


def test_a_schemeless_sqlite_uri_names_a_relative_file() -> None:
    backend = open_backend("sqlite:ledger.db")
    try:
        assert backend.path == "ledger.db"
    finally:
        backend.close()


def test_a_sqlite_uri_with_a_host_is_refused_rather_than_silently_mis_parsed() -> None:
    # "sqlite://data/ledger.db" reads as host "data"; guessing what the operator meant
    # would open a different file from the one they named.
    with pytest.raises(ConfigurationError, match="host"):
        open_backend("sqlite://data/ledger.db")


def test_an_unknown_scheme_is_refused_by_name() -> None:
    with pytest.raises(ConfigurationError, match="redis"):
        open_backend("redis://localhost:6379/0")


def test_a_uri_without_a_scheme_is_refused() -> None:
    with pytest.raises(ConfigurationError, match="scheme"):
        open_backend("/var/lib/ledger.db")


def test_the_refusal_of_an_unknown_scheme_never_quotes_the_whole_uri() -> None:
    # Contract section 10: a URI can carry a password.
    with pytest.raises(ConfigurationError) as caught:
        open_backend("redis://admin:hunter2@localhost:6379/0")

    assert "hunter2" not in str(caught.value)


def test_a_postgres_uri_reaches_the_postgres_backend() -> None:
    pytest.importorskip("psycopg", reason="psycopg (v3) is not installed")

    # Port 1 refuses immediately, so this exercises the dispatch, not a live server.
    with pytest.raises(SignLedgerError):
        open_backend("postgresql://someone:hunter2@127.0.0.1:1/nothing")


def test_a_failure_to_open_a_postgres_uri_never_leaks_the_password() -> None:
    pytest.importorskip("psycopg", reason="psycopg (v3) is not installed")

    with pytest.raises(SignLedgerError) as caught:
        open_backend("postgresql://someone:hunter2@127.0.0.1:1/nothing")

    assert "hunter2" not in str(caught.value)


def test_the_postgres_alias_scheme_is_accepted_too() -> None:
    pytest.importorskip("psycopg", reason="psycopg (v3) is not installed")

    with pytest.raises(SignLedgerError):
        open_backend("postgres://someone@127.0.0.1:1/nothing")


@pytest.mark.mongo
def test_a_mongodb_uri_reaches_the_mongo_backend() -> None:
    pytest.importorskip("pymongo", reason="pymongo is not installed")

    with pytest.raises(SignLedgerError):
        open_backend("mongodb://127.0.0.1:1/?serverSelectionTimeoutMS=200")


# ======================================================================================
# ledgers to run commands against
#
# Every one is a real on-disk SQLite ledger written through the real Ledger API. The
# "tampering" helpers are an honest simulation of adversary B from contract section 1:
# someone with direct write access to the store, who therefore starts by removing the
# append-only triggers that would otherwise refuse them.
# ======================================================================================


def write_ledger(path: Any, count: int = 3, **kwargs: Any) -> str:
    """Create a SQLite ledger at ``path`` holding ``count`` entries; return its id."""
    ledger = Ledger(SQLiteBackend(str(path)), **kwargs)
    try:
        for index in range(count):
            ledger.append({"event": "created", "index": index})
        return ledger.ledger_id
    finally:
        ledger.close()


def unlock_store(path: Any) -> None:
    """Drop the append-only triggers, as an insider with SQL access would."""
    connection = sqlite3.connect(str(path))
    try:
        connection.execute("DROP TRIGGER IF EXISTS ledger_entries_no_update")
        connection.execute("DROP TRIGGER IF EXISTS ledger_entries_no_delete")
        connection.commit()
    finally:
        connection.close()


def rewrite_payload(path: Any, sequence: int, payload: str) -> None:
    """Edit one row's payload in place, leaving its stored digest untouched."""
    unlock_store(path)
    connection = sqlite3.connect(str(path))
    try:
        connection.execute(
            "UPDATE ledger_entries SET data = ? WHERE sequence = ?", (payload, sequence)
        )
        connection.commit()
    finally:
        connection.close()


def truncate_tail(path: Any, keep: int) -> None:
    """Delete every entry from ``keep`` onwards - the attack a bare chain cannot see."""
    unlock_store(path)
    connection = sqlite3.connect(str(path))
    try:
        connection.execute("DELETE FROM ledger_entries WHERE sequence >= ?", (keep,))
        connection.execute("DELETE FROM ledger_checkpoints")
        connection.commit()
    finally:
        connection.close()


# ======================================================================================
# verify
# ======================================================================================


def test_verifying_a_healthy_ledger_succeeds(tmp_path: Any, capsys: Any) -> None:
    path = tmp_path / "ledger.db"
    write_ledger(path, count=3)

    code, out, _ = run(["verify", "--backend", sqlite_uri(path)], capsys)

    assert code == EXIT_OK
    assert "VERIFIED" in out


def test_verifying_a_healthy_ledger_reports_how_many_entries_it_checked(
    tmp_path: Any, capsys: Any
) -> None:
    path = tmp_path / "ledger.db"
    write_ledger(path, count=4)

    _, out, _ = run(["verify", "--backend", sqlite_uri(path), "--json"], capsys)

    assert json.loads(out)["entries_verified"] == 4


def test_verifying_an_empty_ledger_is_not_a_success(tmp_path: Any, capsys: Any) -> None:
    # The v1 defect this whole rebuild starts from: a tick against nothing at all.
    path = tmp_path / "empty.db"
    SQLiteBackend(str(path)).close()

    code, out, _ = run(["verify", "--backend", sqlite_uri(path)], capsys)

    assert code == EXIT_INVALID
    assert "VERIFIED" not in out


def test_verifying_an_empty_ledger_says_why_it_failed(tmp_path: Any, capsys: Any) -> None:
    path = tmp_path / "empty.db"
    SQLiteBackend(str(path)).close()

    _, out, _ = run(["verify", "--backend", sqlite_uri(path), "--json"], capsys)

    assert [failure["kind"] for failure in json.loads(out)["failures"]] == ["count_mismatch"]


def test_verifying_a_rewritten_entry_fails(tmp_path: Any, capsys: Any) -> None:
    path = tmp_path / "ledger.db"
    write_ledger(path, count=3)
    rewrite_payload(path, 1, '{"event":"created","index":999}')

    code, _, _ = run(["verify", "--backend", sqlite_uri(path)], capsys)

    assert code == EXIT_INVALID


def test_a_rewritten_entry_is_named_by_sequence(tmp_path: Any, capsys: Any) -> None:
    path = tmp_path / "ledger.db"
    write_ledger(path, count=3)
    rewrite_payload(path, 1, '{"event":"created","index":999}')

    _, out, _ = run(["verify", "--backend", sqlite_uri(path), "--json"], capsys)

    failures = json.loads(out)["failures"]
    assert (failures[0]["kind"], failures[0]["sequence"]) == ("bad_hash", 1)


def test_the_human_report_of_a_rewritten_entry_names_the_damage(
    tmp_path: Any, capsys: Any
) -> None:
    path = tmp_path / "ledger.db"
    write_ledger(path, count=3)
    rewrite_payload(path, 1, '{"event":"created","index":999}')

    _, out, _ = run(["verify", "--backend", sqlite_uri(path)], capsys)

    assert "bad_hash" in out


def test_a_verify_report_never_quotes_the_entry_payload(tmp_path: Any, capsys: Any) -> None:
    path = tmp_path / "ledger.db"
    ledger = Ledger(SQLiteBackend(str(path)))
    ledger.append({"card_number": "4111111111111111"})
    ledger.close()

    _, out, _ = run(["verify", "--backend", sqlite_uri(path), "--json"], capsys)

    assert "4111111111111111" not in out


def test_verifying_from_a_sequence_checks_only_the_suffix(tmp_path: Any, capsys: Any) -> None:
    path = tmp_path / "ledger.db"
    write_ledger(path, count=5)

    _, out, _ = run(
        ["verify", "--backend", sqlite_uri(path), "--from-sequence", "3", "--json"], capsys
    )

    report = json.loads(out)
    assert (report["ok"], report["entries_verified"], report["first_sequence"]) == (True, 2, 3)


def test_verifying_a_ledger_that_cannot_be_opened_is_an_operational_error(
    capsys: Any,
) -> None:
    code, _, err = run(["verify", "--backend", "redis://localhost:6379/0"], capsys)

    assert code == EXIT_ERROR
    assert "redis" in err


def test_an_operational_error_is_reported_as_json_when_json_was_asked_for(
    capsys: Any,
) -> None:
    code, out, _ = run(["verify", "--backend", "redis://localhost:6379/0", "--json"], capsys)

    report = json.loads(out)
    assert (code, report["ok"], report["command"]) == (EXIT_ERROR, False, "verify")


def test_the_assurance_level_of_an_unsigned_ledger_is_chain_only(
    tmp_path: Any, capsys: Any
) -> None:
    path = tmp_path / "ledger.db"
    write_ledger(path, count=2)

    _, out, _ = run(["verify", "--backend", sqlite_uri(path), "--json"], capsys)

    assert json.loads(out)["assurance"] == "chain_only"


def test_chain_only_assurance_is_reported_with_its_caveat(tmp_path: Any, capsys: Any) -> None:
    # Rendering CHAIN_ONLY as an unqualified pass is how a deployment ends up believing a
    # bare hash chain stops an insider (contract section 1).
    path = tmp_path / "ledger.db"
    write_ledger(path, count=2)

    _, out, _ = run(["verify", "--backend", sqlite_uri(path)], capsys)

    assert "accidental corruption" in out


# ======================================================================================
# verify - signatures
# ======================================================================================


def write_signed_ledger(path: Any, key_path: Any, count: int = 3, **kwargs: Any) -> str:
    """A real Ed25519-signed ledger, with its public key written beside it."""
    signer = Ed25519Signer.generate()
    Path(key_path).write_text(signer.public_key_pem(), encoding="utf-8")
    ledger = Ledger(SQLiteBackend(str(path)), enable_signatures=True, signer=signer, **kwargs)
    try:
        for index in range(count):
            ledger.append({"event": "created", "index": index})
        return ledger.ledger_id
    finally:
        ledger.close()


def test_a_signed_ledger_verified_with_its_key_reaches_signed_assurance(
    tmp_path: Any, capsys: Any
) -> None:
    path = tmp_path / "ledger.db"
    key = tmp_path / "audit.pem"
    write_signed_ledger(path, key)

    code, out, _ = run(
        ["verify", "--backend", sqlite_uri(path), "--key", str(key), "--json"], capsys
    )

    assert (code, json.loads(out)["assurance"]) == (EXIT_OK, "signed")


def test_a_signed_ledger_verified_with_a_strangers_key_is_refused(
    tmp_path: Any, capsys: Any
) -> None:
    path = tmp_path / "ledger.db"
    write_signed_ledger(path, tmp_path / "audit.pem")
    stranger = tmp_path / "stranger.pem"
    stranger.write_text(Ed25519Signer.generate().public_key_pem(), encoding="utf-8")

    code, out, _ = run(
        ["verify", "--backend", sqlite_uri(path), "--key", str(stranger), "--json"], capsys
    )

    assert code == EXIT_INVALID
    assert {failure["kind"] for failure in json.loads(out)["failures"]} == {"unknown_key"}


def test_a_signed_ledger_verified_without_a_key_is_only_chain_checked(
    tmp_path: Any, capsys: Any
) -> None:
    path = tmp_path / "ledger.db"
    write_signed_ledger(path, tmp_path / "audit.pem")

    code, out, _ = run(["verify", "--backend", sqlite_uri(path), "--json"], capsys)

    assert (code, json.loads(out)["assurance"]) == (EXIT_OK, "chain_only")


def test_a_directory_of_keys_is_searched_for_pem_files(tmp_path: Any, capsys: Any) -> None:
    path = tmp_path / "ledger.db"
    keys = tmp_path / "keys"
    keys.mkdir()
    write_signed_ledger(path, keys / "audit.pem")
    (keys / "notes.txt").write_text("not a key", encoding="utf-8")

    code, out, _ = run(
        ["verify", "--backend", sqlite_uri(path), "--key", str(keys), "--json"], capsys
    )

    assert (code, json.loads(out)["assurance"]) == (EXIT_OK, "signed")


def test_a_key_directory_with_no_pem_files_is_an_operational_error(
    tmp_path: Any, capsys: Any
) -> None:
    # An empty resolver answers UNKNOWN_KEY for every entry, which looks exactly like an
    # attack. A typo must not be reported as tampering.
    path = tmp_path / "ledger.db"
    write_ledger(path, count=1)
    empty = tmp_path / "keys"
    empty.mkdir()

    code, _, err = run(["verify", "--backend", sqlite_uri(path), "--key", str(empty)], capsys)

    assert code == EXIT_ERROR
    assert "no .pem files" in err


def test_a_missing_key_file_is_an_operational_error(tmp_path: Any, capsys: Any) -> None:
    path = tmp_path / "ledger.db"
    write_ledger(path, count=1)

    code, _, err = run(
        ["verify", "--backend", sqlite_uri(path), "--key", str(tmp_path / "nope.pem")], capsys
    )

    assert code == EXIT_ERROR
    assert "FileNotFoundError" in err


# ======================================================================================
# verify - anchoring and truncation
# ======================================================================================


def test_a_truncated_tail_is_detected_against_an_anchored_checkpoint(
    tmp_path: Any, capsys: Any
) -> None:
    path = tmp_path / "ledger.db"
    anchor = tmp_path / "anchor.jsonl"
    ledger = Ledger(SQLiteBackend(str(path)), anchor_sink=FileAnchorSink(str(anchor)))
    for index in range(5):
        ledger.append({"event": "created", "index": index})
    ledger.checkpoint()
    ledger.close()
    truncate_tail(path, keep=3)

    code, out, _ = run(
        ["verify", "--backend", sqlite_uri(path), "--anchor", str(anchor), "--json"], capsys
    )

    assert code == EXIT_INVALID
    assert {failure["kind"] for failure in json.loads(out)["failures"]} == {
        "head_mismatch",
        "count_mismatch",
    }


def test_a_truncated_tail_without_an_anchor_is_invisible(tmp_path: Any, capsys: Any) -> None:
    # Documented limitation, pinned deliberately: what survives a truncation is a
    # genuinely self-consistent chain. Only a statement made earlier contradicts it.
    path = tmp_path / "ledger.db"
    write_ledger(path, count=5)
    truncate_tail(path, keep=3)

    code, _, _ = run(["verify", "--backend", sqlite_uri(path)], capsys)

    assert code == EXIT_OK


def test_an_anchored_signed_ledger_on_an_append_only_store_reaches_worm(
    tmp_path: Any, capsys: Any
) -> None:
    path = tmp_path / "ledger.db"
    key = tmp_path / "audit.pem"
    anchor = tmp_path / "anchor.jsonl"
    signer = Ed25519Signer.generate()
    key.write_text(signer.public_key_pem(), encoding="utf-8")
    ledger = Ledger(
        SQLiteBackend(str(path)),
        enable_signatures=True,
        signer=signer,
        anchor_sink=FileAnchorSink(str(anchor)),
    )
    ledger.append({"event": "created"})
    ledger.checkpoint()
    ledger.close()

    _, out, _ = run(
        [
            "verify",
            "--backend",
            sqlite_uri(path),
            "--key",
            str(key),
            "--anchor",
            str(anchor),
            "--json",
        ],
        capsys,
    )

    assert json.loads(out)["assurance"] == "worm"


def test_a_corrupted_anchor_file_is_an_operational_error(tmp_path: Any, capsys: Any) -> None:
    # Skipping a malformed anchor record would let an attacker delete an anchor simply by
    # corrupting it.
    path = tmp_path / "ledger.db"
    write_ledger(path, count=2)
    anchor = tmp_path / "anchor.jsonl"
    anchor.write_text("{not json at all}\n", encoding="utf-8")

    code, _, err = run(
        ["verify", "--backend", sqlite_uri(path), "--anchor", str(anchor)], capsys
    )

    assert code == EXIT_ERROR
    assert "AnchorError" in err


def test_an_anchor_file_that_does_not_exist_yet_falls_back_to_the_stored_checkpoint(
    tmp_path: Any, capsys: Any
) -> None:
    path = tmp_path / "ledger.db"
    write_ledger(path, count=2)

    code, out, _ = run(
        [
            "verify",
            "--backend",
            sqlite_uri(path),
            "--anchor",
            str(tmp_path / "missing.jsonl"),
            "--json",
        ],
        capsys,
    )

    assert (code, json.loads(out)["anchored"]) == (EXIT_OK, False)


# ======================================================================================
# stats
# ======================================================================================


def test_stats_of_a_healthy_ledger_succeeds(tmp_path: Any, capsys: Any) -> None:
    path = tmp_path / "ledger.db"
    write_ledger(path, count=3)

    code, _, _ = run(["stats", "--backend", sqlite_uri(path)], capsys)

    assert code == EXIT_OK


def test_stats_reports_the_entry_count(tmp_path: Any, capsys: Any) -> None:
    path = tmp_path / "ledger.db"
    write_ledger(path, count=4)

    _, out, _ = run(["stats", "--backend", sqlite_uri(path), "--json"], capsys)

    assert json.loads(out)["entry_count"] == 4


def test_stats_reports_the_size_the_store_occupies(tmp_path: Any, capsys: Any) -> None:
    path = tmp_path / "ledger.db"
    write_ledger(path, count=4)

    _, out, _ = run(["stats", "--backend", sqlite_uri(path), "--json"], capsys)

    assert json.loads(out)["size_bytes"] > 0


def test_stats_reports_what_the_store_actually_enforces(tmp_path: Any, capsys: Any) -> None:
    path = tmp_path / "ledger.db"
    write_ledger(path, count=1)

    _, out, _ = run(["stats", "--backend", sqlite_uri(path), "--json"], capsys)

    assert "append_only_enforced" in json.loads(out)["capabilities"]


def test_stats_reports_the_head(tmp_path: Any, capsys: Any) -> None:
    path = tmp_path / "ledger.db"
    write_ledger(path, count=3)

    _, out, _ = run(["stats", "--backend", sqlite_uri(path), "--json"], capsys)

    assert json.loads(out)["head_sequence"] == 2


def test_stats_of_an_empty_ledger_is_not_a_success(tmp_path: Any, capsys: Any) -> None:
    path = tmp_path / "empty.db"
    SQLiteBackend(str(path)).close()

    code, out, _ = run(["stats", "--backend", sqlite_uri(path), "--json"], capsys)

    assert (code, json.loads(out)["head_sequence"]) == (EXIT_INVALID, None)


def test_stats_of_a_tampered_ledger_is_not_a_success(tmp_path: Any, capsys: Any) -> None:
    # A health check that reports "fine" for a rewritten ledger is the v1 defect exactly.
    path = tmp_path / "ledger.db"
    write_ledger(path, count=3)
    rewrite_payload(path, 1, '{"event":"created","index":999}')

    code, _, _ = run(["stats", "--backend", sqlite_uri(path)], capsys)

    assert code == EXIT_INVALID


def test_the_human_stats_output_lists_the_capabilities(tmp_path: Any, capsys: Any) -> None:
    path = tmp_path / "ledger.db"
    write_ledger(path, count=1)

    _, out, _ = run(["stats", "--backend", sqlite_uri(path)], capsys)

    assert "durable_fsync" in out


# ======================================================================================
# inspect
# ======================================================================================


def insert_orphan(path: Any, sequence: int = 9) -> None:
    """Insert a well-formed row that links to nothing - a second chain in one store."""
    backend = SQLiteBackend(str(path))
    try:
        ledger_id = backend.get_oldest().ledger_id
    finally:
        backend.close()
    orphan = Entry(
        ledger_id=ledger_id,
        sequence=sequence,
        entry_id="11111111-1111-4111-8111-111111111111",
        timestamp=datetime(2024, 6, 1, tzinfo=timezone.utc),
        data={"event": "orphan"},
        previous_hash="ab" * 32,
        hash="cd" * 32,
    )
    row = entry_to_row(orphan)
    connection = sqlite3.connect(str(path))
    try:
        connection.execute(
            "INSERT INTO ledger_entries ({}) VALUES ({})".format(
                ",".join(row), ",".join("?" * len(row))
            ),
            tuple(row.values()),
        )
        connection.commit()
    finally:
        connection.close()


def test_inspecting_a_healthy_ledger_succeeds(tmp_path: Any, capsys: Any) -> None:
    path = tmp_path / "ledger.db"
    write_ledger(path, count=3)

    code, _, _ = run(["inspect", "--backend", sqlite_uri(path)], capsys)

    assert code == EXIT_OK


def test_inspect_reports_the_oldest_and_newest_sequences(tmp_path: Any, capsys: Any) -> None:
    path = tmp_path / "ledger.db"
    write_ledger(path, count=3)

    _, out, _ = run(["inspect", "--backend", sqlite_uri(path), "--json"], capsys)

    report = json.loads(out)
    assert (report["oldest_sequence"], report["head_sequence"]) == (0, 2)


def test_inspecting_an_empty_store_says_so_without_failing(tmp_path: Any, capsys: Any) -> None:
    path = tmp_path / "empty.db"
    SQLiteBackend(str(path)).close()

    code, out, _ = run(["inspect", "--backend", sqlite_uri(path)], capsys)

    assert code == EXIT_OK
    assert "0" in out


def test_inspect_does_not_look_for_forks_unless_asked(tmp_path: Any, capsys: Any) -> None:
    path = tmp_path / "ledger.db"
    write_ledger(path, count=2)

    _, out, _ = run(["inspect", "--backend", sqlite_uri(path), "--json"], capsys)

    assert json.loads(out)["forks"] is None


def test_looking_for_forks_in_a_healthy_ledger_finds_none(tmp_path: Any, capsys: Any) -> None:
    path = tmp_path / "ledger.db"
    write_ledger(path, count=4)

    code, out, _ = run(["inspect", "--backend", sqlite_uri(path), "--forks", "--json"], capsys)

    assert (code, json.loads(out)["forks"]) == (EXIT_OK, [])


def test_looking_for_forks_in_an_empty_store_is_not_a_success(
    tmp_path: Any, capsys: Any
) -> None:
    path = tmp_path / "empty.db"
    SQLiteBackend(str(path)).close()

    code, _, _ = run(["inspect", "--backend", sqlite_uri(path), "--forks"], capsys)

    assert code == EXIT_INVALID


def test_an_orphaned_row_is_reported_as_a_fork(tmp_path: Any, capsys: Any) -> None:
    path = tmp_path / "ledger.db"
    write_ledger(path, count=3)
    insert_orphan(path)

    code, out, _ = run(["inspect", "--backend", sqlite_uri(path), "--forks", "--json"], capsys)

    kinds = {anomaly["kind"] for anomaly in json.loads(out)["forks"]}
    assert code == EXIT_INVALID
    assert "orphan" in kinds


def test_an_orphaned_row_also_shows_up_as_a_second_head(tmp_path: Any, capsys: Any) -> None:
    path = tmp_path / "ledger.db"
    write_ledger(path, count=3)
    insert_orphan(path)

    _, out, _ = run(["inspect", "--backend", sqlite_uri(path), "--forks", "--json"], capsys)

    kinds = {anomaly["kind"] for anomaly in json.loads(out)["forks"]}
    assert "multiple_heads" in kinds


def test_the_human_fork_report_names_each_anomaly(tmp_path: Any, capsys: Any) -> None:
    path = tmp_path / "ledger.db"
    write_ledger(path, count=3)
    insert_orphan(path)

    _, out, _ = run(["inspect", "--backend", sqlite_uri(path), "--forks"], capsys)

    assert "orphan" in out


# ======================================================================================
# find_forks
# ======================================================================================


def chain_entry(sequence: int, previous_hash: str, digest: str, entry_id: str = "") -> Entry:
    """A hand-built entry, so a chain shape no backend would accept can still be tested."""
    return Entry(
        ledger_id="11111111-1111-4111-8111-111111111111",
        sequence=sequence,
        entry_id=entry_id or f"entry-{sequence}-{digest[:4]}",
        timestamp=datetime(2024, 6, 1, tzinfo=timezone.utc),
        data={"event": "test"},
        previous_hash=previous_hash,
        hash=digest,
    )


def test_no_entries_at_all_yields_no_anomalies() -> None:
    assert find_forks([]) == []


def test_a_straight_chain_yields_no_anomalies() -> None:
    entries = [
        chain_entry(0, "00" * 32, "aa" * 32),
        chain_entry(1, "aa" * 32, "bb" * 32),
        chain_entry(2, "bb" * 32, "cc" * 32),
    ]

    assert find_forks(entries) == []


def test_two_entries_following_the_same_parent_are_a_branch() -> None:
    entries = [
        chain_entry(0, "00" * 32, "aa" * 32),
        chain_entry(1, "aa" * 32, "bb" * 32),
        chain_entry(2, "aa" * 32, "cc" * 32),
    ]

    kinds = [anomaly["kind"] for anomaly in find_forks(entries)]

    assert "branch" in kinds


def test_a_branch_names_the_sequences_that_diverge() -> None:
    entries = [
        chain_entry(0, "00" * 32, "aa" * 32),
        chain_entry(1, "aa" * 32, "bb" * 32),
        chain_entry(2, "aa" * 32, "cc" * 32),
    ]

    branch = next(a for a in find_forks(entries) if a["kind"] == "branch")

    assert branch["sequences"] == [1, 2]


def test_the_same_sequence_twice_is_reported() -> None:
    entries = [
        chain_entry(0, "00" * 32, "aa" * 32),
        chain_entry(1, "aa" * 32, "bb" * 32),
        chain_entry(1, "bb" * 32, "cc" * 32),
    ]

    kinds = [anomaly["kind"] for anomaly in find_forks(entries)]

    assert "duplicate_sequence" in kinds


def test_the_same_entry_id_twice_is_reported() -> None:
    entries = [
        chain_entry(0, "00" * 32, "aa" * 32, entry_id="same"),
        chain_entry(1, "aa" * 32, "bb" * 32, entry_id="same"),
    ]

    kinds = [anomaly["kind"] for anomaly in find_forks(entries)]

    assert "duplicate_entry_id" in kinds


def test_the_same_digest_twice_is_reported() -> None:
    entries = [
        chain_entry(0, "00" * 32, "aa" * 32),
        chain_entry(1, "aa" * 32, "aa" * 32),
    ]

    kinds = [anomaly["kind"] for anomaly in find_forks(entries)]

    assert "duplicate_hash" in kinds


def test_an_entry_linking_to_nothing_is_an_orphan() -> None:
    entries = [
        chain_entry(0, "00" * 32, "aa" * 32),
        chain_entry(1, "aa" * 32, "bb" * 32),
        chain_entry(7, "ff" * 32, "cc" * 32),
    ]

    orphan = next(a for a in find_forks(entries) if a["kind"] == "orphan")

    assert orphan["sequences"] == [7]


def test_entry_zero_is_never_called_an_orphan() -> None:
    # Entry 0 links to the genesis digest, which is not the hash of any stored entry.
    entries = [chain_entry(0, "00" * 32, "aa" * 32)]

    assert [a["kind"] for a in find_forks(entries)] == []


def test_two_unfollowed_entries_are_two_heads() -> None:
    entries = [
        chain_entry(0, "00" * 32, "aa" * 32),
        chain_entry(1, "aa" * 32, "bb" * 32),
        chain_entry(2, "aa" * 32, "cc" * 32),
    ]

    heads = next(a for a in find_forks(entries) if a["kind"] == "multiple_heads")

    assert heads["sequences"] == [1, 2]


def test_every_anomaly_carries_a_message_an_operator_can_read() -> None:
    entries = [
        chain_entry(0, "00" * 32, "aa" * 32),
        chain_entry(1, "aa" * 32, "bb" * 32),
        chain_entry(2, "aa" * 32, "cc" * 32),
    ]

    assert all(anomaly["message"] for anomaly in find_forks(entries))


def test_anomalies_come_back_in_a_stable_order() -> None:
    entries = [
        chain_entry(0, "00" * 32, "aa" * 32),
        chain_entry(1, "aa" * 32, "bb" * 32),
        chain_entry(2, "aa" * 32, "cc" * 32),
    ]

    assert [a["kind"] for a in find_forks(entries)] == sorted(
        a["kind"] for a in find_forks(entries)
    )


# ======================================================================================
# checkpoint
# ======================================================================================


def test_checkpointing_a_healthy_ledger_succeeds(tmp_path: Any, capsys: Any) -> None:
    path = tmp_path / "ledger.db"
    write_ledger(path, count=5)

    code, _, _ = run(["checkpoint", "--backend", sqlite_uri(path)], capsys)

    assert code == EXIT_OK


def test_a_checkpoint_records_the_current_head(tmp_path: Any, capsys: Any) -> None:
    path = tmp_path / "ledger.db"
    write_ledger(path, count=5)

    _, out, _ = run(["checkpoint", "--backend", sqlite_uri(path), "--json"], capsys)

    report = json.loads(out)
    assert (report["sequence"], report["entry_count"]) == (4, 5)


def test_a_checkpoint_binds_the_merkle_root_of_every_entry(tmp_path: Any, capsys: Any) -> None:
    path = tmp_path / "ledger.db"
    write_ledger(path, count=5)

    _, out, _ = run(["checkpoint", "--backend", sqlite_uri(path), "--json"], capsys)

    backend = SQLiteBackend(str(path))
    try:
        entries = list(backend.iter_entries())
    finally:
        backend.close()
    expected = MerkleTree(bytes.fromhex(entry.hash) for entry in entries).root
    assert json.loads(out)["merkle_root"] == expected


def test_a_checkpoint_is_persisted_in_the_store(tmp_path: Any, capsys: Any) -> None:
    path = tmp_path / "ledger.db"
    write_ledger(path, count=3)

    run(["checkpoint", "--backend", sqlite_uri(path)], capsys)

    backend = SQLiteBackend(str(path))
    try:
        assert backend.get_latest_checkpoint() is not None
    finally:
        backend.close()


def test_a_checkpoint_is_published_to_the_anchor_when_one_is_named(
    tmp_path: Any, capsys: Any
) -> None:
    path = tmp_path / "ledger.db"
    anchor = tmp_path / "anchor.jsonl"
    ledger_id = write_ledger(path, count=3)

    run(["checkpoint", "--backend", sqlite_uri(path), "--anchor", str(anchor)], capsys)

    assert FileAnchorSink(str(anchor)).fetch_latest(ledger_id) is not None


def test_a_checkpoint_is_signed_when_a_signing_key_is_given(
    tmp_path: Any, capsys: Any
) -> None:
    path = tmp_path / "ledger.db"
    write_ledger(path, count=3)
    private = tmp_path / "signing.pem"
    private.write_text(
        get_private_key_pem(
            ed25519.Ed25519PrivateKey.generate(), allow_unencrypted=True
        ),
        encoding="utf-8",
    )

    _, out, _ = run(
        ["checkpoint", "--backend", sqlite_uri(path), "--signing-key", str(private), "--json"],
        capsys,
    )

    report = json.loads(out)
    assert report["signature"] is not None
    assert report["key_id"] is not None


def test_an_unsigned_checkpoint_says_so(tmp_path: Any, capsys: Any) -> None:
    path = tmp_path / "ledger.db"
    write_ledger(path, count=3)

    _, out, _ = run(["checkpoint", "--backend", sqlite_uri(path), "--json"], capsys)

    assert json.loads(out)["signature"] is None


def test_an_encrypted_signing_key_is_unlocked_from_the_environment(
    tmp_path: Any, capsys: Any, monkeypatch: Any
) -> None:
    # The password comes from the environment, never from argv: an argument is visible in
    # `ps` output and lands in shell history.
    path = tmp_path / "ledger.db"
    write_ledger(path, count=2)
    private = tmp_path / "signing.pem"
    private.write_text(
        get_private_key_pem(ed25519.Ed25519PrivateKey.generate(), password=b"hunter2"),
        encoding="utf-8",
    )
    monkeypatch.setenv("SL_KEY_PASSWORD", "hunter2")

    code, out, _ = run(
        [
            "checkpoint",
            "--backend",
            sqlite_uri(path),
            "--signing-key",
            str(private),
            "--password-env",
            "SL_KEY_PASSWORD",
            "--json",
        ],
        capsys,
    )

    assert (code, json.loads(out)["signature"] is None) == (EXIT_OK, False)


def test_an_encrypted_signing_key_without_its_password_is_an_operational_error(
    tmp_path: Any, capsys: Any
) -> None:
    path = tmp_path / "ledger.db"
    write_ledger(path, count=2)
    private = tmp_path / "signing.pem"
    private.write_text(
        get_private_key_pem(ed25519.Ed25519PrivateKey.generate(), password=b"hunter2"),
        encoding="utf-8",
    )

    code, _, _ = run(
        ["checkpoint", "--backend", sqlite_uri(path), "--signing-key", str(private)], capsys
    )

    assert code == EXIT_ERROR


def test_a_password_env_var_that_is_not_set_is_an_operational_error(
    tmp_path: Any, capsys: Any
) -> None:
    path = tmp_path / "ledger.db"
    write_ledger(path, count=2)
    private = tmp_path / "signing.pem"
    private.write_text(
        get_private_key_pem(ed25519.Ed25519PrivateKey.generate(), password=b"hunter2"),
        encoding="utf-8",
    )

    code, _, err = run(
        [
            "checkpoint",
            "--backend",
            sqlite_uri(path),
            "--signing-key",
            str(private),
            "--password-env",
            "SL_UNSET_PASSWORD_VAR",
        ],
        capsys,
    )

    assert code == EXIT_ERROR
    assert "SL_UNSET_PASSWORD_VAR" in err


def test_an_rsa_signing_key_is_recognised(tmp_path: Any, capsys: Any) -> None:
    path = tmp_path / "ledger.db"
    write_ledger(path, count=2)
    private = tmp_path / "signing.pem"
    private.write_text(
        get_private_key_pem(
            rsa.generate_private_key(public_exponent=65537, key_size=2048),
            allow_unencrypted=True,
        ),
        encoding="utf-8",
    )

    _, out, _ = run(
        ["checkpoint", "--backend", sqlite_uri(path), "--signing-key", str(private), "--json"],
        capsys,
    )

    assert json.loads(out)["signature"] is not None


def test_an_ecdsa_signing_key_is_recognised_on_its_own_curve(
    tmp_path: Any, capsys: Any
) -> None:
    path = tmp_path / "ledger.db"
    write_ledger(path, count=2)
    private = tmp_path / "signing.pem"
    private.write_text(
        get_private_key_pem(ec.generate_private_key(ec.SECP384R1()), allow_unencrypted=True),
        encoding="utf-8",
    )

    _, out, _ = run(
        ["checkpoint", "--backend", sqlite_uri(path), "--signing-key", str(private), "--json"],
        capsys,
    )

    assert json.loads(out)["signature"] is not None


def test_checkpointing_an_empty_ledger_is_an_operational_error(
    tmp_path: Any, capsys: Any
) -> None:
    path = tmp_path / "empty.db"
    SQLiteBackend(str(path)).close()

    code, _, err = run(["checkpoint", "--backend", sqlite_uri(path)], capsys)

    assert code == EXIT_ERROR
    assert "empty" in err


def test_checkpointing_a_tampered_ledger_is_refused(tmp_path: Any, capsys: Any) -> None:
    # Publishing a checkpoint over a rewritten chain would launder the tampering into the
    # one record an auditor is meant to be able to trust.
    path = tmp_path / "ledger.db"
    write_ledger(path, count=3)
    rewrite_payload(path, 1, '{"event":"created","index":999}')

    code, out, _ = run(["checkpoint", "--backend", sqlite_uri(path)], capsys)

    assert code == EXIT_INVALID
    assert "refus" in out


def test_a_refused_checkpoint_writes_nothing(tmp_path: Any, capsys: Any) -> None:
    path = tmp_path / "ledger.db"
    write_ledger(path, count=3)
    rewrite_payload(path, 1, '{"event":"created","index":999}')

    run(["checkpoint", "--backend", sqlite_uri(path)], capsys)

    backend = SQLiteBackend(str(path))
    try:
        assert backend.get_latest_checkpoint() is None
    finally:
        backend.close()


def test_a_tampered_ledger_can_still_be_checkpointed_deliberately(
    tmp_path: Any, capsys: Any
) -> None:
    path = tmp_path / "ledger.db"
    write_ledger(path, count=3)
    rewrite_payload(path, 1, '{"event":"created","index":999}')

    code, _, _ = run(["checkpoint", "--backend", sqlite_uri(path), "--force"], capsys)

    assert code == EXIT_OK


# ======================================================================================
# anchor
# ======================================================================================


def test_anchoring_publishes_the_stored_checkpoint(tmp_path: Any, capsys: Any) -> None:
    path = tmp_path / "ledger.db"
    anchor = tmp_path / "anchor.jsonl"
    ledger_id = write_ledger(path, count=3)
    run(["checkpoint", "--backend", sqlite_uri(path)], capsys)

    code, _, _ = run(
        ["anchor", "--backend", sqlite_uri(path), "--anchor", str(anchor)], capsys
    )

    assert code == EXIT_OK
    assert FileAnchorSink(str(anchor)).fetch_latest(ledger_id) is not None


def test_anchoring_reports_where_the_checkpoint_landed(tmp_path: Any, capsys: Any) -> None:
    path = tmp_path / "ledger.db"
    anchor = tmp_path / "anchor.jsonl"
    write_ledger(path, count=3)
    run(["checkpoint", "--backend", sqlite_uri(path)], capsys)

    _, out, _ = run(
        ["anchor", "--backend", sqlite_uri(path), "--anchor", str(anchor), "--json"], capsys
    )

    assert json.loads(out)["location"] == str(anchor)


def test_anchoring_a_store_with_no_checkpoint_is_an_operational_error(
    tmp_path: Any, capsys: Any
) -> None:
    path = tmp_path / "ledger.db"
    write_ledger(path, count=3)

    code, _, err = run(
        ["anchor", "--backend", sqlite_uri(path), "--anchor", str(tmp_path / "a.jsonl")],
        capsys,
    )

    assert code == EXIT_ERROR
    assert "checkpoint" in err


def test_showing_the_anchor_reports_the_published_head(tmp_path: Any, capsys: Any) -> None:
    path = tmp_path / "ledger.db"
    anchor = tmp_path / "anchor.jsonl"
    write_ledger(path, count=4)
    run(["checkpoint", "--backend", sqlite_uri(path), "--anchor", str(anchor)], capsys)

    code, out, _ = run(
        ["anchor", "--backend", sqlite_uri(path), "--anchor", str(anchor), "--show", "--json"],
        capsys,
    )

    assert (code, json.loads(out)["sequence"]) == (EXIT_OK, 3)


def test_showing_an_anchor_that_holds_nothing_is_not_a_success(
    tmp_path: Any, capsys: Any
) -> None:
    path = tmp_path / "ledger.db"
    write_ledger(path, count=2)

    code, _, _ = run(
        [
            "anchor",
            "--backend",
            sqlite_uri(path),
            "--anchor",
            str(tmp_path / "empty.jsonl"),
            "--show",
        ],
        capsys,
    )

    assert code == EXIT_INVALID


def test_showing_the_anchor_of_an_empty_store_is_not_a_success(
    tmp_path: Any, capsys: Any
) -> None:
    path = tmp_path / "empty.db"
    SQLiteBackend(str(path)).close()

    code, _, _ = run(
        [
            "anchor",
            "--backend",
            sqlite_uri(path),
            "--anchor",
            str(tmp_path / "a.jsonl"),
            "--show",
        ],
        capsys,
    )

    assert code == EXIT_INVALID


# ======================================================================================
# export / verify-bundle
# ======================================================================================


def export_bundle(path: Any, bundle: Any, capsys: Any, *extra: str) -> Any:
    """Export ``path`` to ``bundle`` and return the decoded bundle."""
    code, _, err = run(
        ["export", "--backend", sqlite_uri(path), "--bundle", str(bundle), *extra], capsys
    )
    assert code == EXIT_OK, err
    return json.loads(Path(bundle).read_text(encoding="utf-8"))


def test_exporting_a_ledger_writes_the_bundle_file(tmp_path: Any, capsys: Any) -> None:
    path = tmp_path / "ledger.db"
    bundle = tmp_path / "bundle.json"
    write_ledger(path, count=3)

    code, _, _ = run(["export", "--backend", sqlite_uri(path), "--bundle", str(bundle)], capsys)

    assert (code, bundle.exists()) == (EXIT_OK, True)


def test_the_short_output_flag_names_the_bundle(tmp_path: Any, capsys: Any) -> None:
    """``-o`` is the flag the published README documents, so it has to be a flag.

    1.0.0's README named an option that did not exist; readers copying the line got
    ``exit 2`` and then a ``verify-bundle`` on a file nothing had written.
    """
    path = tmp_path / "ledger.db"
    bundle = tmp_path / "bundle.json"
    write_ledger(path, count=3)

    code, _, _ = run(["export", "--backend", sqlite_uri(path), "-o", str(bundle)], capsys)

    assert (code, bundle.exists()) == (EXIT_OK, True)


def test_the_short_output_flag_is_the_same_option_as_bundle() -> None:
    """An alias that lands in a different attribute is a second bug, not a fix."""
    parser = build_parser()

    short = parser.parse_args(["export", "-b", "memory://", "-o", "written.json"])
    long = parser.parse_args(["export", "-b", "memory://", "--bundle", "written.json"])

    assert (short.bundle, long.bundle) == ("written.json", "written.json")


def test_a_bundle_written_by_the_short_flag_verifies_offline(
    tmp_path: Any, capsys: Any
) -> None:
    """The README's next line is ``verify-bundle bundle.json``; it must find one."""
    path = tmp_path / "ledger.db"
    bundle = tmp_path / "bundle.json"
    write_ledger(path, count=3)
    run(["export", "--backend", sqlite_uri(path), "-o", str(bundle)], capsys)

    code, _, _ = run(["verify-bundle", str(bundle)], capsys)

    assert code == EXIT_OK


def test_a_bundle_declares_the_format_it_is_written_in(tmp_path: Any, capsys: Any) -> None:
    path = tmp_path / "ledger.db"
    write_ledger(path, count=3)

    bundle = export_bundle(path, tmp_path / "bundle.json", capsys)

    assert bundle["format"] == BUNDLE_FORMAT


def test_a_bundle_carries_every_entry(tmp_path: Any, capsys: Any) -> None:
    path = tmp_path / "ledger.db"
    write_ledger(path, count=4)

    bundle = export_bundle(path, tmp_path / "bundle.json", capsys)

    assert [entry["sequence"] for entry in bundle["entries"]] == [0, 1, 2, 3]


def test_a_bundle_carries_the_genesis_digest_entry_zero_links_to(
    tmp_path: Any, capsys: Any
) -> None:
    path = tmp_path / "ledger.db"
    write_ledger(path, count=2)

    bundle = export_bundle(path, tmp_path / "bundle.json", capsys)

    assert bundle["genesis_hash"] == bundle["entries"][0]["previous_hash"]


def test_a_bundle_carries_the_merkle_root_of_its_entries(tmp_path: Any, capsys: Any) -> None:
    path = tmp_path / "ledger.db"
    write_ledger(path, count=4)

    bundle = export_bundle(path, tmp_path / "bundle.json", capsys)

    expected = MerkleTree(bytes.fromhex(e["hash"]) for e in bundle["entries"]).root
    assert bundle["merkle_root"] == expected


def test_a_bundle_carries_the_stored_checkpoint(tmp_path: Any, capsys: Any) -> None:
    path = tmp_path / "ledger.db"
    write_ledger(path, count=3)
    run(["checkpoint", "--backend", sqlite_uri(path)], capsys)

    bundle = export_bundle(path, tmp_path / "bundle.json", capsys)

    assert bundle["checkpoint"]["sequence"] == 2


def test_a_bundle_without_a_checkpoint_says_so(tmp_path: Any, capsys: Any) -> None:
    path = tmp_path / "ledger.db"
    write_ledger(path, count=3)

    bundle = export_bundle(path, tmp_path / "bundle.json", capsys)

    assert bundle["checkpoint"] is None


def test_a_bundle_can_carry_the_public_keys_it_was_signed_with(
    tmp_path: Any, capsys: Any
) -> None:
    path = tmp_path / "ledger.db"
    key = tmp_path / "audit.pem"
    write_signed_ledger(path, key)

    bundle = export_bundle(path, tmp_path / "bundle.json", capsys, "--key", str(key))

    assert bundle["public_keys"] == [key.read_text(encoding="utf-8")]


def test_exporting_an_empty_ledger_is_an_operational_error(tmp_path: Any, capsys: Any) -> None:
    path = tmp_path / "empty.db"
    SQLiteBackend(str(path)).close()

    code, _, err = run(
        ["export", "--backend", sqlite_uri(path), "--bundle", str(tmp_path / "b.json")],
        capsys,
    )

    assert code == EXIT_ERROR
    assert "empty" in err


def test_export_reports_what_it_wrote(tmp_path: Any, capsys: Any) -> None:
    path = tmp_path / "ledger.db"
    bundle = tmp_path / "bundle.json"
    write_ledger(path, count=3)

    _, out, _ = run(
        ["export", "--backend", sqlite_uri(path), "--bundle", str(bundle), "--json"], capsys
    )

    report = json.loads(out)
    assert (report["entry_count"], report["bundle"]) == (3, str(bundle))


def test_the_human_export_report_names_the_file(tmp_path: Any, capsys: Any) -> None:
    path = tmp_path / "ledger.db"
    bundle = tmp_path / "bundle.json"
    write_ledger(path, count=3)

    _, out, _ = run(
        ["export", "--backend", sqlite_uri(path), "--bundle", str(bundle)], capsys
    )

    assert str(bundle) in out


def test_a_bundle_from_a_healthy_ledger_verifies(tmp_path: Any, capsys: Any) -> None:
    path = tmp_path / "ledger.db"
    bundle = tmp_path / "bundle.json"
    write_ledger(path, count=4)
    export_bundle(path, bundle, capsys)

    code, out, _ = run(["verify-bundle", str(bundle)], capsys)

    assert code == EXIT_OK
    assert "VERIFIED" in out


def test_a_bundle_verifies_after_the_database_it_came_from_is_deleted(
    tmp_path: Any, capsys: Any
) -> None:
    # The whole point of a bundle: a third party with no access to the store, and no
    # database driver installed, can still check the evidence.
    path = tmp_path / "ledger.db"
    bundle = tmp_path / "bundle.json"
    write_ledger(path, count=4)
    export_bundle(path, bundle, capsys)
    for leftover in tmp_path.glob("ledger.db*"):
        leftover.unlink()

    code, _, _ = run(["verify-bundle", str(bundle)], capsys)

    assert code == EXIT_OK


def test_a_bundle_verification_opens_no_database(tmp_path: Any, capsys: Any) -> None:
    # Proven rather than asserted, in a subprocess: the optional drivers are made
    # unimportable and sqlite3.connect is replaced with a landmine. (sqlite3 itself is
    # still imported, because core.verification imports signledger.backends.base - see
    # the report. Importing a stdlib module is not database access; opening one is.)
    path = tmp_path / "ledger.db"
    bundle = tmp_path / "bundle.json"
    write_ledger(path, count=3)
    export_bundle(path, bundle, capsys)

    program = (
        "import sys, sqlite3\n"
        "blocked = {'psycopg', 'pymongo'}\n"
        "class Block:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] in blocked:\n"
        "            raise ImportError(name)\n"
        "for name in [n for n in sys.modules if n.split('.')[0] in blocked]:\n"
        "    del sys.modules[name]\n"
        "sys.meta_path.insert(0, Block())\n"
        "def landmine(*args, **kwargs):\n"
        "    raise AssertionError('verify-bundle opened a database')\n"
        "sqlite3.connect = landmine\n"
        "from signledger.cli import main\n"
        f"sys.exit(main(['verify-bundle', {str(bundle)!r}]))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == EXIT_OK, result.stderr


def test_a_bundle_with_a_rewritten_payload_fails(tmp_path: Any, capsys: Any) -> None:
    path = tmp_path / "ledger.db"
    bundle = tmp_path / "bundle.json"
    write_ledger(path, count=3)
    decoded = export_bundle(path, bundle, capsys)
    decoded["entries"][1]["data"] = {"event": "created", "index": 999}
    bundle.write_text(json.dumps(decoded), encoding="utf-8")

    code, out, _ = run(["verify-bundle", str(bundle), "--json"], capsys)

    assert code == EXIT_INVALID
    assert "bad_hash" in {failure["kind"] for failure in json.loads(out)["failures"]}


def test_a_bundle_with_an_entry_removed_from_the_middle_fails(
    tmp_path: Any, capsys: Any
) -> None:
    path = tmp_path / "ledger.db"
    bundle = tmp_path / "bundle.json"
    write_ledger(path, count=4)
    decoded = export_bundle(path, bundle, capsys)
    del decoded["entries"][2]
    bundle.write_text(json.dumps(decoded), encoding="utf-8")

    code, out, _ = run(["verify-bundle", str(bundle), "--json"], capsys)

    kinds = {failure["kind"] for failure in json.loads(out)["failures"]}
    assert code == EXIT_INVALID
    assert "sequence_gap" in kinds


def test_a_bundle_whose_merkle_root_was_swapped_fails(tmp_path: Any, capsys: Any) -> None:
    path = tmp_path / "ledger.db"
    bundle = tmp_path / "bundle.json"
    write_ledger(path, count=3)
    decoded = export_bundle(path, bundle, capsys)
    decoded["merkle_root"] = "ff" * 32
    bundle.write_text(json.dumps(decoded), encoding="utf-8")

    code, out, _ = run(["verify-bundle", str(bundle), "--json"], capsys)

    kinds = {failure["kind"] for failure in json.loads(out)["failures"]}
    assert code == EXIT_INVALID
    assert "bundle_merkle_mismatch" in kinds


def test_a_bundle_whose_genesis_was_swapped_fails(tmp_path: Any, capsys: Any) -> None:
    path = tmp_path / "ledger.db"
    bundle = tmp_path / "bundle.json"
    write_ledger(path, count=3)
    decoded = export_bundle(path, bundle, capsys)
    decoded["genesis_hash"] = "ff" * 32
    bundle.write_text(json.dumps(decoded), encoding="utf-8")

    code, out, _ = run(["verify-bundle", str(bundle), "--json"], capsys)

    kinds = {failure["kind"] for failure in json.loads(out)["failures"]}
    assert code == EXIT_INVALID
    assert "bundle_genesis_mismatch" in kinds


def test_a_bundle_whose_declared_count_was_swapped_fails(tmp_path: Any, capsys: Any) -> None:
    path = tmp_path / "ledger.db"
    bundle = tmp_path / "bundle.json"
    write_ledger(path, count=3)
    decoded = export_bundle(path, bundle, capsys)
    decoded["entry_count"] = 99
    bundle.write_text(json.dumps(decoded), encoding="utf-8")

    code, out, _ = run(["verify-bundle", str(bundle), "--json"], capsys)

    kinds = {failure["kind"] for failure in json.loads(out)["failures"]}
    assert code == EXIT_INVALID
    assert "bundle_count_mismatch" in kinds


def test_a_bundle_carrying_a_checkpoint_that_no_longer_matches_fails(
    tmp_path: Any, capsys: Any
) -> None:
    path = tmp_path / "ledger.db"
    bundle = tmp_path / "bundle.json"
    write_ledger(path, count=4)
    run(["checkpoint", "--backend", sqlite_uri(path)], capsys)
    decoded = export_bundle(path, bundle, capsys)
    # A truncated export: the checkpoint still remembers four entries.
    decoded["entries"] = decoded["entries"][:2]
    decoded["entry_count"] = 2
    decoded["merkle_root"] = MerkleTree(
        bytes.fromhex(entry["hash"]) for entry in decoded["entries"]
    ).root
    bundle.write_text(json.dumps(decoded), encoding="utf-8")

    code, out, _ = run(["verify-bundle", str(bundle), "--json"], capsys)

    kinds = {failure["kind"] for failure in json.loads(out)["failures"]}
    assert code == EXIT_INVALID
    assert {"head_mismatch", "count_mismatch"} <= kinds


def test_a_bundle_with_no_entries_is_never_a_success(tmp_path: Any, capsys: Any) -> None:
    bundle = tmp_path / "bundle.json"
    bundle.write_text(
        json.dumps(
            {
                "format": BUNDLE_FORMAT,
                "ledger_id": "abc",
                "hash_algorithm": "sha256",
                "genesis_hash": "",
                "entry_count": 0,
                "merkle_root": "",
                "entries": [],
                "checkpoint": None,
                "public_keys": [],
            }
        ),
        encoding="utf-8",
    )

    code, _, _ = run(["verify-bundle", str(bundle)], capsys)

    assert code == EXIT_INVALID


def test_a_bundle_in_an_unknown_format_is_an_operational_error(
    tmp_path: Any, capsys: Any
) -> None:
    bundle = tmp_path / "bundle.json"
    bundle.write_text(json.dumps({"format": "somebody/else/9"}), encoding="utf-8")

    code, _, err = run(["verify-bundle", str(bundle)], capsys)

    assert code == EXIT_ERROR
    assert "format" in err


def test_a_bundle_that_is_not_json_is_an_operational_error(tmp_path: Any, capsys: Any) -> None:
    bundle = tmp_path / "bundle.json"
    bundle.write_text("{not json at all}", encoding="utf-8")

    code, _, err = run(["verify-bundle", str(bundle)], capsys)

    assert code == EXIT_ERROR
    assert "JSON" in err


def test_a_bundle_that_is_a_json_list_is_an_operational_error(
    tmp_path: Any, capsys: Any
) -> None:
    bundle = tmp_path / "bundle.json"
    bundle.write_text("[1, 2, 3]", encoding="utf-8")

    code, _, err = run(["verify-bundle", str(bundle)], capsys)

    assert code == EXIT_ERROR
    assert "object" in err


def test_a_bundle_missing_a_required_field_is_an_operational_error(
    tmp_path: Any, capsys: Any
) -> None:
    bundle = tmp_path / "bundle.json"
    bundle.write_text(json.dumps({"format": BUNDLE_FORMAT}), encoding="utf-8")

    code, _, err = run(["verify-bundle", str(bundle)], capsys)

    assert code == EXIT_ERROR
    assert "missing" in err


def test_a_bundle_whose_entries_are_malformed_is_an_operational_error(
    tmp_path: Any, capsys: Any
) -> None:
    path = tmp_path / "ledger.db"
    bundle = tmp_path / "bundle.json"
    write_ledger(path, count=2)
    decoded = export_bundle(path, bundle, capsys)
    decoded["entries"][0]["timestamp"] = "not a timestamp"
    bundle.write_text(json.dumps(decoded), encoding="utf-8")

    code, _, err = run(["verify-bundle", str(bundle)], capsys)

    assert code == EXIT_ERROR
    assert "entry" in err


def test_a_bundle_whose_entry_list_is_not_a_list_is_an_operational_error(
    tmp_path: Any, capsys: Any
) -> None:
    bundle = tmp_path / "bundle.json"
    bundle.write_text(
        json.dumps(
            {
                "format": BUNDLE_FORMAT,
                "ledger_id": "abc",
                "hash_algorithm": "sha256",
                "genesis_hash": "",
                "entry_count": 0,
                "merkle_root": "",
                "entries": {"nope": 1},
                "checkpoint": None,
                "public_keys": [],
            }
        ),
        encoding="utf-8",
    )

    code, _, err = run(["verify-bundle", str(bundle)], capsys)

    assert code == EXIT_ERROR
    assert "entries" in err


def test_a_bundle_file_that_does_not_exist_is_an_operational_error(
    tmp_path: Any, capsys: Any
) -> None:
    code, _, err = run(["verify-bundle", str(tmp_path / "nope.json")], capsys)

    assert code == EXIT_ERROR
    assert "FileNotFoundError" in err


def test_a_signed_bundle_verified_with_an_external_key_reaches_signed_assurance(
    tmp_path: Any, capsys: Any
) -> None:
    path = tmp_path / "ledger.db"
    key = tmp_path / "audit.pem"
    bundle = tmp_path / "bundle.json"
    write_signed_ledger(path, key)
    export_bundle(path, bundle, capsys)

    code, out, _ = run(["verify-bundle", str(bundle), "--key", str(key), "--json"], capsys)

    assert (code, json.loads(out)["assurance"]) == (EXIT_OK, "signed")


def test_a_signed_bundle_is_only_chain_checked_by_default(tmp_path: Any, capsys: Any) -> None:
    path = tmp_path / "ledger.db"
    bundle = tmp_path / "bundle.json"
    write_signed_ledger(path, tmp_path / "audit.pem")
    export_bundle(path, bundle, capsys, "--key", str(tmp_path / "audit.pem"))

    _, out, _ = run(["verify-bundle", str(bundle), "--json"], capsys)

    assert json.loads(out)["assurance"] == "chain_only"


def test_trusting_the_bundles_own_keys_still_checks_the_signatures(
    tmp_path: Any, capsys: Any
) -> None:
    path = tmp_path / "ledger.db"
    key = tmp_path / "audit.pem"
    bundle = tmp_path / "bundle.json"
    write_signed_ledger(path, key)
    export_bundle(path, bundle, capsys, "--key", str(key))

    code, out, _ = run(["verify-bundle", str(bundle), "--trust-embedded-keys", "--json"], capsys)

    assert (code, json.loads(out)["failures"]) == (EXIT_OK, [])


def test_trusting_the_bundles_own_keys_cannot_raise_the_assurance_level(
    tmp_path: Any, capsys: Any
) -> None:
    # A key delivered inside the thing it authenticates authenticates nothing: whoever
    # rewrote the entries could re-sign them and swap the key in the same edit.
    path = tmp_path / "ledger.db"
    key = tmp_path / "audit.pem"
    bundle = tmp_path / "bundle.json"
    write_signed_ledger(path, key)
    export_bundle(path, bundle, capsys, "--key", str(key))

    _, out, _ = run(["verify-bundle", str(bundle), "--trust-embedded-keys", "--json"], capsys)

    report = json.loads(out)
    assert report["assurance"] == "chain_only"
    assert "carried inside the bundle" in report["assurance_note"]


def test_trusting_embedded_keys_still_reports_a_signature_that_does_not_verify(
    tmp_path: Any, capsys: Any
) -> None:
    path = tmp_path / "ledger.db"
    key = tmp_path / "audit.pem"
    bundle = tmp_path / "bundle.json"
    write_signed_ledger(path, key)
    decoded = export_bundle(path, bundle, capsys, "--key", str(key))
    decoded["entries"][1]["signature"] = base64.b64encode(b"\x00" * 64).decode("ascii")
    bundle.write_text(json.dumps(decoded), encoding="utf-8")

    code, out, _ = run(["verify-bundle", str(bundle), "--trust-embedded-keys", "--json"], capsys)

    kinds = {failure["kind"] for failure in json.loads(out)["failures"]}
    assert code == EXIT_INVALID
    assert "bad_signature" in kinds


def test_trusting_embedded_keys_when_the_bundle_carries_none_is_an_operational_error(
    tmp_path: Any, capsys: Any
) -> None:
    path = tmp_path / "ledger.db"
    bundle = tmp_path / "bundle.json"
    write_signed_ledger(path, tmp_path / "audit.pem")
    export_bundle(path, bundle, capsys)

    code, _, err = run(["verify-bundle", str(bundle), "--trust-embedded-keys"], capsys)

    assert code == EXIT_ERROR
    assert "no public keys" in err


def test_a_bundle_can_be_verified_from_a_sequence_onwards(tmp_path: Any, capsys: Any) -> None:
    path = tmp_path / "ledger.db"
    bundle = tmp_path / "bundle.json"
    write_ledger(path, count=5)
    export_bundle(path, bundle, capsys)

    _, out, _ = run(["verify-bundle", str(bundle), "--from-sequence", "3", "--json"], capsys)

    report = json.loads(out)
    assert (report["ok"], report["first_sequence"]) == (True, 3)


def test_a_bundle_exported_from_a_tampered_ledger_does_not_verify(
    tmp_path: Any, capsys: Any
) -> None:
    path = tmp_path / "ledger.db"
    bundle = tmp_path / "bundle.json"
    write_ledger(path, count=3)
    rewrite_payload(path, 1, '{"event":"created","index":999}')
    export_bundle(path, bundle, capsys)

    code, _, _ = run(["verify-bundle", str(bundle)], capsys)

    assert code == EXIT_INVALID


# ======================================================================================
# format versions this build cannot hash
# ======================================================================================


def insert_future_format_entry(path: Any) -> None:
    """Append a well-linked row written in a format_version this build has no encoder for."""
    backend = SQLiteBackend(str(path))
    try:
        head = backend.get_head()
    finally:
        backend.close()
    future = Entry(
        format_version=3,
        ledger_id=head.ledger_id,
        sequence=head.sequence + 1,
        entry_id="22222222-2222-4222-8222-222222222222",
        timestamp=datetime(2024, 6, 2, tzinfo=timezone.utc),
        data={"event": "from the future"},
        previous_hash=head.hash,
        hash="ff" * 32,
    )
    row = entry_to_row(future)
    connection = sqlite3.connect(str(path))
    try:
        connection.execute(
            "INSERT INTO ledger_entries ({}) VALUES ({})".format(
                ",".join(row), ",".join("?" * len(row))
            ),
            tuple(row.values()),
        )
        connection.commit()
    finally:
        connection.close()


def test_an_entry_in_an_unsupported_format_is_reported_not_guessed_at(
    tmp_path: Any, capsys: Any
) -> None:
    # Contract section 12 promises every historical format stays verifiable. This build
    # registers version 2 only, so anything else must fail closed and say so rather than
    # be reported as a bad hash.
    path = tmp_path / "ledger.db"
    write_ledger(path, count=2)
    insert_future_format_entry(path)

    code, out, _ = run(["verify", "--backend", sqlite_uri(path), "--json"], capsys)

    kinds = {failure["kind"] for failure in json.loads(out)["failures"]}
    assert code == EXIT_INVALID
    assert kinds == {"format_unsupported"}


# ======================================================================================
# migrate
# ======================================================================================


def test_migrating_copies_every_entry(tmp_path: Any, capsys: Any) -> None:
    source = tmp_path / "source.db"
    target = tmp_path / "target.db"
    write_ledger(source, count=4)

    code, _, _ = run(
        ["migrate", "--backend", sqlite_uri(source), "--target", sqlite_uri(target)], capsys
    )

    backend = SQLiteBackend(str(target))
    try:
        assert (code, backend.count()) == (EXIT_OK, 4)
    finally:
        backend.close()


def test_migrated_entries_keep_their_digests_exactly(tmp_path: Any, capsys: Any) -> None:
    source = tmp_path / "source.db"
    target = tmp_path / "target.db"
    write_ledger(source, count=3)
    before = SQLiteBackend(str(source))
    try:
        original = [entry.hash for entry in before.iter_entries()]
    finally:
        before.close()

    run(["migrate", "--backend", sqlite_uri(source), "--target", sqlite_uri(target)], capsys)

    after = SQLiteBackend(str(target))
    try:
        assert [entry.hash for entry in after.iter_entries()] == original
    finally:
        after.close()


def test_the_migrated_store_verifies(tmp_path: Any, capsys: Any) -> None:
    source = tmp_path / "source.db"
    target = tmp_path / "target.db"
    write_ledger(source, count=4)
    run(["migrate", "--backend", sqlite_uri(source), "--target", sqlite_uri(target)], capsys)

    code, _, _ = run(["verify", "--backend", sqlite_uri(target)], capsys)

    assert code == EXIT_OK


def test_migrating_copies_the_checkpoints_too(tmp_path: Any, capsys: Any) -> None:
    source = tmp_path / "source.db"
    target = tmp_path / "target.db"
    write_ledger(source, count=3)
    run(["checkpoint", "--backend", sqlite_uri(source)], capsys)

    run(["migrate", "--backend", sqlite_uri(source), "--target", sqlite_uri(target)], capsys)

    backend = SQLiteBackend(str(target))
    try:
        assert backend.get_latest_checkpoint() is not None
    finally:
        backend.close()


def test_migration_reports_what_it_moved(tmp_path: Any, capsys: Any) -> None:
    source = tmp_path / "source.db"
    target = tmp_path / "target.db"
    write_ledger(source, count=3)

    _, out, _ = run(
        [
            "migrate",
            "--backend",
            sqlite_uri(source),
            "--target",
            sqlite_uri(target),
            "--json",
        ],
        capsys,
    )

    assert json.loads(out)["entries_migrated"] == 3


def test_migrating_a_ledger_that_does_not_verify_is_refused(
    tmp_path: Any, capsys: Any
) -> None:
    # Copying a rewritten chain into a fresh store launders it: the new store is
    # pristine, append-only, and holds the tampered history.
    source = tmp_path / "source.db"
    target = tmp_path / "target.db"
    write_ledger(source, count=3)
    rewrite_payload(source, 1, '{"event":"created","index":999}')

    code, out, _ = run(
        ["migrate", "--backend", sqlite_uri(source), "--target", sqlite_uri(target)], capsys
    )

    assert code == EXIT_INVALID
    assert "refus" in out


def test_a_refused_migration_writes_nothing_into_the_target(
    tmp_path: Any, capsys: Any
) -> None:
    source = tmp_path / "source.db"
    target = tmp_path / "target.db"
    write_ledger(source, count=3)
    rewrite_payload(source, 1, '{"event":"created","index":999}')

    run(["migrate", "--backend", sqlite_uri(source), "--target", sqlite_uri(target)], capsys)

    backend = SQLiteBackend(str(target))
    try:
        assert backend.count() == 0
    finally:
        backend.close()


def test_migrating_a_ledger_this_build_cannot_hash_is_refused(
    tmp_path: Any, capsys: Any
) -> None:
    source = tmp_path / "source.db"
    target = tmp_path / "target.db"
    write_ledger(source, count=2)
    insert_future_format_entry(source)

    code, _, _ = run(
        ["migrate", "--backend", sqlite_uri(source), "--target", sqlite_uri(target)], capsys
    )

    assert code == EXIT_INVALID


def test_migrating_an_empty_ledger_is_an_operational_error(
    tmp_path: Any, capsys: Any
) -> None:
    source = tmp_path / "empty.db"
    SQLiteBackend(str(source)).close()

    code, _, err = run(
        [
            "migrate",
            "--backend",
            sqlite_uri(source),
            "--target",
            sqlite_uri(tmp_path / "t.db"),
        ],
        capsys,
    )

    assert code == EXIT_ERROR
    assert "empty" in err


def test_migrating_into_a_store_that_already_holds_a_chain_is_refused(
    tmp_path: Any, capsys: Any
) -> None:
    source = tmp_path / "source.db"
    target = tmp_path / "target.db"
    write_ledger(source, count=3)
    write_ledger(target, count=1)

    code, _, err = run(
        ["migrate", "--backend", sqlite_uri(source), "--target", sqlite_uri(target)], capsys
    )

    assert code == EXIT_ERROR
    assert "empty" in err


def test_a_migration_record_is_written_when_asked_for(tmp_path: Any, capsys: Any) -> None:
    source = tmp_path / "source.db"
    record = tmp_path / "migration.json"
    write_ledger(source, count=3)

    run(
        [
            "migrate",
            "--backend",
            sqlite_uri(source),
            "--target",
            sqlite_uri(tmp_path / "t.db"),
            "--record",
            str(record),
        ],
        capsys,
    )

    assert record.exists()


def test_a_migration_record_declares_its_own_format(tmp_path: Any, capsys: Any) -> None:
    source = tmp_path / "source.db"
    record = tmp_path / "migration.json"
    write_ledger(source, count=2)

    run(
        [
            "migrate",
            "--backend",
            sqlite_uri(source),
            "--target",
            sqlite_uri(tmp_path / "t.db"),
            "--record",
            str(record),
        ],
        capsys,
    )

    assert json.loads(record.read_text(encoding="utf-8"))["format"] == MIGRATION_FORMAT


def test_a_migration_record_binds_the_source_head_and_count(
    tmp_path: Any, capsys: Any
) -> None:
    source = tmp_path / "source.db"
    record = tmp_path / "migration.json"
    write_ledger(source, count=3)
    backend = SQLiteBackend(str(source))
    try:
        head_hash = backend.get_head().hash
    finally:
        backend.close()

    run(
        [
            "migrate",
            "--backend",
            sqlite_uri(source),
            "--target",
            sqlite_uri(tmp_path / "t.db"),
            "--record",
            str(record),
        ],
        capsys,
    )

    decoded = json.loads(record.read_text(encoding="utf-8"))
    assert (decoded["source_head_hash"], decoded["source_entry_count"]) == (head_hash, 3)


def test_a_migration_record_is_signed_when_a_key_is_given(
    tmp_path: Any, capsys: Any
) -> None:
    source = tmp_path / "source.db"
    record = tmp_path / "migration.json"
    private = tmp_path / "signing.pem"
    write_ledger(source, count=2)
    private.write_text(
        get_private_key_pem(ed25519.Ed25519PrivateKey.generate(), allow_unencrypted=True),
        encoding="utf-8",
    )

    run(
        [
            "migrate",
            "--backend",
            sqlite_uri(source),
            "--target",
            sqlite_uri(tmp_path / "t.db"),
            "--record",
            str(record),
            "--signing-key",
            str(private),
        ],
        capsys,
    )

    decoded = json.loads(record.read_text(encoding="utf-8"))
    assert decoded["signature"] is not None
    assert decoded["key_id"] is not None


def test_an_unsigned_migration_record_says_so(tmp_path: Any, capsys: Any) -> None:
    source = tmp_path / "source.db"
    record = tmp_path / "migration.json"
    write_ledger(source, count=2)

    run(
        [
            "migrate",
            "--backend",
            sqlite_uri(source),
            "--target",
            sqlite_uri(tmp_path / "t.db"),
            "--record",
            str(record),
        ],
        capsys,
    )

    decoded = json.loads(record.read_text(encoding="utf-8"))
    assert decoded["signature"] is None


def test_a_migration_record_names_the_scheme_and_never_the_connection_string(
    tmp_path: Any, capsys: Any
) -> None:
    # Contract section 10: a DSN carries credentials, and this record is meant to be
    # handed to an auditor.
    source = tmp_path / "source.db"
    record = tmp_path / "migration.json"
    write_ledger(source, count=2)

    run(
        [
            "migrate",
            "--backend",
            sqlite_uri(source),
            "--target",
            sqlite_uri(tmp_path / "t.db"),
            "--record",
            str(record),
        ],
        capsys,
    )

    text = record.read_text(encoding="utf-8")
    decoded = json.loads(text)
    assert decoded["source_scheme"] == "sqlite"
    assert str(source) not in text


def test_a_migration_record_is_verifiable_against_the_signing_key(
    tmp_path: Any, capsys: Any
) -> None:
    source = tmp_path / "source.db"
    record = tmp_path / "migration.json"
    private = tmp_path / "signing.pem"
    key = ed25519.Ed25519PrivateKey.generate()
    write_ledger(source, count=2)
    private.write_text(get_private_key_pem(key, allow_unencrypted=True), encoding="utf-8")

    run(
        [
            "migrate",
            "--backend",
            sqlite_uri(source),
            "--target",
            sqlite_uri(tmp_path / "t.db"),
            "--record",
            str(record),
            "--signing-key",
            str(private),
        ],
        capsys,
    )

    decoded = json.loads(record.read_text(encoding="utf-8"))
    assert verify(
        migration_signing_payload(decoded), decoded["signature"], key.public_key()
    )


# ======================================================================================
# the real entry point
#
# These run the console script wiring in a real subprocess. Everything above calls
# main() directly, which would happily keep passing while the installed `signledger`
# command was broken - which is exactly what happened in v1, where an empty cli/
# package shadowed cli.py and the console script could not import at all.
# ======================================================================================


def console_script_target() -> str:
    """The entry point string pyproject actually declares."""
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    section = text.split("[project.scripts]", 1)[1].split("[", 1)[0]
    line = next(part for part in section.splitlines() if part.strip().startswith("signledger"))
    return line.split("=", 1)[1].strip().strip('"')


def write_console_script(tmp_path: Any) -> Any:
    """Generate the shim pip writes for the declared entry point, verbatim."""
    module, _, attribute = console_script_target().partition(":")
    script = tmp_path / "signledger_console"
    script.write_text(
        "import sys\n"
        f"from {module} import {attribute}\n"
        f"sys.exit({attribute}())\n",
        encoding="utf-8",
    )
    return script


def run_process(argv: List[str]) -> Any:
    """Run a real subprocess with the package importable, as an install would make it."""
    return subprocess.run(
        argv,
        cwd=str(REPO_ROOT),
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_the_cli_is_a_module_and_not_a_package() -> None:
    # v1's empty signledger/cli/ directory shadowed cli.py, so the console script's
    # import target did not exist.
    import signledger.cli

    assert not hasattr(signledger.cli, "__path__")


def test_pyproject_points_the_console_script_at_something_importable() -> None:
    module, _, attribute = console_script_target().partition(":")

    imported = __import__(module, fromlist=[attribute])

    assert callable(getattr(imported, attribute))


def test_the_console_script_verifies_a_healthy_ledger(tmp_path: Any) -> None:
    path = tmp_path / "ledger.db"
    write_ledger(path, count=3)
    script = write_console_script(tmp_path)

    result = run_process([sys.executable, str(script), "verify", "--backend", sqlite_uri(path)])

    assert result.returncode == EXIT_OK, result.stderr
    assert "VERIFIED" in result.stdout


def test_the_console_script_exits_one_for_a_tampered_ledger(tmp_path: Any) -> None:
    path = tmp_path / "ledger.db"
    write_ledger(path, count=3)
    rewrite_payload(path, 1, '{"event":"created","index":999}')
    script = write_console_script(tmp_path)

    result = run_process([sys.executable, str(script), "verify", "--backend", sqlite_uri(path)])

    assert result.returncode == EXIT_INVALID


def test_the_console_script_exits_two_for_an_operational_error(tmp_path: Any) -> None:
    script = write_console_script(tmp_path)

    result = run_process([sys.executable, str(script), "verify", "--backend", "redis://x"])

    assert result.returncode == EXIT_ERROR


def test_the_console_script_exits_one_on_an_empty_ledger(tmp_path: Any) -> None:
    path = tmp_path / "empty.db"
    SQLiteBackend(str(path)).close()
    script = write_console_script(tmp_path)

    result = run_process([sys.executable, str(script), "verify", "--backend", sqlite_uri(path)])

    assert result.returncode == EXIT_INVALID
    assert "VERIFIED" not in result.stdout


def test_the_package_runs_as_a_module(tmp_path: Any) -> None:
    path = tmp_path / "ledger.db"
    write_ledger(path, count=2)

    result = run_process(
        [sys.executable, "-m", "signledger", "verify", "--backend", sqlite_uri(path)]
    )

    assert result.returncode == EXIT_OK, result.stderr


def test_running_the_module_with_no_command_prints_help(tmp_path: Any) -> None:
    result = run_process([sys.executable, "-m", "signledger"])

    assert result.returncode == EXIT_ERROR
    assert "verify-bundle" in result.stdout


def test_help_exits_zero(tmp_path: Any) -> None:
    result = run_process([sys.executable, "-m", "signledger", "--help"])

    assert result.returncode == EXIT_OK


def test_an_unknown_subcommand_is_an_operational_error(tmp_path: Any) -> None:
    result = run_process([sys.executable, "-m", "signledger", "frobnicate"])

    assert result.returncode == EXIT_ERROR


def test_every_subcommand_is_reachable_from_the_real_entry_point(tmp_path: Any) -> None:
    for command in (
        "verify",
        "stats",
        "inspect",
        "checkpoint",
        "anchor",
        "export",
        "verify-bundle",
        "migrate",
    ):
        result = run_process([sys.executable, "-m", "signledger", command, "--help"])
        assert result.returncode == EXIT_OK, f"{command}: {result.stderr}"


def test_a_missing_backend_argument_is_an_operational_error() -> None:
    with pytest.raises(SystemExit) as caught:
        main(["verify"])

    assert caught.value.code == EXIT_ERROR


def test_argv_defaults_to_the_process_arguments(
    tmp_path: Any, capsys: Any, monkeypatch: Any
) -> None:
    path = tmp_path / "ledger.db"
    write_ledger(path, count=2)
    monkeypatch.setattr(sys, "argv", ["signledger", "verify", "--backend", sqlite_uri(path)])

    assert main() == EXIT_OK


# ======================================================================================
# optional dependencies
# ======================================================================================


@contextlib.contextmanager
def without(blocked: str, *purge: str) -> Any:
    """Make ``blocked`` unimportable and force ``purge`` to be imported afresh."""
    prefixes = (blocked, *purge)
    saved = {
        name: module
        for name, module in sys.modules.items()
        if any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)
    }
    for name in saved:
        del sys.modules[name]

    class Blocker:
        def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> None:
            if fullname == blocked or fullname.startswith(blocked + "."):
                raise ImportError(f"{fullname} is blocked for this test")
            return

    blocker = Blocker()
    sys.meta_path.insert(0, blocker)
    try:
        yield
    finally:
        sys.meta_path.remove(blocker)
        for name in [n for n in sys.modules if n in saved]:
            del sys.modules[name]
        sys.modules.update(saved)


def test_a_postgres_uri_without_psycopg_names_the_extra_to_install() -> None:
    with without(
        "psycopg", "signledger.backends.postgresql", "signledger.backends.postgres"
    ), pytest.raises(ConfigurationError, match=r"signledger\[postgresql\]"):
        open_backend("postgresql://someone@127.0.0.1:1/nothing")


def test_a_mongodb_uri_without_pymongo_names_the_extra_to_install() -> None:
    with without(
        "pymongo", "signledger.backends.mongodb", "signledger.backends.mongo"
    ), pytest.raises(ConfigurationError, match=r"signledger\[mongodb\]"):
        open_backend("mongodb://127.0.0.1:1/")


def test_using_keys_without_the_signatures_extra_is_an_operational_error(
    tmp_path: Any, capsys: Any
) -> None:
    path = tmp_path / "ledger.db"
    key = tmp_path / "audit.pem"
    write_signed_ledger(path, key)

    with without("cryptography", "signledger.crypto.keys", "signledger.crypto.signatures"):
        code, _, err = run(
            ["verify", "--backend", sqlite_uri(path), "--key", str(key)], capsys
        )

    assert code == EXIT_ERROR
    assert "signledger[signatures]" in err


def test_verifying_a_chain_needs_no_optional_dependency_at_all(tmp_path: Any) -> None:
    path = tmp_path / "ledger.db"
    write_ledger(path, count=3)

    program = (
        "import sys\n"
        "blocked = {'cryptography', 'psycopg', 'pymongo'}\n"
        "class Block:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] in blocked:\n"
        "            raise ImportError(name)\n"
        "sys.meta_path.insert(0, Block())\n"
        "from signledger.cli import main\n"
        f"sys.exit(main(['verify', '--backend', {sqlite_uri(path)!r}]))\n"
    )
    result = run_process([sys.executable, "-c", program])

    assert result.returncode == EXIT_OK, result.stderr


def test_running_with_no_subcommand_prints_help_and_fails(capsys: Any) -> None:
    code, out, _ = run([], capsys)

    assert code == EXIT_ERROR
    assert "verify-bundle" in out


def test_a_bundle_with_a_malformed_checkpoint_is_an_operational_error(
    tmp_path: Any, capsys: Any
) -> None:
    path = tmp_path / "ledger.db"
    bundle = tmp_path / "bundle.json"
    write_ledger(path, count=3)
    run(["checkpoint", "--backend", sqlite_uri(path)], capsys)
    decoded = export_bundle(path, bundle, capsys)
    del decoded["checkpoint"]["head_hash"]
    bundle.write_text(json.dumps(decoded), encoding="utf-8")

    code, _, err = run(["verify-bundle", str(bundle)], capsys)

    assert code == EXIT_ERROR
    assert "checkpoint could not be read" in err


def test_a_bundle_checkpoint_with_an_unreadable_timestamp_is_an_operational_error(
    tmp_path: Any, capsys: Any
) -> None:
    path = tmp_path / "ledger.db"
    bundle = tmp_path / "bundle.json"
    write_ledger(path, count=3)
    run(["checkpoint", "--backend", sqlite_uri(path)], capsys)
    decoded = export_bundle(path, bundle, capsys)
    decoded["checkpoint"]["timestamp"] = "yesterday"
    bundle.write_text(json.dumps(decoded), encoding="utf-8")

    code, _, err = run(["verify-bundle", str(bundle)], capsys)

    assert code == EXIT_ERROR
    assert "checkpoint could not be read" in err


def test_the_module_entry_point_delegates_to_the_cli() -> None:
    # One parser, one main(). Two entry points that each build their own drift apart,
    # and the one nobody runs in CI is the one that breaks.
    import signledger.__main__ as module_entry
    from signledger.cli import main as cli_main

    assert module_entry.main is cli_main
