"""Tests for ``signledger.cli`` and the ``signledger`` console script."""

import base64
import contextlib
import hashlib
import json
import os
import re
import socket
import sqlite3
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Any, Iterator, List, Tuple
from urllib.parse import urlsplit

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec, ed25519, rsa

from signledger import __version__
from signledger.anchor.sinks import FileAnchorSink
from signledger.backends.base import entry_to_row
from signledger.backends.memory import MemoryBackend
from signledger.backends.sqlite import SQLiteBackend
from signledger.cli import (
    ASSURANCE_NOTES,
    BUNDLE_FORMAT,
    EMBEDDED_KEY_NOTE,
    EXIT_ERROR,
    EXIT_INVALID,
    EXIT_OK,
    MIGRATION_FORMAT,
    _sqlite_path,
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
from signledger.crypto.signatures import (
    ECDSASigner,
    Ed25519Signer,
    RSASigner,
    get_private_key_pem,
    verify,
)

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


# ======================================================================================
# Windows paths in a sqlite URI
#
# The first CI run on windows-latest failed 172 tests with one message - "a sqlite URI
# takes a path, not a host" - because a Windows tmp_path is `C:\Users\runneradmin\...`,
# so the backend URI is `sqlite://C:\Users\...\ledger.db` and `urlsplit` reads the `C:`
# as a URI authority.
#
# Every test below is a pure-function test over a Windows-shaped *string*, so all of them
# execute on macOS and Linux too. A test that only ran under `os.name == "nt"` would be
# skipped on the machine this bug is fixed on, and would protect nobody.
# ======================================================================================

#: What `tmp_path / "ledger.db"` renders as on a GitHub windows-latest runner.
WINDOWS_TMP_DB = r"C:\Users\runneradmin\AppData\Local\Temp\pytest-0\test_verify0\ledger.db"


def test_the_sqlite_uri_this_suite_builds_survives_a_windows_tmp_path() -> None:
    # This is the CI failure itself, reproduced without Windows: `sqlite_uri` is what
    # every store-backed test in this file calls, and `PureWindowsPath` renders a
    # Windows path on any OS.
    windows_tmp_path = PureWindowsPath(WINDOWS_TMP_DB)

    assert _sqlite_path(sqlite_uri(windows_tmp_path)) == WINDOWS_TMP_DB


def test_a_sqlite_uri_with_a_drive_letter_authority_names_a_file_on_that_drive() -> None:
    # `C:` is a drive, never a host: a host cannot contain a backslash, and no host is
    # one letter followed by a colon.
    assert _sqlite_path("sqlite://" + WINDOWS_TMP_DB) == WINDOWS_TMP_DB
    assert _sqlite_path("sqlite://C:/Users/me/audit.db") == "C:/Users/me/audit.db"
    assert _sqlite_path("sqlite://d:/data/audit.db") == "d:/data/audit.db"


def test_a_sqlite_uri_with_a_drive_letter_and_one_colon_names_the_same_file() -> None:
    # The documented `sqlite:<path>` spelling, which is what `--backend` advertises.
    assert _sqlite_path(r"sqlite:C:\Users\me\audit.db") == r"C:\Users\me\audit.db"
    assert _sqlite_path("sqlite:C:/Users/me/audit.db") == "C:/Users/me/audit.db"


def test_a_triple_slash_sqlite_uri_with_a_drive_letter_names_the_same_file(
    monkeypatch: Any,
) -> None:
    # `sqlite:///C:/x.db` is the URL spelling of an absolute Windows path; SQLite's own
    # `file:` parser drops that leading slash for exactly the same reason.
    #
    # Only on Windows. `/c:/data/audit.db` is an ordinary absolute path on POSIX - a
    # directory called `c:` at the root - and dropping the slash there would quietly make
    # it relative. The two spellings cannot be told apart, so the platform decides.
    monkeypatch.setattr(os, "name", "nt")

    assert _sqlite_path("sqlite:///C:/Users/me/audit.db") == "C:/Users/me/audit.db"
    assert _sqlite_path(r"sqlite:///C:\Users\me\audit.db") == r"C:\Users\me\audit.db"


def test_a_sqlite_uri_naming_a_unc_share_keeps_both_leading_backslashes() -> None:
    # `\\server\share\audit.db` is a real Windows path shape, and it is unambiguous: a
    # backslash is not legal in a host, so an authority starting with one is a UNC path.
    assert _sqlite_path(r"sqlite:\\server\share\audit.db") == r"\\server\share\audit.db"
    assert _sqlite_path(r"sqlite://\\server\share\audit.db") == r"\\server\share\audit.db"
    assert _sqlite_path("sqlite:////server/share/audit.db") == "//server/share/audit.db"


def test_a_posix_absolute_path_is_still_read_as_the_path_it_is() -> None:
    # The drive-letter rule must not touch the ordinary reading of a rooted path.
    assert _sqlite_path("sqlite:///var/lib/audit.db") == "/var/lib/audit.db"
    assert _sqlite_path("sqlite:/x") == "/x"
    assert _sqlite_path("sqlite:audit.db") == "audit.db"


def test_a_sqlite_uri_with_a_real_host_is_still_refused_after_the_windows_fix() -> None:
    # Widening the rule to admit `C:` must not widen it to admit everything: guessing
    # what `sqlite://data/ledger.db` meant would open a file the operator never named.
    for uri in (
        "sqlite://somehost/path",
        "sqlite://data/ledger.db",
        "sqlite://cc:/data.db",
        "sqlite://ledger.db/x",
    ):
        with pytest.raises(ConfigurationError, match="host"):
            _sqlite_path(uri)


def test_the_host_refusal_still_names_the_fix_and_still_quotes_no_uri() -> None:
    with pytest.raises(ConfigurationError) as caught:
        _sqlite_path("sqlite://backup:hunter2@fileserver/audit.db")

    message = str(caught.value)
    assert "sqlite:///absolute/path.db" in message
    assert "sqlite:relative/path.db" in message
    # Contract section 10: a URI can carry a credential, so none of it is echoed.
    assert "hunter2" not in message
    assert "fileserver" not in message


def test_a_bare_windows_path_is_reported_as_a_missing_scheme_not_an_unknown_one() -> None:
    # `urlsplit(r"C:\ledger.db").scheme` is "c". Reporting `unsupported backend scheme
    # 'c'` names a scheme the operator never typed and offers no fix.
    with pytest.raises(ConfigurationError) as caught:
        open_backend(r"C:\ledger.db")

    message = str(caught.value)
    assert "sqlite:" in message
    assert "scheme 'c'" not in message


def test_every_uri_the_windows_refusals_document_is_one_this_parser_reads_back() -> None:
    """A message that tells an operator to write a form the parser mangles is the bug.

    ``--help`` and the schemeless refusal are already held to this (see
    ``documented_sqlite_uris`` below); the two refusals a Windows operator actually hits
    were not, so the fix they name has to be checked the same way. Pure: each form is
    parsed, never opened.
    """
    messages = []
    with pytest.raises(ConfigurationError) as caught:
        _sqlite_path("sqlite://fileserver/audit.db")
    messages.append(str(caught.value))
    with pytest.raises(ConfigurationError) as caught:
        open_backend(r"C:\ledger.db")
    messages.append(str(caught.value))

    documented = [form for text in messages for form in documented_sqlite_uris(text)]
    assert [form for form in documented if "C:" in form], "no Windows form is offered"
    for form in documented:
        # Read back as a Windows path on every OS: the file each message names, not a
        # trailing bracket or a slash the parser failed to drop.
        assert PureWindowsPath(_sqlite_path(form)).suffix == ".db", form


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
    """``--force`` overrides the refusal to export, and nothing else.

    Exporting a broken ledger is refused by default (an operator archiving on a schedule
    needs the signal), but forensics needs the evidence out of the store - so the escape
    hatch exists, and what comes out of it still fails verification.
    """
    path = tmp_path / "ledger.db"
    bundle = tmp_path / "bundle.json"
    write_ledger(path, count=3)
    rewrite_payload(path, 1, '{"event":"created","index":999}')
    export_bundle(path, bundle, capsys, "--force")

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


# ======================================================================================
# the key-handling surface an operator touches during an incident
#
# Everything below drives real key material through the command line: real Ed25519,
# ECDSA and RSA PEMs on disk, real passphrase-encrypted PEMs, and the unknown_key /
# bad_signature verdicts reached through argv rather than through the Python API.
#
# This is the surface someone reaches for at 3am with a suspected breach, so the things
# under test are the ones that would mislead them: an exit code that says nothing is
# wrong, an assurance level that claims more than was measured, and a passphrase that
# ends up in a CI log.
# ======================================================================================

#: A real passphrase, used to encrypt a real PEM. Never appears in any output.
PASSPHRASE = "correct horse battery staple"

#: What an operator fat-fingers instead.
WRONG_PASSPHRASE = "correct horse battery stapl"


def write_ledger_signed_by(path: Any, signer: Any, count: int = 3) -> str:
    """A real SQLite ledger signed by ``signer``; returns its ledger id."""
    ledger = Ledger(SQLiteBackend(str(path)), enable_signatures=True, signer=signer)
    try:
        for index in range(count):
            ledger.append({"event": "created", "index": index})
        return ledger.ledger_id
    finally:
        ledger.close()


def write_public_pem(path: Any, signer: Any) -> Any:
    """Publish ``signer``'s public key to ``path``, the way an operator distributes one."""
    Path(path).write_text(signer.public_key_pem(), encoding="utf-8")
    return path


def write_encrypted_signing_key(path: Any, passphrase: str = PASSPHRASE) -> Any:
    """A real passphrase-encrypted Ed25519 private key PEM on disk."""
    Path(path).write_text(
        get_private_key_pem(
            ed25519.Ed25519PrivateKey.generate(), password=passphrase.encode("utf-8")
        ),
        encoding="utf-8",
    )
    return path


def swap_signatures(path: Any, first: int, second: int) -> None:
    """Exchange two entries' signatures, leaving every digest and ``key_id`` alone.

    Both signatures stay valid signatures by the ledger's own key, and ``signature`` is
    outside the hash preimage - so the chain still verifies and the ``key_id`` still
    resolves. The only thing left that can catch this is the signature check itself.
    """
    unlock_store(path)
    connection = sqlite3.connect(str(path))
    try:
        held = dict(
            connection.execute(
                "SELECT sequence, signature FROM ledger_entries WHERE sequence IN (?, ?)",
                (first, second),
            ).fetchall()
        )
        connection.execute(
            "UPDATE ledger_entries SET signature = ? WHERE sequence = ?", (held[second], first)
        )
        connection.execute(
            "UPDATE ledger_entries SET signature = ? WHERE sequence = ?", (held[first], second)
        )
        connection.commit()
    finally:
        connection.close()


def scribble_over_signatures(path: Any) -> None:
    """Replace every signature with well-formed base64 that signs nothing."""
    unlock_store(path)
    connection = sqlite3.connect(str(path))
    try:
        connection.execute(
            "UPDATE ledger_entries SET signature = ?",
            (base64.b64encode(b"\x00" * 64).decode("ascii"),),
        )
        connection.commit()
    finally:
        connection.close()


# --------------------------------------------------------------------------------------
# --key with an externally supplied public key
# --------------------------------------------------------------------------------------


def test_an_ecdsa_signed_ledger_verifies_against_the_public_pem_beside_it(
    tmp_path: Any, capsys: Any
) -> None:
    # p384, not the default p256: the CLI resolves a key by its own fingerprint, so it
    # must never need telling which curve or algorithm the ledger was signed with.
    path = tmp_path / "ledger.db"
    signer = ECDSASigner.generate("p384")
    key = write_public_pem(tmp_path / "audit.pem", signer)
    write_ledger_signed_by(path, signer)

    code, out, _ = run(
        ["verify", "--backend", sqlite_uri(path), "--key", str(key), "--json"], capsys
    )

    assert (code, json.loads(out)["assurance"]) == (EXIT_OK, "signed")


def test_an_rsa_signed_ledger_verifies_against_the_public_pem_beside_it(
    tmp_path: Any, capsys: Any
) -> None:
    path = tmp_path / "ledger.db"
    signer = RSASigner.generate(key_size=2048)
    key = write_public_pem(tmp_path / "audit.pem", signer)
    write_ledger_signed_by(path, signer)

    code, out, _ = run(
        ["verify", "--backend", sqlite_uri(path), "--key", str(key), "--json"], capsys
    )

    assert (code, json.loads(out)["assurance"]) == (EXIT_OK, "signed")


def test_a_stranger_who_holds_the_wrong_public_key_gets_a_non_zero_exit(
    tmp_path: Any, capsys: Any
) -> None:
    # `signledger verify || alert` is the whole point of the exit code. A verification
    # that could not check what it was asked to check must never exit 0.
    path = tmp_path / "ledger.db"
    write_signed_ledger(path, tmp_path / "audit.pem")
    stranger = write_public_pem(tmp_path / "stranger.pem", Ed25519Signer.generate())

    code, out, _ = run(
        ["verify", "--backend", sqlite_uri(path), "--key", str(stranger), "--json"], capsys
    )

    assert (code, json.loads(out)["assurance"]) == (EXIT_INVALID, "none")


def test_every_pem_in_a_key_directory_is_loaded_not_just_the_first(
    tmp_path: Any, capsys: Any
) -> None:
    # A key ring holds every key an auditor has been given. Loading one of them would make
    # a correctly signed ledger look like it names a key nobody can supply.
    path = tmp_path / "ledger.db"
    keys = tmp_path / "keys"
    keys.mkdir()
    signer = Ed25519Signer.generate()
    write_public_pem(keys / "a-stranger.pem", Ed25519Signer.generate())
    write_public_pem(keys / "b-audit.pem", signer)
    write_public_pem(keys / "c-stranger.pem", Ed25519Signer.generate())
    write_ledger_signed_by(path, signer)

    code, out, _ = run(
        ["verify", "--backend", sqlite_uri(path), "--key", str(keys), "--json"], capsys
    )

    assert (code, json.loads(out)["assurance"]) == (EXIT_OK, "signed")


def test_each_repeated_key_option_is_loaded(tmp_path: Any, capsys: Any) -> None:
    # `--key` is repeatable, and the ledger's own key is deliberately not the last one:
    # taking only the final path would silently discard the rest of the key ring.
    path = tmp_path / "ledger.db"
    signer = Ed25519Signer.generate()
    mine = write_public_pem(tmp_path / "audit.pem", signer)
    stranger = write_public_pem(tmp_path / "stranger.pem", Ed25519Signer.generate())
    write_ledger_signed_by(path, signer)

    code, out, _ = run(
        [
            "verify",
            "--backend",
            sqlite_uri(path),
            "--key",
            str(mine),
            "--key",
            str(stranger),
            "--json",
        ],
        capsys,
    )

    assert (code, json.loads(out)["assurance"]) == (EXIT_OK, "signed")


def test_supplying_a_key_for_a_ledger_that_was_never_signed_fails_closed(
    tmp_path: Any, capsys: Any
) -> None:
    # An operator who passes --key is asking "are these signed by this key?". The honest
    # answer for an unsigned ledger is no, not "there was nothing to check, so yes".
    path = tmp_path / "ledger.db"
    write_ledger(path, count=3)
    key = write_public_pem(tmp_path / "audit.pem", Ed25519Signer.generate())

    code, out, _ = run(
        ["verify", "--backend", sqlite_uri(path), "--key", str(key), "--json"], capsys
    )

    assert code == EXIT_INVALID
    assert {failure["kind"] for failure in json.loads(out)["failures"]} == {"missing_signature"}


# --------------------------------------------------------------------------------------
# bad_signature, reached through the command line
# --------------------------------------------------------------------------------------


def test_two_signatures_swapped_between_entries_are_reported_as_bad_signatures(
    tmp_path: Any, capsys: Any
) -> None:
    # Both signatures were made by the ledger's own key, so the key_id resolves and the
    # chain is untouched: unknown_key and bad_hash are both the wrong verdict here.
    path = tmp_path / "ledger.db"
    key = tmp_path / "audit.pem"
    write_signed_ledger(path, key, count=3)
    swap_signatures(path, 1, 2)

    code, out, _ = run(
        ["verify", "--backend", sqlite_uri(path), "--key", str(key), "--json"], capsys
    )

    assert code == EXIT_INVALID
    assert {failure["kind"] for failure in json.loads(out)["failures"]} == {"bad_signature"}


def test_a_swapped_signature_names_the_sequences_it_was_found_at(
    tmp_path: Any, capsys: Any
) -> None:
    path = tmp_path / "ledger.db"
    key = tmp_path / "audit.pem"
    write_signed_ledger(path, key, count=3)
    swap_signatures(path, 1, 2)

    _, out, _ = run(
        ["verify", "--backend", sqlite_uri(path), "--key", str(key), "--json"], capsys
    )

    assert {failure["sequence"] for failure in json.loads(out)["failures"]} == {1, 2}


def test_a_bad_signature_never_lets_the_report_claim_an_assurance_level(
    tmp_path: Any, capsys: Any
) -> None:
    path = tmp_path / "ledger.db"
    key = tmp_path / "audit.pem"
    write_signed_ledger(path, key, count=3)
    swap_signatures(path, 1, 2)

    _, out, _ = run(
        ["verify", "--backend", sqlite_uri(path), "--key", str(key), "--json"], capsys
    )

    assert json.loads(out)["assurance"] == "none"


def test_a_signature_failure_exits_non_zero_through_the_installed_console_script(
    tmp_path: Any, capsys: Any
) -> None:
    path = tmp_path / "ledger.db"
    key = tmp_path / "audit.pem"
    write_signed_ledger(path, key, count=3)
    swap_signatures(path, 1, 2)
    script = write_console_script(tmp_path)

    result = run_process(
        [sys.executable, str(script), "verify", "-b", sqlite_uri(path), "--key", str(key)]
    )

    assert result.returncode == EXIT_INVALID, result.stderr


# --------------------------------------------------------------------------------------
# verify with no key at all: what a signed ledger reports when nobody checked
# --------------------------------------------------------------------------------------


def test_a_keyless_verify_never_reports_signed_assurance_for_a_signed_ledger(
    tmp_path: Any, capsys: Any
) -> None:
    # Every signature in this store is base64 padding. Nothing checked them, because no
    # key was supplied - so the level reported has to be the one that promises nothing
    # about signatures. Reporting `signed` because the column is populated is the v1
    # defect: generating signatures and never verifying them.
    path = tmp_path / "ledger.db"
    write_signed_ledger(path, tmp_path / "audit.pem", count=3)
    scribble_over_signatures(path)

    code, out, _ = run(["verify", "--backend", sqlite_uri(path), "--json"], capsys)

    report = json.loads(out)
    assert (code, report["assurance"]) == (EXIT_OK, "chain_only")


def test_a_keyless_verify_spells_out_that_a_rewrite_would_go_unnoticed(
    tmp_path: Any, capsys: Any
) -> None:
    # The exit code is 0 here, so the caveat is the only thing standing between a reader
    # and "this ledger is tamper-proof".
    path = tmp_path / "ledger.db"
    write_signed_ledger(path, tmp_path / "audit.pem", count=3)

    _, out, _ = run(["verify", "--backend", sqlite_uri(path)], capsys)

    assert ASSURANCE_NOTES["chain_only"] in out
    assert ASSURANCE_NOTES["signed"] not in out


# --------------------------------------------------------------------------------------
# key files an operator hands over by mistake
# --------------------------------------------------------------------------------------


def test_a_garbage_key_file_is_an_operational_error_not_a_stack_trace(
    tmp_path: Any, capsys: Any
) -> None:
    path = tmp_path / "ledger.db"
    write_ledger(path, count=2)
    junk = tmp_path / "audit.pem"
    junk.write_text("-----BEGIN PUBLIC KEY-----\nnot base64 at all\n", encoding="utf-8")

    code, out, err = run(
        ["verify", "--backend", sqlite_uri(path), "--key", str(junk)], capsys
    )

    assert code == EXIT_ERROR
    assert "public key could not be loaded" in err
    assert "Traceback" not in out + err


def test_a_truncated_key_file_is_an_operational_error(tmp_path: Any, capsys: Any) -> None:
    path = tmp_path / "ledger.db"
    write_ledger(path, count=2)
    signer = Ed25519Signer.generate()
    half = tmp_path / "audit.pem"
    half.write_text(signer.public_key_pem()[:40], encoding="utf-8")

    code, _, err = run(["verify", "--backend", sqlite_uri(path), "--key", str(half)], capsys)

    assert code == EXIT_ERROR
    assert "public key could not be loaded" in err


def test_a_private_key_handed_to_the_key_option_is_refused_without_echoing_it(
    tmp_path: Any, capsys: Any
) -> None:
    # --key takes a public key. An operator who points it at the signing key must get a
    # refusal that does not paste the private key into the terminal, the log or the CI
    # transcript (contract section 10).
    path = tmp_path / "ledger.db"
    write_ledger(path, count=2)
    private = tmp_path / "signing.pem"
    pem = get_private_key_pem(ed25519.Ed25519PrivateKey.generate(), allow_unencrypted=True)
    private.write_text(pem, encoding="utf-8")
    secret = pem.splitlines()[1]

    code, out, err = run(
        ["verify", "--backend", sqlite_uri(path), "--key", str(private), "--json"], capsys
    )

    assert code == EXIT_ERROR
    assert secret not in out + err


def test_a_key_path_that_is_a_directory_of_directories_is_an_operational_error(
    tmp_path: Any, capsys: Any
) -> None:
    # A ".pem" that is really a directory: opened, not stat-ed, so it must surface as an
    # operational error rather than an IsADirectoryError escaping main().
    path = tmp_path / "ledger.db"
    write_ledger(path, count=2)
    keys = tmp_path / "keys"
    (keys / "audit.pem").mkdir(parents=True)

    code, out, err = run(["verify", "--backend", sqlite_uri(path), "--key", str(keys)], capsys)

    assert code == EXIT_ERROR
    assert "Traceback" not in out + err


# --------------------------------------------------------------------------------------
# --signing-key with a real encrypted PEM and --password-env
# --------------------------------------------------------------------------------------


def test_an_encrypted_signing_key_is_refused_when_the_passphrase_is_wrong(
    tmp_path: Any, capsys: Any, monkeypatch: Any
) -> None:
    path = tmp_path / "ledger.db"
    write_ledger(path, count=2)
    private = write_encrypted_signing_key(tmp_path / "signing.pem")
    monkeypatch.setenv("SL_KEY_PASSWORD", WRONG_PASSPHRASE)

    code, _, err = run(
        [
            "checkpoint",
            "--backend",
            sqlite_uri(path),
            "--signing-key",
            str(private),
            "--password-env",
            "SL_KEY_PASSWORD",
        ],
        capsys,
    )

    assert code == EXIT_ERROR
    assert "private key could not be loaded" in err


def test_a_wrong_passphrase_is_never_quoted_back_to_the_operator(
    tmp_path: Any, capsys: Any, monkeypatch: Any
) -> None:
    # Contract section 10. A refusal that echoes the attempted passphrase puts it in the
    # terminal scrollback, the log and the CI transcript of everyone who reads the job.
    path = tmp_path / "ledger.db"
    write_ledger(path, count=2)
    private = write_encrypted_signing_key(tmp_path / "signing.pem")
    monkeypatch.setenv("SL_KEY_PASSWORD", WRONG_PASSPHRASE)

    _, out, err = run(
        [
            "checkpoint",
            "--backend",
            sqlite_uri(path),
            "--signing-key",
            str(private),
            "--password-env",
            "SL_KEY_PASSWORD",
        ],
        capsys,
    )

    assert WRONG_PASSPHRASE not in out + err
    assert PASSPHRASE not in out + err


def test_a_wrong_passphrase_never_reaches_the_json_output(
    tmp_path: Any, capsys: Any, monkeypatch: Any
) -> None:
    path = tmp_path / "ledger.db"
    write_ledger(path, count=2)
    private = write_encrypted_signing_key(tmp_path / "signing.pem")
    monkeypatch.setenv("SL_KEY_PASSWORD", WRONG_PASSPHRASE)

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

    assert (code, WRONG_PASSPHRASE in json.dumps(json.loads(out))) == (EXIT_ERROR, False)


def test_a_wrong_passphrase_ends_in_an_operational_error_and_no_traceback(
    tmp_path: Any, monkeypatch: Any
) -> None:
    # In-process this would surface as a test error; through the console script it is what
    # an operator actually sees. A traceback here would mean cryptography's own exception
    # escaped main(), and its args carry the material that was being decrypted.
    path = tmp_path / "ledger.db"
    write_ledger(path, count=2)
    private = write_encrypted_signing_key(tmp_path / "signing.pem")
    script = write_console_script(tmp_path)
    monkeypatch.setenv("SL_KEY_PASSWORD", WRONG_PASSPHRASE)

    result = run_process(
        [
            sys.executable,
            str(script),
            "checkpoint",
            "-b",
            sqlite_uri(path),
            "--signing-key",
            str(private),
            "--password-env",
            "SL_KEY_PASSWORD",
        ]
    )

    assert result.returncode == EXIT_ERROR
    assert "Traceback" not in result.stdout + result.stderr
    assert WRONG_PASSPHRASE not in result.stdout + result.stderr


def test_a_password_env_var_pointed_at_a_plaintext_key_is_refused_clearly(
    tmp_path: Any, capsys: Any, monkeypatch: Any
) -> None:
    # The inverse mistake: --password-env against a key that is not encrypted. Underneath
    # this is a TypeError, and an unhandled one exits 1 - which a monitoring script reads
    # as "the ledger is broken" rather than "your command line is wrong".
    path = tmp_path / "ledger.db"
    write_ledger(path, count=2)
    private = tmp_path / "signing.pem"
    private.write_text(
        get_private_key_pem(ed25519.Ed25519PrivateKey.generate(), allow_unencrypted=True),
        encoding="utf-8",
    )
    monkeypatch.setenv("SL_KEY_PASSWORD", PASSPHRASE)

    code, _, err = run(
        [
            "checkpoint",
            "--backend",
            sqlite_uri(path),
            "--signing-key",
            str(private),
            "--password-env",
            "SL_KEY_PASSWORD",
        ],
        capsys,
    )

    assert code == EXIT_ERROR
    assert "password mismatch" in err


def test_an_encrypted_signing_key_produces_a_checkpoint_that_verifies(
    tmp_path: Any, capsys: Any, monkeypatch: Any
) -> None:
    # The passphrase is right, so the whole path has to work end to end: unlock the PEM,
    # sign the checkpoint, and produce a signature that verifies against the public key.
    path = tmp_path / "ledger.db"
    write_ledger(path, count=2)
    private = write_encrypted_signing_key(tmp_path / "signing.pem")
    monkeypatch.setenv("SL_KEY_PASSWORD", PASSPHRASE)
    signer = Ed25519Signer.from_pem(
        Path(private).read_text(encoding="utf-8"), PASSPHRASE.encode("utf-8")
    )

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

    assert (code, json.loads(out)["key_id"]) == (EXIT_OK, signer.key_id)


def test_an_encrypted_signing_key_signs_a_migration_record_from_the_environment(
    tmp_path: Any, capsys: Any, monkeypatch: Any
) -> None:
    source = tmp_path / "source.db"
    record = tmp_path / "migration.json"
    write_ledger(source, count=2)
    private = write_encrypted_signing_key(tmp_path / "signing.pem")
    monkeypatch.setenv("SL_KEY_PASSWORD", PASSPHRASE)
    signer = Ed25519Signer.from_pem(
        Path(private).read_text(encoding="utf-8"), PASSPHRASE.encode("utf-8")
    )

    code, _, err = run(
        [
            "migrate",
            "--backend",
            sqlite_uri(source),
            "--target",
            sqlite_uri(tmp_path / "target.db"),
            "--record",
            str(record),
            "--signing-key",
            str(private),
            "--password-env",
            "SL_KEY_PASSWORD",
        ],
        capsys,
    )

    assert code == EXIT_OK, err
    decoded = json.loads(record.read_text(encoding="utf-8"))
    assert verify(
        migration_signing_payload(decoded), decoded["signature"], signer.public_key
    )


def test_a_mistyped_passphrase_leaves_the_migration_target_untouched(
    tmp_path: Any, capsys: Any, monkeypatch: Any
) -> None:
    # The key is what signs the record at the *end* of a migration, but it is the operator
    # input most likely to be wrong. Loading it after the entries have been copied leaves
    # a populated target behind a command that reported failure.
    source = tmp_path / "source.db"
    target = tmp_path / "target.db"
    write_ledger(source, count=3)
    private = write_encrypted_signing_key(tmp_path / "signing.pem")
    monkeypatch.setenv("SL_KEY_PASSWORD", WRONG_PASSPHRASE)

    code, _, _ = run(
        [
            "migrate",
            "--backend",
            sqlite_uri(source),
            "--target",
            sqlite_uri(target),
            "--record",
            str(tmp_path / "migration.json"),
            "--signing-key",
            str(private),
            "--password-env",
            "SL_KEY_PASSWORD",
        ],
        capsys,
    )

    backend = SQLiteBackend(str(target))
    try:
        assert (code, backend.count()) == (EXIT_ERROR, 0)
    finally:
        backend.close()


def test_a_migration_refused_for_a_mistyped_passphrase_can_simply_be_retried(
    tmp_path: Any, capsys: Any, monkeypatch: Any
) -> None:
    # The recovery story. A half-copied target cannot be migrated into a second time -
    # "the migration target must be empty" - and on an append-only store it cannot be
    # emptied either, so a typo would poison the destination permanently.
    source = tmp_path / "source.db"
    target = tmp_path / "target.db"
    record = tmp_path / "migration.json"
    write_ledger(source, count=3)
    private = write_encrypted_signing_key(tmp_path / "signing.pem")
    argv = [
        "migrate",
        "--backend",
        sqlite_uri(source),
        "--target",
        sqlite_uri(target),
        "--record",
        str(record),
        "--signing-key",
        str(private),
        "--password-env",
        "SL_KEY_PASSWORD",
    ]
    monkeypatch.setenv("SL_KEY_PASSWORD", WRONG_PASSPHRASE)
    run(argv, capsys)

    monkeypatch.setenv("SL_KEY_PASSWORD", PASSPHRASE)
    code, _, err = run(argv, capsys)

    assert code == EXIT_OK, err
    assert json.loads(record.read_text(encoding="utf-8"))["signature"] is not None


# --------------------------------------------------------------------------------------
# --trust-embedded-keys: self-certification, and how loudly it says so
# --------------------------------------------------------------------------------------


def test_an_explicit_key_beats_the_bundles_own_keys(tmp_path: Any, capsys: Any) -> None:
    # The bundle carries the key that really did sign it, and the operator supplies a
    # different one. Falling back to the embedded key here would let any bundle prove
    # itself against whatever key it chose to ship - the exact thing --key exists to stop.
    path = tmp_path / "ledger.db"
    key = tmp_path / "audit.pem"
    bundle = tmp_path / "bundle.json"
    write_signed_ledger(path, key)
    export_bundle(path, bundle, capsys, "--key", str(key))
    stranger = write_public_pem(tmp_path / "stranger.pem", Ed25519Signer.generate())

    code, out, _ = run(
        [
            "verify-bundle",
            str(bundle),
            "--key",
            str(stranger),
            "--trust-embedded-keys",
            "--json",
        ],
        capsys,
    )

    assert code == EXIT_INVALID
    assert {failure["kind"] for failure in json.loads(out)["failures"]} == {"unknown_key"}


def test_a_key_obtained_separately_still_earns_signed_assurance_alongside_the_flag(
    tmp_path: Any, capsys: Any
) -> None:
    # --trust-embedded-keys is a fallback, not a downgrade switch: a key the verifier
    # obtained independently is real evidence whether or not the flag is also present.
    path = tmp_path / "ledger.db"
    key = tmp_path / "audit.pem"
    bundle = tmp_path / "bundle.json"
    write_signed_ledger(path, key)
    export_bundle(path, bundle, capsys, "--key", str(key))

    code, out, _ = run(
        ["verify-bundle", str(bundle), "--key", str(key), "--trust-embedded-keys", "--json"],
        capsys,
    )

    assert (code, json.loads(out)["assurance"]) == (EXIT_OK, "signed")


def test_the_human_bundle_report_says_the_keys_came_from_inside_the_bundle(
    tmp_path: Any, capsys: Any
) -> None:
    # Most operators never pass --json. The self-certification caveat has to survive into
    # the output they actually read.
    path = tmp_path / "ledger.db"
    key = tmp_path / "audit.pem"
    bundle = tmp_path / "bundle.json"
    write_signed_ledger(path, key)
    export_bundle(path, bundle, capsys, "--key", str(key))

    _, out, _ = run(["verify-bundle", str(bundle), "--trust-embedded-keys"], capsys)

    assert EMBEDDED_KEY_NOTE in out
    assert ASSURANCE_NOTES["signed"] not in out


def test_trusting_embedded_keys_never_lifts_a_failed_bundle_above_none(
    tmp_path: Any, capsys: Any
) -> None:
    # Contract section 1: NONE is "chain broken or unverified". A bundle whose signature
    # does not verify has established nothing, and the embedded-key caveat is a *ceiling*
    # on assurance, never a floor to lift a failure up to.
    path = tmp_path / "ledger.db"
    key = tmp_path / "audit.pem"
    bundle = tmp_path / "bundle.json"
    write_signed_ledger(path, key)
    decoded = export_bundle(path, bundle, capsys, "--key", str(key))
    decoded["entries"][1]["signature"] = base64.b64encode(b"\x00" * 64).decode("ascii")
    bundle.write_text(json.dumps(decoded), encoding="utf-8")

    code, out, _ = run(
        ["verify-bundle", str(bundle), "--trust-embedded-keys", "--json"], capsys
    )

    assert (code, json.loads(out)["assurance"]) == (EXIT_INVALID, "none")


def test_a_failed_embedded_verification_never_claims_the_signatures_checked_out(
    tmp_path: Any, capsys: Any
) -> None:
    # The embedded note reads "the signatures were checked against keys carried inside the
    # bundle, which proves internal consistency only". Printed against a bundle whose
    # signature did not verify, that sentence is simply false.
    path = tmp_path / "ledger.db"
    key = tmp_path / "audit.pem"
    bundle = tmp_path / "bundle.json"
    write_signed_ledger(path, key)
    decoded = export_bundle(path, bundle, capsys, "--key", str(key))
    decoded["entries"][1]["signature"] = base64.b64encode(b"\x00" * 64).decode("ascii")
    bundle.write_text(json.dumps(decoded), encoding="utf-8")

    _, out, _ = run(["verify-bundle", str(bundle), "--trust-embedded-keys"], capsys)

    assert ASSURANCE_NOTES["none"] in out
    assert EMBEDDED_KEY_NOTE not in out


def test_a_tampered_payload_is_still_reported_when_the_bundles_keys_are_trusted(
    tmp_path: Any, capsys: Any
) -> None:
    # Self-certification is a limit on what a *passing* check is worth. It must not become
    # a way to make a failing one quieter.
    path = tmp_path / "ledger.db"
    key = tmp_path / "audit.pem"
    bundle = tmp_path / "bundle.json"
    write_signed_ledger(path, key)
    decoded = export_bundle(path, bundle, capsys, "--key", str(key))
    decoded["entries"][1]["data"] = {"event": "created", "index": 999}
    bundle.write_text(json.dumps(decoded), encoding="utf-8")

    code, out, _ = run(
        ["verify-bundle", str(bundle), "--trust-embedded-keys", "--json"], capsys
    )

    assert code == EXIT_INVALID
    assert "bad_hash" in {failure["kind"] for failure in json.loads(out)["failures"]}


def test_export_refuses_an_empty_key_directory_exactly_as_verify_does(
    tmp_path: Any, capsys: Any
) -> None:
    """The same typo must not be an error on one command and a silent success on another.

    ``verify --key <empty dir>`` already exits 2 rather than reporting every entry as
    UNKNOWN_KEY, so that a mistyped path is not mistaken for an attack. ``export`` took the
    identical argument, found nothing, and wrote a bundle with ``public_keys: []`` while
    exiting 0 - so an operator archiving a signed ledger got a keyless bundle and a success
    message, and only discovered it whenever someone next tried to verify the archive.
    """
    path = tmp_path / "ledger.db"
    write_ledger(path, count=1)
    empty = tmp_path / "keys"
    empty.mkdir()

    code, _, err = run(
        [
            "export",
            "--backend",
            sqlite_uri(path),
            "--bundle",
            str(tmp_path / "bundle.json"),
            "--key",
            str(empty),
        ],
        capsys,
    )

    assert code == EXIT_ERROR
    assert "no .pem files" in err


def test_export_does_not_write_a_bundle_it_refused_to_build(
    tmp_path: Any, capsys: Any
) -> None:
    """A refused export must leave nothing behind to be mistaken for an archive."""
    path = tmp_path / "ledger.db"
    write_ledger(path, count=1)
    empty = tmp_path / "keys"
    empty.mkdir()
    bundle = tmp_path / "bundle.json"

    run(
        [
            "export",
            "--backend",
            sqlite_uri(path),
            "--bundle",
            str(bundle),
            "--key",
            str(empty),
        ],
        capsys,
    )

    assert not bundle.exists()


def test_an_empty_password_variable_is_reported_as_unset_not_as_a_passphrase(
    tmp_path: Any, capsys: Any, monkeypatch: Any
) -> None:
    """``export SL_PASS=$(cat /secrets/missing)`` sets the variable to an empty string.

    That was treated as a real, empty passphrase, so the operator got the generic "the
    private key could not be loaded - the password may be wrong" instead of being told the
    variable carries nothing. The two failures need different fixes, so they need
    different messages.
    """
    private = tmp_path / "signing.pem"
    private.write_bytes(
        ed25519.Ed25519PrivateKey.generate().private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.BestAvailableEncryption(b"correct horse"),
        )
    )
    monkeypatch.setenv("SL_EMPTY_PASS", "")

    ledger = tmp_path / "ledger.db"
    write_ledger(ledger, count=1)

    code, _, err = run(
        [
            "checkpoint",
            "--backend",
            sqlite_uri(ledger),
            "--signing-key",
            str(private),
            "--password-env",
            "SL_EMPTY_PASS",
        ],
        capsys,
    )

    assert code == EXIT_ERROR
    assert "SL_EMPTY_PASS" in err
    assert "empty" in err


# ======================================================================================
# naming the store: the database, the table and the collection
#
# Everything below was reproduced by a consumer project running against a live server.
# A CLI that cannot be pointed at the store the library was pointed at is a CLI that
# reports on some *other* store - which is worse than not answering at all.
# ======================================================================================

POSTGRES_DSN_ENV = "SIGNLEDGER_TEST_POSTGRES_DSN"
DEFAULT_POSTGRES_DSN = "postgresql://postgres@localhost:5432/signledger_test"

MONGO_URI_ENV = "SIGNLEDGER_TEST_MONGO_URI"
DEFAULT_MONGO_URI = "mongodb://localhost:27017"


def store_prefix() -> str:
    """A table/collection prefix no other module in the suite uses."""
    return "slcli_" + uuid.uuid4().hex[:12]


def _require_server(url: str, default_port: int, label: str) -> str:
    """Skip - never fail - when no live server answers. Names host:port, never the URI."""
    parts = urlsplit(url)
    host, port = parts.hostname or "localhost", parts.port or default_port
    try:
        with socket.create_connection((host, port), timeout=1.0):
            pass
    except OSError as err:
        pytest.skip(f"no live {label} server at {host}:{port} ({type(err).__name__})")
    return url


@pytest.fixture
def pg_store(request: Any) -> Iterator[Tuple[str, str]]:
    """A live PostgreSQL DSN plus a table prefix nothing else touches."""
    pytest.importorskip("psycopg", reason="psycopg (v3) is not installed")
    import psycopg
    from psycopg import sql

    dsn = _require_server(
        os.environ.get(POSTGRES_DSN_ENV, DEFAULT_POSTGRES_DSN), 5432, "PostgreSQL"
    )
    prefix = store_prefix()
    try:
        yield dsn, prefix
    finally:
        with psycopg.connect(dsn, autocommit=True) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT tablename FROM pg_tables WHERE schemaname = 'public' "
                    "AND tablename LIKE %s",
                    (prefix + "%",),
                )
                for (name,) in cursor.fetchall():
                    cursor.execute(
                        sql.SQL("DROP TABLE IF EXISTS {} CASCADE").format(sql.Identifier(name))
                    )


@pytest.fixture
def mongo_store() -> Iterator[Tuple[str, str, str]]:
    """A live MongoDB URI, a throwaway *database* and a collection prefix.

    The database is deliberately never ``signledger`` - the backend's own default is
    exactly what a URI-borne database name has to be able to override.
    """
    pytest.importorskip("pymongo", reason="pymongo is not installed")
    import pymongo

    uri = _require_server(os.environ.get(MONGO_URI_ENV, DEFAULT_MONGO_URI), 27017, "MongoDB")
    database = "signledger_test_" + uuid.uuid4().hex[:12]
    try:
        yield uri, database, store_prefix()
    finally:
        with pymongo.MongoClient(uri, serverSelectionTimeoutMS=1500) as client:
            client.drop_database(database)


def mongo_uri_for(uri: str, database: str) -> str:
    """The URI an operator writes when they mean *that* database."""
    return uri.rstrip("/") + "/" + database


def write_postgres_ledger(dsn: str, prefix: str, count: int = 3) -> str:
    """A real PostgreSQL ledger on tables only this test knows about."""
    from signledger.backends.postgresql import PostgresBackend

    ledger = Ledger(
        PostgresBackend(
            dsn,
            entries_table=f"{prefix}_entries",
            checkpoints_table=f"{prefix}_checkpoints",
        )
    )
    try:
        for index in range(count):
            ledger.append({"event": "created", "index": index})
        return ledger.ledger_id
    finally:
        ledger.close()


def write_mongo_ledger(uri: str, database: str, count: int = 3, **named: str) -> str:
    """A real MongoDB ledger in ``database``, on whichever collections were named."""
    from signledger.backends.mongodb import MongoBackend

    ledger = Ledger(MongoBackend(uri, database=database, **named))
    try:
        for index in range(count):
            ledger.append({"event": "created", "index": index})
        return ledger.ledger_id
    finally:
        ledger.close()


# --------------------------------------------------------------------------------------
# C1 - the database component of a mongodb URI
# --------------------------------------------------------------------------------------


@pytest.mark.mongo
def test_the_database_a_mongodb_uri_names_is_the_database_that_is_opened(
    mongo_store: Tuple[str, str, str],
) -> None:
    """The URI said one database and the backend opened another one, silently.

    ``MongoBackend`` defaults to ``database="signledger"``, and the CLI passed no
    ``database=`` at all - so every mongodb URI, whatever it named, was answered from
    ``signledger``.
    """
    uri, database, _ = mongo_store

    backend = open_backend(mongo_uri_for(uri, database))
    try:
        assert backend.database_name == database
    finally:
        backend.close()


@pytest.mark.mongo
def test_verify_reports_the_ledger_that_lives_in_the_database_the_uri_names(
    mongo_store: Tuple[str, str, str], capsys: Any
) -> None:
    """The worst failure this project has: VERIFIED, exit 0, about a different store."""
    uri, database, _ = mongo_store
    ledger_id = write_mongo_ledger(uri, database, count=3)

    code, out, _ = run(["verify", "-b", mongo_uri_for(uri, database), "--json"], capsys)

    report = json.loads(out)
    assert (code, report["ledger_id"], report["entries_verified"]) == (EXIT_OK, ledger_id, 3)


@pytest.mark.mongo
def test_a_mongodb_uri_without_a_database_still_opens_the_default_one(
    mongo_store: Tuple[str, str, str],
) -> None:
    """A URI that names no database is not an error; it means "the default"."""
    uri, _, prefix = mongo_store

    backend = open_backend(uri, entries_table=f"{prefix}_entries")
    try:
        assert backend.database_name == "signledger"
    finally:
        backend.close()
        import pymongo

        with pymongo.MongoClient(uri, serverSelectionTimeoutMS=1500) as client:
            for name in client["signledger"].list_collection_names():
                if name.startswith(prefix):
                    client["signledger"].drop_collection(name)


# --------------------------------------------------------------------------------------
# C2 - addressing a table or collection that is not the default
# --------------------------------------------------------------------------------------


def test_verify_can_be_pointed_at_a_non_default_sqlite_table(
    tmp_path: Any, capsys: Any
) -> None:
    """``entries_table=`` is a documented constructor argument. The CLI had no flag."""
    path = tmp_path / "ledger.db"
    ledger = Ledger(
        SQLiteBackend(
            str(path), entries_table="audit_entries", checkpoints_table="audit_checkpoints"
        )
    )
    try:
        for index in range(3):
            ledger.append({"event": "created", "index": index})
        ledger_id = ledger.ledger_id
    finally:
        ledger.close()

    code, out, _ = run(
        [
            "verify",
            "-b",
            sqlite_uri(path),
            "--entries-table",
            "audit_entries",
            "--checkpoints-table",
            "audit_checkpoints",
            "--json",
        ],
        capsys,
    )

    report = json.loads(out)
    assert (code, report["ledger_id"], report["entries_verified"]) == (EXIT_OK, ledger_id, 3)


def test_a_checkpoint_is_written_to_the_checkpoints_table_that_was_named(
    tmp_path: Any, capsys: Any
) -> None:
    """The checkpoints table has to be addressable too, or `checkpoint` writes elsewhere."""
    path = tmp_path / "ledger.db"
    ledger = Ledger(
        SQLiteBackend(
            str(path), entries_table="audit_entries", checkpoints_table="audit_checkpoints"
        )
    )
    try:
        ledger.append({"event": "created"})
    finally:
        ledger.close()

    code, _, err = run(
        [
            "checkpoint",
            "-b",
            sqlite_uri(path),
            "--entries-table",
            "audit_entries",
            "--checkpoints-table",
            "audit_checkpoints",
        ],
        capsys,
    )

    connection = sqlite3.connect(str(path))
    try:
        stored = connection.execute("SELECT COUNT(*) FROM audit_checkpoints").fetchone()[0]
    finally:
        connection.close()
    assert (code, stored) == (EXIT_OK, 1), err


def test_the_collection_spelling_of_the_flag_is_accepted_too(
    tmp_path: Any, capsys: Any
) -> None:
    """MongoDB has collections, not tables. One flag, both words."""
    path = tmp_path / "ledger.db"
    ledger = Ledger(SQLiteBackend(str(path), entries_table="audit_entries"))
    try:
        ledger.append({"event": "created"})
    finally:
        ledger.close()

    code, _, err = run(
        ["verify", "-b", sqlite_uri(path), "--entries-collection", "audit_entries"], capsys
    )

    assert code == EXIT_OK, err


@pytest.mark.postgres
def test_verify_can_be_pointed_at_a_non_default_postgres_table(
    pg_store: Tuple[str, str], capsys: Any
) -> None:
    """`signledger verify || alert` was unavailable to anyone who named their tables."""
    dsn, prefix = pg_store
    ledger_id = write_postgres_ledger(dsn, prefix, count=3)

    code, out, _ = run(
        [
            "verify",
            "-b",
            dsn,
            "--entries-table",
            f"{prefix}_entries",
            "--checkpoints-table",
            f"{prefix}_checkpoints",
            "--json",
        ],
        capsys,
    )

    report = json.loads(out)
    assert (code, report["ledger_id"], report["entries_verified"]) == (EXIT_OK, ledger_id, 3)


@pytest.mark.mongo
def test_verify_can_be_pointed_at_a_non_default_mongo_collection(
    mongo_store: Tuple[str, str, str], capsys: Any
) -> None:
    dsn, database, prefix = mongo_store
    ledger_id = write_mongo_ledger(
        dsn,
        database,
        count=3,
        entries_collection=f"{prefix}_entries",
        checkpoints_collection=f"{prefix}_checkpoints",
    )

    code, out, _ = run(
        [
            "verify",
            "-b",
            mongo_uri_for(dsn, database),
            "--entries-collection",
            f"{prefix}_entries",
            "--checkpoints-collection",
            f"{prefix}_checkpoints",
            "--json",
        ],
        capsys,
    )

    report = json.loads(out)
    assert (code, report["ledger_id"], report["entries_verified"]) == (EXIT_OK, ledger_id, 3)


def test_migrate_can_name_the_tables_in_the_target_store(tmp_path: Any, capsys: Any) -> None:
    """A migration into a named table was impossible; only the source could be addressed."""
    source = tmp_path / "source.db"
    target = tmp_path / "target.db"
    write_ledger(source, count=3)

    code, _, err = run(
        [
            "migrate",
            "-b",
            sqlite_uri(source),
            "--target",
            sqlite_uri(target),
            "--target-entries-table",
            "archive_entries",
            "--target-checkpoints-table",
            "archive_checkpoints",
        ],
        capsys,
    )

    connection = sqlite3.connect(str(target))
    try:
        moved = connection.execute("SELECT COUNT(*) FROM archive_entries").fetchone()[0]
    finally:
        connection.close()
    assert (code, moved) == (EXIT_OK, 3), err


@pytest.mark.parametrize(
    "uri",
    [
        "sqlite:audit.db?entries_table=audit_entries",
        "postgresql://postgres@127.0.0.1:1/nope?checkpoints_table=audit_checkpoints",
        "mongodb://127.0.0.1:1/db?entries_collection=audit_entries",
    ],
    ids=["sqlite", "postgresql", "mongodb"],
)
def test_a_table_named_in_the_uri_query_string_is_refused_by_name(uri: str) -> None:
    """``?entries_table=`` reached libpq and died as ``ProgrammingError``.

    On MongoDB it was worse: pymongo warned, dropped the option and verified the default
    collection of the default database anyway.
    """
    with pytest.raises(ConfigurationError) as caught:
        open_backend(uri)

    message = str(caught.value)
    assert "--entries-table" in message
    assert "--checkpoints-table" in message


def test_a_connection_option_postgres_does_not_accept_is_named(capsys: Any) -> None:
    """A typo in a libpq option used to be ``StorageError: connect failed:

    ProgrammingError`` - which names neither the option nor the fix.
    """
    pytest.importorskip("psycopg", reason="psycopg (v3) is not installed")

    code, _, err = run(
        ["verify", "-b", "postgresql://postgres@127.0.0.1:1/nope?entires_table=x"], capsys
    )

    assert code == EXIT_ERROR
    assert "entires_table" in err
    assert "ProgrammingError" not in err


def test_a_connection_option_postgres_does_accept_still_reaches_the_driver() -> None:
    """Guards the check above: refusing every query parameter would pass that test too."""
    pytest.importorskip("psycopg", reason="psycopg (v3) is not installed")

    # Port 1 refuses immediately: reaching a *connection* failure is the proof.
    with pytest.raises(SignLedgerError) as caught:
        open_backend("postgresql://postgres@127.0.0.1:1/nope?sslmode=disable")

    assert "sslmode" not in str(caught.value)


def test_a_postgres_option_refusal_never_echoes_the_password() -> None:
    pytest.importorskip("psycopg", reason="psycopg (v3) is not installed")

    with pytest.raises(ConfigurationError) as caught:
        open_backend("postgresql://someone:hunter2@127.0.0.1:1/nope?entries_table=x")

    assert "hunter2" not in str(caught.value)


@pytest.mark.parametrize(
    "named",
    [
        {"entries_table": "audit_entries"},
        {"checkpoints_table": "audit_checkpoints"},
    ],
    ids=["entries", "checkpoints"],
)
def test_naming_a_table_on_the_memory_backend_is_refused(named: Any) -> None:
    """The memory backend has no tables. Accepting the flag would silently ignore it."""
    with pytest.raises(ConfigurationError) as caught:
        open_backend("memory://", **named)

    assert "memory" in str(caught.value)


def test_the_backend_help_says_how_to_name_a_table(capsys: Any) -> None:
    with pytest.raises(SystemExit):
        main(["verify", "--help"])

    out = capsys.readouterr().out
    assert "--entries-table" in out
    assert "--checkpoints-table" in out


# --------------------------------------------------------------------------------------
# C3 - a read-only command must not provision the store it is reading
# --------------------------------------------------------------------------------------


def test_verify_does_not_create_the_store_it_was_pointed_at(
    tmp_path: Any, capsys: Any
) -> None:
    """An integrity check that writes to what it is checking is not an integrity check.

    Reproduced against PostgreSQL: ``pg_tables`` was empty before ``verify`` and held
    ``ledger_entries``, ``ledger_checkpoints`` and the append-only trigger after it. The
    SQLite spelling of the same defect is a 32 KiB database file conjured out of a typo.
    """
    missing = tmp_path / "typo.db"

    code, _, err = run(["verify", "-b", sqlite_uri(missing)], capsys)

    assert not missing.exists()
    assert code == EXIT_ERROR
    assert "does not exist" in err


@pytest.mark.parametrize(
    "argv",
    [
        ["verify"],
        ["stats"],
        ["inspect"],
        ["export", "--bundle", "@out"],
        ["anchor", "--anchor", "@out", "--show"],
    ],
    ids=["verify", "stats", "inspect", "export", "anchor-show"],
)
def test_a_read_only_command_refuses_a_store_that_does_not_exist(
    argv: List[str], tmp_path: Any, capsys: Any
) -> None:
    missing = tmp_path / "typo.db"
    filled = [str(tmp_path / "written") if word == "@out" else word for word in argv]

    code, _, err = run([*filled, "-b", sqlite_uri(missing)], capsys)

    assert (code, missing.exists()) == (EXIT_ERROR, False)
    assert "does not exist" in err


def test_the_refusal_to_open_a_missing_store_names_the_path(
    tmp_path: Any, capsys: Any
) -> None:
    """"It does not exist" is only actionable if it says *what* does not exist."""
    missing = tmp_path / "typo.db"

    _, _, err = run(["verify", "-b", sqlite_uri(missing)], capsys)

    assert str(missing) in err


def test_a_read_only_command_refuses_a_throwaway_in_memory_sqlite_database(
    capsys: Any,
) -> None:
    """``sqlite://`` is a database that is created empty on the spot. Nothing to verify."""
    code, _, err = run(["verify", "-b", "sqlite://"], capsys)

    assert code == EXIT_ERROR
    assert "in-memory" in err


def test_checkpoint_still_creates_the_store_it_writes_to(tmp_path: Any, capsys: Any) -> None:
    """Guards the refusals above: a *write* command may provision what it writes to."""
    path = tmp_path / "new.db"

    code, _, err = run(["checkpoint", "-b", sqlite_uri(path)], capsys)

    assert path.exists()
    # ... and then refuses, because an empty ledger has no head to record.
    assert (code, "empty" in err) == (EXIT_ERROR, True)


def test_migrate_refuses_a_source_store_that_does_not_exist(
    tmp_path: Any, capsys: Any
) -> None:
    missing = tmp_path / "typo.db"

    code, _, err = run(
        ["migrate", "-b", sqlite_uri(missing), "--target", sqlite_uri(tmp_path / "t.db")],
        capsys,
    )

    assert (code, missing.exists()) == (EXIT_ERROR, False)
    assert "does not exist" in err


def test_migrate_still_creates_the_target_it_is_asked_to_fill(
    tmp_path: Any, capsys: Any
) -> None:
    """Guards the refusal above: the target of a migration is meant to be created."""
    source = tmp_path / "source.db"
    target = tmp_path / "target.db"
    write_ledger(source, count=2)

    code, _, err = run(
        ["migrate", "-b", sqlite_uri(source), "--target", sqlite_uri(target)], capsys
    )

    assert (code, target.exists()) == (EXIT_OK, True), err


def database_digest(path: Any) -> str:
    """Every byte of the SQLite file, so "wrote nothing" means exactly that."""
    with open(str(path), "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def journal_mode(path: Any) -> str:
    """The journal mode recorded in the file's own header."""
    connection = sqlite3.connect(str(path))
    try:
        return str(connection.execute("PRAGMA journal_mode").fetchone()[0]).lower()
    finally:
        connection.close()


@pytest.mark.parametrize(
    "argv",
    [
        ["verify"],
        ["stats"],
        ["inspect", "--forks"],
        ["export", "--bundle", "@out"],
        ["anchor", "--anchor", "@out", "--show"],
    ],
    ids=["verify", "stats", "inspect", "export", "anchor-show"],
)
def test_a_read_only_command_leaves_an_existing_store_byte_identical(
    argv: List[str], tmp_path: Any, capsys: Any
) -> None:
    """Not merely "creates no tables": the file itself must come back unchanged.

    ``PRAGMA journal_mode = WAL`` is a write to the database header, and it ran on every
    open - so even a ``verify`` against a store that already existed rewrote the file it
    was checking.
    """
    path = tmp_path / "ledger.db"
    write_ledger(path, count=3)
    filled = [str(tmp_path / "written") if word == "@out" else word for word in argv]
    before = database_digest(path)

    run([*filled, "-b", sqlite_uri(path)], capsys)

    assert database_digest(path) == before


def test_verify_does_not_switch_the_store_into_wal_mode(tmp_path: Any, capsys: Any) -> None:
    """A rollback-journal ledger stays one. The journal mode is the operator's choice."""
    path = tmp_path / "ledger.db"
    write_ledger(path, count=3)
    connection = sqlite3.connect(str(path))
    connection.execute("PRAGMA journal_mode = DELETE").fetchall()
    connection.close()

    code, _, err = run(["verify", "-b", sqlite_uri(path)], capsys)

    assert (code, journal_mode(path)) == (EXIT_OK, "delete"), err


# --------------------------------------------------------------------------------------
# C4 - export must not archive a ledger that does not verify
# --------------------------------------------------------------------------------------


def test_export_refuses_a_ledger_that_does_not_verify(tmp_path: Any, capsys: Any) -> None:
    """``checkpoint`` refuses and exits 1. ``export`` printed a merkle root and exited 0.

    An operator scripting `signledger export -o nightly.json` got no signal at all.
    """
    path = tmp_path / "ledger.db"
    bundle = tmp_path / "bundle.json"
    write_ledger(path, count=3)
    rewrite_payload(path, 1, '{"event":"created","index":999}')

    code, out, _ = run(
        ["export", "-b", sqlite_uri(path), "--bundle", str(bundle)], capsys
    )

    assert code == EXIT_INVALID
    assert "refusing to export" in out


def test_the_refused_export_writes_no_bundle(tmp_path: Any, capsys: Any) -> None:
    path = tmp_path / "ledger.db"
    bundle = tmp_path / "bundle.json"
    write_ledger(path, count=3)
    rewrite_payload(path, 1, '{"event":"created","index":999}')

    run(["export", "-b", sqlite_uri(path), "--bundle", str(bundle)], capsys)

    assert not bundle.exists()


def test_the_refusal_to_export_says_how_to_override_it(tmp_path: Any, capsys: Any) -> None:
    path = tmp_path / "ledger.db"
    write_ledger(path, count=3)
    rewrite_payload(path, 1, '{"event":"created","index":999}')

    _, out, _ = run(
        ["export", "-b", sqlite_uri(path), "--bundle", str(tmp_path / "b.json")], capsys
    )

    assert "--force" in out


def test_the_refused_export_reports_what_it_found_as_json(
    tmp_path: Any, capsys: Any
) -> None:
    path = tmp_path / "ledger.db"
    write_ledger(path, count=3)
    rewrite_payload(path, 1, '{"event":"created","index":999}')

    _, out, _ = run(
        ["export", "-b", sqlite_uri(path), "--bundle", str(tmp_path / "b.json"), "--json"],
        capsys,
    )

    report = json.loads(out)
    assert (report["command"], report["ok"], report["exit_code"]) == ("export", False, 1)
    assert [failure["kind"] for failure in report["failures"]] == ["bad_hash"]


def test_a_forced_export_of_a_broken_ledger_still_writes_the_bundle(
    tmp_path: Any, capsys: Any
) -> None:
    """Forensics needs the evidence out of the store. ``--force`` is how, as it is
    for ``checkpoint``."""
    path = tmp_path / "ledger.db"
    bundle = tmp_path / "bundle.json"
    write_ledger(path, count=3)
    rewrite_payload(path, 1, '{"event":"created","index":999}')

    code, _, err = run(
        ["export", "-b", sqlite_uri(path), "--bundle", str(bundle), "--force"], capsys
    )

    assert (code, bundle.exists()) == (EXIT_OK, True), err


# --------------------------------------------------------------------------------------
# C5 - the first five minutes
# --------------------------------------------------------------------------------------


def test_the_version_flag_reports_the_version(capsys: Any) -> None:
    """The first thing anyone types. It exited 2 with "unrecognized arguments"."""
    with pytest.raises(SystemExit) as caught:
        main(["--version"])

    assert caught.value.code == EXIT_OK
    assert __version__ in capsys.readouterr().out


def documented_sqlite_uris(text: str) -> List[str]:
    """Every ``sqlite:`` form a piece of user-facing text tells an operator to write."""
    return re.findall(r"sqlite:[^\s,]*", text)


def test_the_backend_help_only_names_sqlite_uris_the_parser_accepts(
    tmp_path: Any, capsys: Any, monkeypatch: Any
) -> None:
    """``--help`` advertised ``sqlite://<path>``, which the parser rejects as a host."""
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        main(["verify", "--help"])
    documented = documented_sqlite_uris(capsys.readouterr().out)
    assert documented, "the help no longer documents a sqlite URI at all"

    for form in documented:
        open_backend(form.replace("<path>", "audit.db")).close()

    assert (tmp_path / "audit.db").exists()


def test_the_schemeless_refusal_only_names_sqlite_uris_the_parser_accepts(
    tmp_path: Any, monkeypatch: Any
) -> None:
    """The same wrong form was in the error an operator sees when they omit the scheme."""
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ConfigurationError) as caught:
        open_backend("/var/lib/ledger.db")

    documented = documented_sqlite_uris(str(caught.value))
    assert documented, "the refusal no longer says what a sqlite URI looks like"
    for form in documented:
        open_backend(form.replace("<path>", "audit.db")).close()


def scribble_over_a_timestamp(path: Any) -> None:
    """Put a string in the BIGINT the timestamp is rebuilt from.

    SQLite stores it happily - column types are advisory - which is exactly how an
    operator ends up with a store the library cannot read.
    """
    unlock_store(path)
    connection = sqlite3.connect(str(path))
    try:
        connection.execute("UPDATE ledger_entries SET timestamp_us = 'not-a-number'")
        connection.commit()
    finally:
        connection.close()


def test_a_column_holding_the_wrong_type_is_an_operational_error(
    tmp_path: Any, capsys: Any
) -> None:
    """It escaped ``main`` as a raw ``TypeError`` traceback and exited 1.

    Exit 1 is the contract's "the ledger did not verify". A store the tool cannot read
    is exit 2, and a traceback is not an operator-facing message.
    """
    path = tmp_path / "ledger.db"
    write_ledger(path, count=2)
    scribble_over_a_timestamp(path)

    code, _, err = run(["verify", "-b", sqlite_uri(path)], capsys)

    assert code == EXIT_ERROR
    assert err.startswith("error: ")
    assert "Traceback" not in err


def test_that_operational_error_names_the_fault_without_quoting_the_row(
    tmp_path: Any, capsys: Any
) -> None:
    """The type name, never the driver's message: it can carry the row itself."""
    path = tmp_path / "ledger.db"
    write_ledger(path, count=2)
    scribble_over_a_timestamp(path)

    _, _, err = run(["verify", "-b", sqlite_uri(path)], capsys)

    assert "TypeError" in err
    assert "unsupported type for timedelta" not in err


def test_that_operational_error_is_reported_as_json_when_json_was_asked_for(
    tmp_path: Any, capsys: Any
) -> None:
    path = tmp_path / "ledger.db"
    write_ledger(path, count=2)
    scribble_over_a_timestamp(path)

    code, out, _ = run(["verify", "-b", sqlite_uri(path), "--json"], capsys)

    report = json.loads(out)
    assert (code, report["ok"], report["exit_code"]) == (EXIT_ERROR, False, EXIT_ERROR)


def test_a_failed_verification_does_not_say_the_entries_were_verified(
    tmp_path: Any, capsys: Any
) -> None:
    """"entries 3 verified" printed directly above "result FAILED" reads as reassurance."""
    path = tmp_path / "ledger.db"
    write_ledger(path, count=3)
    rewrite_payload(path, 1, '{"event":"created","index":999}')

    _, out, _ = run(["verify", "-b", sqlite_uri(path)], capsys)

    entries_line = next(line for line in out.splitlines() if line.startswith("entries"))
    assert "FAILED" in out
    assert "verified" not in entries_line


def test_a_passing_verification_still_says_the_entries_were_verified(
    tmp_path: Any, capsys: Any
) -> None:
    """Guards the line above: dropping the word entirely would pass it."""
    path = tmp_path / "ledger.db"
    write_ledger(path, count=3)

    _, out, _ = run(["verify", "-b", sqlite_uri(path)], capsys)

    entries_line = next(line for line in out.splitlines() if line.startswith("entries"))
    assert "3 verified" in entries_line


def test_a_posix_absolute_path_beginning_with_a_letter_and_colon_keeps_its_root(
    monkeypatch: Any,
) -> None:
    """``/c:/data/audit.db`` is an ordinary absolute path on POSIX, not a drive letter.

    The leading slash is dropped in front of a drive designator because
    ``sqlite:///C:/Users/me/audit.db`` names a file on the C: drive and SQLite needs it
    unrooted. On POSIX the identical string names a directory called ``c:`` at the
    filesystem root, and dropping the slash silently turns an absolute path into a
    relative one resolved against the working directory - so a verify would read, or
    create, some other file entirely.

    The two spellings are indistinguishable, so the platform has to decide.
    """
    monkeypatch.setattr(os, "name", "posix")

    assert _sqlite_path("sqlite:///c:/data/audit.db") == "/c:/data/audit.db"


def test_a_windows_drive_still_loses_the_slash_the_uri_added(monkeypatch: Any) -> None:
    """The other half of the same decision, asserted from any OS."""
    monkeypatch.setattr(os, "name", "nt")

    assert _sqlite_path("sqlite:///C:/Users/me/audit.db") == "C:/Users/me/audit.db"


def test_a_sqlite_path_containing_a_hash_names_the_file_that_was_typed() -> None:
    """``#`` and ``?`` end a URL, but a ``sqlite:`` URI carries a filename, not a URL.

    ``urlsplit`` reads everything after ``#`` as a fragment and everything after ``?`` as
    a query, so ``sqlite:audit#2026.db`` used to resolve to ``audit``. If a file happened
    to exist under that name, ``verify`` opened it, reported a different ledger entirely
    and exited 0 - a success about a store nobody asked about, which is the same failure
    as reading the wrong MongoDB database.

    There is no escape hatch either: the path is deliberately not unquoted, so percent-
    encoding ``%23`` names a file with a literal ``%23`` in it rather than a ``#``.
    """
    assert _sqlite_path("sqlite:reports/audit#2026.db") == "reports/audit#2026.db"
    assert _sqlite_path("sqlite:reports/audit?draft.db") == "reports/audit?draft.db"
    assert _sqlite_path("sqlite:///var/lib/audit#2026.db") == "/var/lib/audit#2026.db"


def test_the_shapes_a_sqlite_uri_already_supported_are_unchanged(
    monkeypatch: Any,
) -> None:
    """The fragment fix must not disturb any spelling that already worked."""
    monkeypatch.setattr(os, "name", "nt")

    assert _sqlite_path("sqlite:relative/path.db") == "relative/path.db"
    assert _sqlite_path("sqlite:/absolute/posix.db") == "/absolute/posix.db"
    assert _sqlite_path("sqlite:///absolute/posix.db") == "/absolute/posix.db"
    assert _sqlite_path("sqlite:C:/Users/me/audit.db") == "C:/Users/me/audit.db"
    assert _sqlite_path("sqlite://C:/Users/me/audit.db") == "C:/Users/me/audit.db"
    assert _sqlite_path(r"sqlite://C:\Users\me\audit.db") == r"C:\Users\me\audit.db"
    assert _sqlite_path(r"sqlite://\\server\share\audit.db") == r"\\server\share\audit.db"
    assert _sqlite_path("sqlite:////server/share/audit.db") == "//server/share/audit.db"
    assert _sqlite_path("sqlite://") == ":memory:"
    assert _sqlite_path("sqlite:") == ":memory:"


def test_a_real_host_is_still_refused_after_the_fragment_fix() -> None:
    with pytest.raises(ConfigurationError):
        _sqlite_path("sqlite://somehost/path.db")
    with pytest.raises(ConfigurationError):
        _sqlite_path("sqlite://backup:pw@fileserver/audit.db")
