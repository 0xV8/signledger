"""No exception message may leak a payload, a DSN or a credential (contract section 10).

This is a *security* property, not tidiness. An audit library's exceptions end up in
places its payloads were deliberately kept out of: application logs, Sentry, a 500 page,
a CI transcript, a support ticket. Redaction exists precisely so that the record which
was too sensitive to log stays too sensitive to log when it fails to be written.

The tests use one sentinel string per concern and assert it never surfaces. Every one of
them fails if a message starts interpolating the value it is complaining about.
"""

import json
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Callable

import pytest

from signledger import Entry, Ledger
from signledger.core.batch import BatchItem, BatchProcessor
from signledger.core.exceptions import SignLedgerError

#: Appears as a *value* in payloads. Must never be echoed back.
SECRET = "s3cr3t-payload-value-8f21ba"

#: A password embedded in a connection string.
DSN_PASSWORD = "pgpw-9d41c7-do-not-log"


def _message_of(call: Callable[[], Any]) -> str:
    """Run ``call``, which must raise, and return everything the exception can render.

    ``str(err)`` is not the whole story: a caller printing a traceback also sees
    ``repr(err)`` and ``err.args``. All three are checked, because redaction that only
    covers ``__str__`` is redaction an ordinary logger walks straight past.
    """
    with pytest.raises(BaseException) as caught:
        call()
    err = caught.value
    return " || ".join([str(err), repr(err), repr(getattr(err, "args", ()))])


# --------------------------------------------------------------------------------------
# entry payloads
# --------------------------------------------------------------------------------------


class TestPayloadNeverAppearsInAnError:
    """The values a caller hands to ``append`` are the ones that must not come back."""

    def test_a_non_dict_payload_does_not_echo_the_payload(self) -> None:
        """``ledger.append("...")`` is a one-call route to the leak.

        Raw ``pydantic.ValidationError`` renders ``input_value=...`` verbatim, so an
        untranslated pydantic error is a section 10 violation on the library's single
        most-used method.
        """
        with Ledger() as ledger:
            assert SECRET not in _message_of(lambda: ledger.append(SECRET))

    def test_a_non_dict_metadata_does_not_echo_the_metadata(self) -> None:
        with Ledger() as ledger:
            assert SECRET not in _message_of(lambda: ledger.append({"a": 1}, metadata=SECRET))

    def test_a_list_payload_does_not_echo_its_items(self) -> None:
        with Ledger() as ledger:
            assert SECRET not in _message_of(lambda: ledger.append([SECRET]))

    def test_a_non_json_native_value_does_not_echo_the_sibling_values(self) -> None:
        """The path is reported (contract section 2 requires it); the values are not."""
        with Ledger() as ledger:
            message = _message_of(
                lambda: ledger.append({"password": SECRET, "amount": Decimal("1.5")})
            )
        assert SECRET not in message
        assert "amount" in message  # the path is still actionable

    def test_an_oversized_int_does_not_echo_the_integer(self) -> None:
        with Ledger() as ledger:
            message = _message_of(lambda: ledger.append({"account": 2**60}))
        assert "1152921504606846976" not in message

    def test_a_nan_does_not_echo_the_neighbouring_secret(self) -> None:
        with Ledger() as ledger:
            message = _message_of(lambda: ledger.append({"token": SECRET, "score": float("nan")}))
        assert SECRET not in message

    def test_a_non_string_key_does_not_echo_the_value_behind_it(self) -> None:
        with Ledger() as ledger:
            message = _message_of(lambda: ledger.append({1: SECRET}))
        assert SECRET not in message

    def test_an_oversized_payload_does_not_echo_the_payload(self) -> None:
        """A megabyte of secret in a log line is the worst possible failure mode."""
        with Ledger() as ledger:
            message = _message_of(lambda: ledger.append({"blob": SECRET * 100_000}))
        assert SECRET not in message

    def test_a_too_deeply_nested_payload_does_not_echo_the_payload(self) -> None:
        payload: Any = SECRET
        for _ in range(200):
            payload = {"n": payload}
        with Ledger() as ledger:
            assert SECRET not in _message_of(lambda: ledger.append(payload))

    def test_an_empty_payload_error_names_the_field_only(self) -> None:
        with Ledger() as ledger:
            message = _message_of(lambda: ledger.append({}))
        assert "data" in message


class TestEntryConstructionIsRedacted:
    """The same guarantee at the layer every other component builds on."""

    def _kwargs(self, **overrides: Any) -> Any:
        base = {
            "format_version": 2,
            "ledger_id": str(uuid.uuid4()),
            "sequence": 0,
            "entry_id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc),
            "data": {"event": "ok"},
            "metadata": {},
            "previous_hash": "0" * 64,
        }
        base.update(overrides)
        return base

    @pytest.mark.parametrize(
        "field", ["ledger_id", "entry_id", "previous_hash", "hash", "signature", "key_id"]
    )
    def test_a_wrong_typed_field_does_not_echo_the_value(self, field: str) -> None:
        """Every one of these fields can carry attacker- or user-controlled text."""
        message = _message_of(lambda: Entry(**self._kwargs(**{field: [SECRET]})))
        assert SECRET not in message

    def test_a_wrong_typed_field_still_names_the_field(self) -> None:
        """Redaction must not cost the operator the ability to fix the bug."""
        message = _message_of(lambda: Entry(**self._kwargs(ledger_id=[SECRET])))
        assert "ledger_id" in message

    def test_a_bad_timestamp_does_not_echo_the_value(self) -> None:
        message = _message_of(lambda: Entry(**self._kwargs(timestamp=SECRET)))
        assert SECRET not in message

    def test_a_bad_sequence_does_not_echo_the_value(self) -> None:
        message = _message_of(lambda: Entry(**self._kwargs(sequence=SECRET)))
        assert SECRET not in message

    def test_from_dict_does_not_echo_the_payload(self) -> None:
        message = _message_of(lambda: Entry.from_dict({**self._kwargs(), "data": SECRET}))
        assert SECRET not in message

    def test_model_validate_does_not_echo_the_payload(self) -> None:
        """``model_validate`` is the other public way in, so it is checked separately.

        It happens to be redacted for free because pydantic routes it through
        ``__init__``. That is an implementation detail of pydantic, not a promise - so it
        is pinned here rather than assumed. If a future pydantic stops delegating, this
        test fails and says so, which is what an unreachable defensive branch could never
        do.
        """
        message = _message_of(lambda: Entry.model_validate({**self._kwargs(), "data": SECRET}))
        assert SECRET not in message

    def test_model_validate_still_builds_a_valid_entry(self) -> None:
        """Guards the test above: redaction must not have broken the success path."""
        entry = Entry.model_validate(self._kwargs())
        assert entry.sequence == 0
        assert entry.data == {"event": "ok"}

    def test_the_redacted_message_names_every_bad_field(self) -> None:
        """One report for the whole entry, not just the first thing pydantic tripped on."""
        message = _message_of(
            lambda: Entry(**self._kwargs(ledger_id=[SECRET], entry_id=[SECRET]))
        )
        assert "ledger_id" in message
        assert "entry_id" in message
        assert SECRET not in message

    def test_every_construction_error_is_a_signledger_error(self) -> None:
        """One ``except SignLedgerError`` must contain the whole surface.

        A raw ``pydantic.ValidationError`` escaping here is both an unredacted message
        and a hole in the documented exception hierarchy.
        """
        with pytest.raises(SignLedgerError):
            Entry(**self._kwargs(data=SECRET))


class TestBatchErrorsAreRedacted:
    """Dead letters are written to logs by definition - that is what they are for."""

    def test_a_dead_letter_message_does_not_echo_the_payload(self) -> None:
        with Ledger() as ledger:
            result = BatchProcessor(ledger, atomic=False).append_batch(
                [{"event": "fine"}, SECRET]
            )
        assert result.failed == 1
        assert all(SECRET not in letter.message for letter in result.dead_letters)

    def test_a_dead_letter_error_repr_does_not_echo_the_payload(self) -> None:
        with Ledger() as ledger:
            result = BatchProcessor(ledger, atomic=False).append_batch([BatchItem(data=SECRET)])
        assert all(SECRET not in repr(letter.error) for letter in result.dead_letters)

    def test_an_abandoned_atomic_batch_does_not_echo_the_payload(self) -> None:
        """An atomic batch reports the refusal rather than raising it.

        The whole batch is abandoned in the prepare phase, so nothing is written - but
        the dead letter describing why still travels to a log, and must be clean.
        """
        with Ledger() as ledger:
            processor = BatchProcessor(ledger, atomic=True)
            result = processor.append_batch([{"a": 1}, SECRET])
            assert ledger.count() == 0

        assert result.ok is False
        assert result.committed == 0
        rendered = json.dumps(
            [[letter.error, letter.message] for letter in result.dead_letters]
        )
        assert SECRET not in rendered


# --------------------------------------------------------------------------------------
# connection strings
# --------------------------------------------------------------------------------------


class TestConnectionStringsAreNeverInAMessage:
    """A DSN carries a password. It must not travel with the error it caused."""

    def test_a_sqlite_open_failure_does_not_echo_the_path(self, tmp_path: Any) -> None:
        from signledger.backends.sqlite import SQLiteBackend

        unwritable = tmp_path / "no-such-directory" / f"{DSN_PASSWORD}.db"
        message = _message_of(lambda: SQLiteBackend(str(unwritable)))
        assert DSN_PASSWORD not in message

    def test_a_postgres_connect_failure_does_not_echo_the_dsn(self) -> None:
        pytest.importorskip("psycopg", reason="psycopg (v3) is not installed")
        from signledger.backends.postgresql import PostgresBackend

        dsn = f"postgresql://postgres:{DSN_PASSWORD}@127.0.0.1:1/nope"
        message = _message_of(lambda: PostgresBackend(dsn, connect_timeout=1))
        assert DSN_PASSWORD not in message

    def test_a_mongo_connect_failure_does_not_echo_the_uri(self) -> None:
        pytest.importorskip("pymongo", reason="pymongo is not installed")
        from signledger.backends.mongodb import MongoBackend

        uri = f"mongodb://user:{DSN_PASSWORD}@127.0.0.1:1/?connectTimeoutMS=200"
        message = _message_of(
            lambda: MongoBackend(uri, database="x", server_selection_timeout_ms=200)
        )
        assert DSN_PASSWORD not in message

    def test_the_cli_does_not_echo_a_dsn_it_cannot_open(self, capsys: Any) -> None:
        from signledger.cli import main

        dsn = f"postgresql://postgres:{DSN_PASSWORD}@127.0.0.1:1/nope"
        main(["verify", "-b", dsn])
        captured = capsys.readouterr()
        assert DSN_PASSWORD not in captured.out + captured.err


# --------------------------------------------------------------------------------------
# the whole exception surface, swept
# --------------------------------------------------------------------------------------


class TestNoErrorAnywhereEchoesThePayload:
    """A sweep, so a *new* leak in a *new* code path is caught without a new test."""

    @pytest.mark.parametrize(
        "payload",
        [
            SECRET,
            [SECRET],
            {"k": Decimal(1), "secret": SECRET},
            {"k": float("inf"), "secret": SECRET},
            {"k": {1, 2}, "secret": SECRET},
            {"k": b"bytes", "secret": SECRET},
            {"k": datetime.now(timezone.utc), "secret": SECRET},
            {"k": 2**70, "secret": SECRET},
            {SECRET: 1},
            {},
        ],
    )
    def test_no_shipped_backend_leaks_the_payload_on_a_refused_append(
        self, payload: Any
    ) -> None:
        """The same hostile payload, through the ledger, on the in-process backends."""
        with Ledger() as ledger:
            try:
                ledger.append(payload)
            except BaseException as err:  # noqa: BLE001 - the point is to inspect anything
                rendered = f"{err!s} || {err!r} || {getattr(err, 'args', ())!r}"
                assert SECRET not in rendered

    def test_a_report_rendered_as_json_never_carries_the_payload(self) -> None:
        """Verification failures are designed to be shipped somewhere central."""
        from signledger.backends.memory import MemoryBackend

        backend = MemoryBackend()
        with Ledger(backend) as ledger:
            entry = ledger.append({"secret": SECRET})
            stored = backend._entries[0]  # noqa: SLF001 - simulating a store-level rewrite
            backend._entries[0] = stored.model_copy(update={"hash": "0" * 64})
            report = ledger.check_integrity()

        assert report.ok is False
        rendered = json.dumps([failure.message for failure in report.failures])
        assert SECRET not in rendered
        assert entry.entry_id is not None
