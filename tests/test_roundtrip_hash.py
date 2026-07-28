"""``store -> load -> re-hash`` is byte-identical, on every backend.

This is contract section 4's round-trip guarantee, and it is the single most load-bearing
property of a storage layer for a tamper-evident log. If a payload does not come back
byte-for-byte, the entry re-hashes to something else, verification reports the ledger as
tampered, and the operator cannot tell a serialization bug from an actual attack. v1
failed this on empty metadata, on non-UTC sessions and on microsecond timestamps.

Every case asserts the strongest available statement: the *preimage bytes* match, not
merely the hex digest, so a test failure points at the encoding rather than at the hash.
"""

from datetime import datetime, timedelta, timezone

import pytest

from signledger.backends.base import build_next_entry
from signledger.crypto.canonical import entry_preimage
from signledger.crypto.hashing import HashChain

UTC = timezone.utc
LEDGER_ID = "roundtrip-9d0c2f61-7a3e-4c5b-8d21-6e4f0a9b3c7d"
CREATED_AT = datetime(2024, 1, 1, tzinfo=UTC)


def nested(depth: int):
    """A dict nested ``depth`` levels deep, well inside ``MAX_NESTING_DEPTH``."""
    node = {"leaf": "bottom"}
    for level in reversed(range(depth)):
        node = {f"level_{level}": node}
    return node


PAYLOADS = {
    "ascii": {"event": "user.login", "actor": "svc-api", "ok": True},
    "emoji": {"note": "shipped 🚀🎉", "flag": "🏳️‍🌈", "family": "👨‍👩‍👧‍👦"},
    "cjk": {"名前": "山田太郎", "説明": "監査ログの記録", "中文": "审计日志"},
    "rtl": {"اسم": "محمد", "עברית": "ביקורת", "mixed": "abc عربي 123"},
    "combining": {"cafe": "cafe\u0301", "precomposed": "café", "hangul": "한국어"},
    "non_bmp": {"math": "𝔘𝔫𝔦𝔠𝔬𝔡𝔢", "ancient": "𐌰𐌱𐌲", "emoji_key_🔑": "value"},
    "utf16_sort_keys": {"\ue000": "private-use", "\U0001f600": "grinning", "z": "ascii-z"},
    "control_chars": {"tab": "a\tb", "newline": "a\nb", "quote": 'he said "hi"',
                      "backslash": "C:\\path\\to", "null_ish": "a\u0000b", "cr": "a\rb"},
    "whitespace": {"empty": "", "spaces": "   ", "unicode_space": "a\u00a0b"},
    "numbers": {"zero": 0, "neg_zero_float": -0.0, "one_float": 1.0, "one_int": 1,
                "max_safe": 2**53, "min_safe": -(2**53), "tiny": 1e-7, "huge": 1e20,
                "pi": 3.141592653589793, "neg": -42},
    "json_scalars": {"nothing": None, "yes": True, "no": False, "empty_list": [],
                     "empty_dict": {}, "list_of_dicts": [{"a": 1}, {"b": 2}]},
    "deeply_nested": nested(55),
    "wide_list": {"items": list(range(200))},
    "long_string": {"blob": "x" * 5000},
    "mixed_structure": {
        "actor": {"id": "u-1", "roles": ["admin", "auditor"], "meta": {"tz": "UTC"}},
        "changes": [{"field": "email", "from": None, "to": "a@b.c"}],
        "amount": "12.34",
    },
}

TIMESTAMPS = {
    "whole_second": datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC),
    "one_microsecond": datetime(2024, 6, 1, 12, 0, 0, 1, tzinfo=UTC),
    "max_microseconds": datetime(2024, 6, 1, 12, 0, 0, 999999, tzinfo=UTC),
    "half_second": datetime(2024, 6, 1, 12, 0, 0, 500000, tzinfo=UTC),
    "pre_epoch": datetime(1955, 11, 5, 6, 15, 0, 123456, tzinfo=UTC),
    # 2255 is the ceiling, not an arbitrary choice - see TestTimestampRangeLimit.
    "far_future": datetime(2200, 1, 1, 0, 0, 0, 999999, tzinfo=UTC),
    "far_past": datetime(1700, 3, 4, 5, 6, 7, 89, tzinfo=UTC),
    "offset_east": datetime(
        2024, 6, 1, 17, 30, 0, 654321, tzinfo=timezone(timedelta(hours=5, minutes=30))
    ),
    "offset_west": datetime(2024, 6, 1, 5, 0, 0, 42, tzinfo=timezone(timedelta(hours=-7))),
}


def store(backend, data, **kwargs):
    return backend.append_atomic(
        lambda tip: build_next_entry(
            tip, ledger_id=LEDGER_ID, created_at=CREATED_AT, data=data, **kwargs
        )
    )


def assert_round_trip(stored, loaded):
    """The strongest statement available: identical preimage bytes, then digest."""
    assert loaded is not None
    assert entry_preimage(loaded) == entry_preimage(stored)
    assert loaded.hash == stored.hash
    assert HashChain().verify_hash(loaded)
    assert loaded == stored


class TestPayloadRoundTrip:
    @pytest.mark.parametrize("name", sorted(PAYLOADS))
    def test_a_payload_reloads_with_an_identical_preimage(self, backend, name):
        stored = store(backend, PAYLOADS[name])

        assert_round_trip(stored, backend.get_by_sequence(0))

    @pytest.mark.parametrize("name", sorted(PAYLOADS))
    def test_a_payload_reloads_identically_when_fetched_by_id(self, backend, name):
        stored = store(backend, PAYLOADS[name])

        assert_round_trip(stored, backend.get_entry(stored.entry_id))

    @pytest.mark.parametrize("name", sorted(PAYLOADS))
    def test_a_payload_carried_in_metadata_also_survives(self, backend, name):
        stored = store(backend, {"event": "x"}, metadata=PAYLOADS[name])

        loaded = backend.get_by_sequence(0)

        assert loaded.metadata == PAYLOADS[name]
        assert_round_trip(stored, loaded)


class TestEmptyMetadataRoundTrip:
    """``metadata`` is never NULL on disk. This one line broke all of v1."""

    def test_an_entry_without_metadata_reloads_and_re_hashes(self, backend):
        stored = store(backend, {"event": "no-metadata"})

        assert_round_trip(stored, backend.get_by_sequence(0))

    def test_an_entry_with_an_explicitly_empty_metadata_dict_reloads(self, backend):
        stored = store(backend, {"event": "empty-metadata"}, metadata={})

        loaded = backend.get_by_sequence(0)

        assert loaded.metadata == {}
        assert_round_trip(stored, loaded)

    def test_the_reloaded_metadata_is_a_dict_and_never_none(self, backend):
        store(backend, {"event": "x"})

        assert isinstance(backend.get_by_sequence(0).metadata, dict)

    def test_a_whole_ledger_of_metadata_free_entries_re_hashes(self, backend):
        stored = [store(backend, {"index": index}) for index in range(10)]

        for original, loaded in zip(stored, backend.iter_entries()):
            assert_round_trip(original, loaded)


class TestTimestampRoundTrip:
    @pytest.mark.parametrize("name", sorted(TIMESTAMPS))
    def test_a_timestamp_reloads_to_the_same_instant(self, backend, name):
        stored = store(backend, {"event": "clock"}, timestamp=TIMESTAMPS[name])

        assert_round_trip(stored, backend.get_by_sequence(0))

    @pytest.mark.parametrize("name", sorted(TIMESTAMPS))
    def test_a_reloaded_timestamp_is_utc(self, backend, name):
        """v1 embedded the session offset in the digest, so a non-UTC Postgres session
        re-hashed every row differently and reported the whole log as tampered."""
        store(backend, {"event": "clock"}, timestamp=TIMESTAMPS[name])

        assert backend.get_by_sequence(0).timestamp.utcoffset() == timedelta(0)

    def test_microsecond_differences_produce_different_hashes(self, backend):
        """Microseconds must reach the digest, or truncation is undetectable."""
        at_one = datetime(2024, 6, 1, 12, 0, 0, 1, tzinfo=UTC)
        at_two = datetime(2024, 6, 1, 12, 0, 0, 2, tzinfo=UTC)
        first = store(backend, {"event": "a"}, timestamp=at_one)
        second = store(backend, {"event": "a"}, timestamp=at_two)

        assert first.hash != second.hash

    def test_two_entries_sharing_a_timestamp_both_re_hash(self, backend):
        moment = datetime(2024, 6, 1, 12, 0, 0, 123456, tzinfo=UTC)
        stored = [store(backend, {"index": i}, timestamp=moment) for i in range(4)]

        for original, loaded in zip(stored, backend.iter_entries()):
            assert_round_trip(original, loaded)


class TestTimestampRangeLimit:
    """SLCS/1 encodes a datetime as integer microseconds (rule 3) and rejects integers
    outside +/-2**53 (rule 6). Together those cap the representable instant at
    2255-06-05, and floor it at 1684-07-28.

    That is a documented consequence rather than a defect, but it is undiscoverable from
    the contract and the resulting message tells the caller to "convert to str", which a
    ``timestamp`` field cannot do. Pinned here so the boundary cannot move silently.
    """

    LAST_REPRESENTABLE = datetime(2255, 6, 5, 23, 47, 34, 740992, tzinfo=UTC)
    FIRST_REPRESENTABLE = datetime(1684, 7, 28, 0, 12, 25, 259008, tzinfo=UTC)

    def test_the_last_representable_instant_round_trips(self, backend):
        stored = store(backend, {"event": "edge"}, timestamp=self.LAST_REPRESENTABLE)

        assert_round_trip(stored, backend.get_by_sequence(0))

    def test_the_first_representable_instant_round_trips(self, backend):
        stored = store(backend, {"event": "edge"}, timestamp=self.FIRST_REPRESENTABLE)

        assert_round_trip(stored, backend.get_by_sequence(0))

    def test_an_instant_past_the_ceiling_is_refused_at_construction(self, backend):
        from signledger.core.exceptions import CanonicalizationError

        with pytest.raises(CanonicalizationError):
            store(
                backend,
                {"event": "too-late"},
                timestamp=self.LAST_REPRESENTABLE + timedelta(microseconds=1),
            )

    def test_nothing_is_written_when_the_timestamp_is_refused(self, backend):
        from signledger.core.exceptions import CanonicalizationError

        with pytest.raises(CanonicalizationError):
            store(backend, {"event": "too-late"}, timestamp=datetime(2999, 1, 1, tzinfo=UTC))

        assert backend.count() == 0


class TestSignatureFieldRoundTrip:
    def test_the_key_identity_fields_survive_and_stay_in_the_preimage(self, backend):
        stored = store(
            backend, {"event": "signed"}, key_id="kid-abc", signature_algorithm="ed25519"
        )

        assert_round_trip(stored, backend.get_by_sequence(0))

    def test_a_signature_survives_the_round_trip(self, backend, signing_keypair):
        chain = HashChain()

        def build(tip):
            entry = build_next_entry(
                tip,
                ledger_id=LEDGER_ID,
                created_at=CREATED_AT,
                data={"event": "signed"},
                chain=chain,
                key_id=signing_keypair.key_id,
                signature_algorithm=signing_keypair.algorithm,
            )
            payload = (
                b"signledger/v2\x00" + LEDGER_ID.encode() + b"\x00"
                + bytes.fromhex(entry.hash)
            )
            return entry.model_copy(update={"signature": signing_keypair.sign(payload)})

        stored = backend.append_atomic(build)
        loaded = backend.get_by_sequence(0)

        assert loaded.signature == stored.signature
        assert_round_trip(stored, loaded)

    def test_an_unsigned_entry_reloads_with_null_signature_fields(self, backend):
        store(backend, {"event": "unsigned"})
        loaded = backend.get_by_sequence(0)

        assert loaded.signature is None
        assert loaded.key_id is None
        assert loaded.signature_algorithm is None


class TestChainRoundTrip:
    def test_a_reloaded_chain_still_links_end_to_end(self, backend):
        for index, name in enumerate(sorted(PAYLOADS)):
            store(backend, PAYLOADS[name], timestamp=CREATED_AT + timedelta(seconds=index))

        loaded = list(backend.iter_entries())

        assert len(loaded) == len(PAYLOADS)
        for previous, current in zip(loaded, loaded[1:]):
            assert current.previous_hash == previous.hash

    def test_every_reloaded_entry_verifies_against_its_own_digest(self, backend):
        chain = HashChain()
        for index, name in enumerate(sorted(PAYLOADS)):
            store(backend, PAYLOADS[name], timestamp=CREATED_AT + timedelta(seconds=index))

        assert all(chain.verify_hash(entry) for entry in backend.iter_entries())

    def test_a_filtered_page_re_hashes_exactly_like_a_full_scan(self, backend):
        for index in range(10):
            store(backend, {"index": index}, timestamp=CREATED_AT + timedelta(seconds=index))

        full = list(backend.iter_entries())
        page = list(backend.iter_entries(start_seq=4, limit=3))

        assert [entry_preimage(e) for e in page] == [entry_preimage(e) for e in full[4:7]]


class TestSQLitePersistenceRoundTrip:
    """Reopening the file is the only way to prove the bytes on disk are right."""

    def test_entries_re_hash_after_the_database_is_closed_and_reopened(self, tmp_path):
        from signledger.backends.sqlite import SQLiteBackend

        path = str(tmp_path / "roundtrip.db")
        writer = SQLiteBackend(path)
        stored = [
            store(writer, PAYLOADS[name], timestamp=CREATED_AT + timedelta(seconds=index))
            for index, name in enumerate(sorted(PAYLOADS))
        ]
        writer.close()

        reader = SQLiteBackend(path)
        try:
            for original, loaded in zip(stored, reader.iter_entries()):
                assert_round_trip(original, loaded)
        finally:
            reader.close()

    def test_an_entry_without_metadata_re_hashes_after_a_reopen(self, tmp_path):
        from signledger.backends.sqlite import SQLiteBackend

        path = str(tmp_path / "meta.db")
        writer = SQLiteBackend(path)
        stored = store(writer, {"event": "no-metadata"})
        writer.close()

        reader = SQLiteBackend(path)
        try:
            assert_round_trip(stored, reader.get_by_sequence(0))
        finally:
            reader.close()

    def test_the_metadata_column_is_never_null_on_disk(self, tmp_path):
        import sqlite3

        from signledger.backends.sqlite import SQLiteBackend

        path = str(tmp_path / "nulls.db")
        backend = SQLiteBackend(path)
        store(backend, {"event": "x"})
        table = backend.entries_table
        backend.close()

        raw = sqlite3.connect(path)
        try:
            nulls = raw.execute(
                f"SELECT COUNT(*) FROM {table} WHERE metadata IS NULL OR data IS NULL"
            ).fetchone()[0]
        finally:
            raw.close()

        assert nulls == 0
