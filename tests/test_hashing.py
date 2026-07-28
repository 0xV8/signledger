"""Tests for the hash chain (contract sections 3, 12)."""

import hashlib
import json
import os
import pathlib
import subprocess
import sys
from datetime import datetime, timedelta, timezone

import pytest

from signledger.core.entry import Entry
from signledger.core.exceptions import (
    CanonicalizationError,
    ConfigurationError,
    ValidationError,
)
from signledger.crypto.canonical import canonicalize, entry_preimage
from signledger.crypto.hashing import SUPPORTED_ALGORITHMS, HashChain, genesis_hash

TS = datetime(2024, 3, 1, 12, 0, 0, tzinfo=timezone.utc)


def make_entry(**overrides):
    fields = {
        "ledger_id": "11111111-1111-4111-8111-111111111111",
        "sequence": 7,
        "entry_id": "22222222-2222-4222-8222-222222222222",
        "timestamp": TS,
        "data": {"event": "created"},
        "metadata": {"actor": "svc"},
        "previous_hash": "00" * 32,
    }
    fields.update(overrides)
    return Entry(**fields)


def test_calculate_hash_is_the_digest_of_the_entry_preimage():
    entry = make_entry()
    expected = hashlib.sha256(entry_preimage(entry)).hexdigest()
    assert HashChain().calculate_hash(entry) == expected


# --------------------------------------------------------------------------------------
# Algorithms
# --------------------------------------------------------------------------------------


def test_default_algorithm_is_sha256():
    assert HashChain().algorithm == "sha256"


@pytest.mark.parametrize(
    ("algorithm", "hex_length"),
    [("sha256", 64), ("sha512", 128), ("sha3_256", 64), ("sha3_512", 128)],
)
def test_supported_algorithm_produces_its_own_digest_length(algorithm, hex_length):
    digest = HashChain(algorithm).calculate_hash(make_entry())
    assert len(digest) == hex_length
    assert int(digest, 16) >= 0  # valid hex


@pytest.mark.parametrize("algorithm", ["sha256", "sha512", "sha3_256", "sha3_512"])
def test_each_algorithm_matches_hashlib_over_the_preimage(algorithm):
    entry = make_entry()
    expected = hashlib.new(algorithm, entry_preimage(entry)).hexdigest()
    assert HashChain(algorithm).calculate_hash(entry) == expected


def test_supported_algorithms_is_exactly_the_documented_set():
    assert set(SUPPORTED_ALGORITHMS) == {"sha256", "sha512", "sha3_256", "sha3_512"}


def test_unknown_algorithm_is_rejected_at_construction():
    with pytest.raises(ConfigurationError, match="md5"):
        HashChain("md5")


def test_algorithm_rejection_does_not_leak_a_partially_built_object():
    with pytest.raises(ConfigurationError):
        HashChain("")


def test_two_algorithms_disagree_on_the_same_entry():
    entry = make_entry()
    assert HashChain("sha256").calculate_hash(entry) != HashChain("sha3_256").calculate_hash(entry)


# --------------------------------------------------------------------------------------
# What the digest covers
# --------------------------------------------------------------------------------------


def test_hash_ignores_the_hash_field_itself():
    entry = make_entry()
    assert HashChain().calculate_hash(entry) == HashChain().calculate_hash(
        entry.model_copy(update={"hash": "ff" * 32})
    )


def test_hash_ignores_the_signature_field():
    entry = make_entry()
    assert HashChain().calculate_hash(entry) == HashChain().calculate_hash(
        entry.model_copy(update={"signature": "c2ln"})
    )


def test_changing_the_payload_changes_the_hash():
    chain = HashChain()
    assert chain.calculate_hash(make_entry()) != chain.calculate_hash(
        make_entry(data={"event": "deleted"})
    )


def test_changing_the_key_id_changes_the_hash():
    """key_id is in the preimage so an attacker cannot re-point an entry at another key."""
    chain = HashChain()
    assert chain.calculate_hash(make_entry(key_id="kid-1")) != chain.calculate_hash(
        make_entry(key_id="kid-2")
    )


def test_hash_is_deterministic_across_calls():
    chain = HashChain()
    assert chain.calculate_hash(make_entry()) == chain.calculate_hash(make_entry())


# --------------------------------------------------------------------------------------
# format_version dispatch (contract section 12)
# --------------------------------------------------------------------------------------


def test_format_version_2_is_supported():
    assert HashChain().calculate_hash(make_entry(format_version=2))


def test_unsupported_format_version_is_rejected():
    """A v3 encoder can be slotted in later; until then, refuse rather than guess."""
    entry = make_entry().model_copy(update={"format_version": 3})
    with pytest.raises(ValidationError, match="format_version 3"):
        HashChain().calculate_hash(entry)


def test_format_version_1_is_rejected_until_a_v1_encoder_is_restored():
    entry = make_entry().model_copy(update={"format_version": 1})
    with pytest.raises(ValidationError, match="format_version 1"):
        HashChain().calculate_hash(entry)


# --------------------------------------------------------------------------------------
# verify_hash
# --------------------------------------------------------------------------------------


def sealed(chain, **overrides):
    draft = make_entry(**overrides)
    return draft.model_copy(update={"hash": chain.calculate_hash(draft)})


def test_verify_hash_accepts_a_correctly_sealed_entry():
    chain = HashChain()
    assert chain.verify_hash(sealed(chain)) is True


def test_verify_hash_rejects_a_tampered_payload():
    chain = HashChain()
    entry = sealed(chain)
    assert chain.verify_hash(entry.model_copy(update={"data": {"event": "deleted"}})) is False


def test_verify_hash_rejects_an_unsealed_draft():
    assert HashChain().verify_hash(make_entry()) is False


def test_verify_hash_rejects_a_non_ascii_hash_without_raising():
    """hmac.compare_digest raises TypeError on non-ASCII str; a verifier must not crash."""
    chain = HashChain()
    assert chain.verify_hash(sealed(chain).model_copy(update={"hash": "é" * 64})) is False


def test_verify_hash_rejects_a_hash_of_the_wrong_length():
    chain = HashChain()
    assert chain.verify_hash(sealed(chain).model_copy(update={"hash": "ab"})) is False


# --------------------------------------------------------------------------------------
# genesis_hash
# --------------------------------------------------------------------------------------


def test_genesis_hash_is_the_digest_of_a_domain_separated_preimage():
    expected = hashlib.sha256(
        canonicalize(
            {
                "created_at": TS,
                "format_version": 2,
                "ledger_id": "led-1",
                "type": "signledger/genesis",
            }
        )
    ).hexdigest()
    assert genesis_hash("led-1", TS, 2) == expected


def test_genesis_hash_carries_a_type_tag_no_entry_preimage_has():
    """Domain separation: no Entry can be built whose preimage is a genesis preimage."""
    entry_fields = set(json.loads(entry_preimage(make_entry()).decode()))
    assert "type" not in entry_fields


def test_genesis_hash_is_deterministic():
    assert genesis_hash("led-1", TS) == genesis_hash("led-1", TS)


def test_genesis_hash_defaults_to_format_version_2():
    assert genesis_hash("led-1", TS) == genesis_hash("led-1", TS, 2)


def test_genesis_hash_changes_with_the_ledger_id():
    assert genesis_hash("led-1", TS) != genesis_hash("led-2", TS)


def test_genesis_hash_changes_with_the_creation_time():
    later = datetime(2024, 3, 1, 12, 0, 1, tzinfo=timezone.utc)
    assert genesis_hash("led-1", TS) != genesis_hash("led-1", later)


def test_genesis_hash_changes_with_the_format_version():
    assert genesis_hash("led-1", TS, 2) != genesis_hash("led-1", TS, 3)


def test_genesis_hash_rejects_a_naive_creation_time():
    with pytest.raises(CanonicalizationError, match="naive datetime"):
        genesis_hash("led-1", datetime(2024, 3, 1, 12, 0, 0))


def test_genesis_hash_rejects_an_empty_ledger_id():
    with pytest.raises(ValidationError, match="ledger_id"):
        genesis_hash("", TS)


def test_genesis_hash_honours_the_requested_algorithm():
    assert len(genesis_hash("led-1", TS, algorithm="sha512")) == 128


def test_genesis_hash_rejects_an_unknown_algorithm():
    with pytest.raises(ConfigurationError, match="md5"):
        genesis_hash("led-1", TS, algorithm="md5")


def test_genesis_hash_of_the_same_instant_in_another_timezone_is_identical():
    kolkata = TS.astimezone(timezone(timedelta(hours=5, minutes=30)))
    assert genesis_hash("led-1", kolkata) == genesis_hash("led-1", TS)


# --------------------------------------------------------------------------------------
# Package re-exports
# --------------------------------------------------------------------------------------


def test_crypto_package_re_exports_the_wave_one_surface():
    import signledger.crypto as crypto

    assert crypto.canonicalize is canonicalize
    assert crypto.entry_preimage is entry_preimage
    assert crypto.HashChain is HashChain
    assert crypto.genesis_hash is genesis_hash


def test_crypto_package_re_exports_the_slcs1_limits():
    import signledger.crypto as crypto

    assert crypto.MAX_ENTRY_BYTES == 1_048_576
    assert crypto.MAX_NESTING_DEPTH == 64


def test_crypto_package_exports_everything_it_advertises():
    import signledger.crypto as crypto

    for name in crypto.__all__:
        assert hasattr(crypto, name), name


# --------------------------------------------------------------------------------------
# Import graph
# --------------------------------------------------------------------------------------


# Each order runs in a fresh interpreter: once one order has succeeded the modules are
# cached, and every later order passes trivially.
#
# The probe tolerates a top-level ``signledger/__init__.py`` that does not import - the
# rebuild has not restored it yet, and this check is about the *submodule* graph. Once
# that file is healthy the fallback simply never fires.
IMPORT_PROBE = """
import importlib, importlib.machinery, os, sys, types

root, target = sys.argv[1], sys.argv[2]
try:
    importlib.import_module("signledger")
except Exception:
    package = types.ModuleType("signledger")
    package.__path__ = [os.path.join(root, "signledger")]
    spec = importlib.machinery.ModuleSpec("signledger", None, is_package=True)
    spec.submodule_search_locations = package.__path__
    package.__spec__ = spec
    sys.modules["signledger"] = package
importlib.import_module(target)
"""


@pytest.mark.parametrize(
    "first_import",
    [
        "signledger.core.entry",
        "signledger.core.exceptions",
        "signledger.crypto.canonical",
        "signledger.crypto.hashing",
        "signledger.crypto",
        "signledger.core",
        "signledger.anchor",
        "signledger.anchor.checkpoint",
    ],
)
def test_any_module_can_be_imported_first(first_import):
    """core.entry needs the canonicalizer and crypto describes entries; neither may win.

    v1 shipped a commit titled "Fix circular imports"; this is the test that makes the
    fix stick.
    """
    root = str(pathlib.Path(__file__).resolve().parent.parent)
    result = subprocess.run(
        [sys.executable, "-c", IMPORT_PROBE, root, first_import],
        capture_output=True,
        text=True,
        cwd=root,
        env={**os.environ, "PYTHONPATH": root},
        check=False,
    )
    assert result.returncode == 0, result.stderr
