"""Tests for ``signledger.crypto.keys`` (architecture contract, section 6)."""

import pathlib

import pytest

from signledger.core.exceptions import ConfigurationError
from signledger.crypto.keys import FileKeyResolver, InMemoryKeyResolver, KeyResolver
from signledger.crypto.signatures import (
    ECDSASigner,
    Ed25519Signer,
    RSASigner,
    signing_payload,
    verify,
)

# ---------------------------------------------------------------------------------
# InMemoryKeyResolver
# ---------------------------------------------------------------------------------


def test_in_memory_resolver_returns_the_key_registered_under_its_own_key_id() -> None:
    signer = Ed25519Signer.generate()
    resolver = InMemoryKeyResolver()
    resolver.add(signer.public_key)

    assert resolver.resolve(signer.key_id) is not None


def test_in_memory_resolver_add_returns_the_derived_key_id() -> None:
    signer = Ed25519Signer.generate()
    resolver = InMemoryKeyResolver()

    assert resolver.add(signer.public_key) == signer.key_id


def test_in_memory_resolver_returns_none_for_an_unknown_key_id() -> None:
    resolver = InMemoryKeyResolver([Ed25519Signer.generate().public_key])

    assert resolver.resolve("00" * 32) is None


def test_in_memory_resolver_accepts_keys_at_construction() -> None:
    signer = Ed25519Signer.generate()

    resolver = InMemoryKeyResolver([signer.public_key])

    assert resolver.resolve(signer.key_id) is not None


def test_a_resolved_key_actually_verifies_that_signers_signature() -> None:
    signer = ECDSASigner.generate(curve="p384")
    resolver = InMemoryKeyResolver([signer.public_key])
    payload = signing_payload("ledger-a", "aa" * 32)

    resolved = resolver.resolve(signer.key_id)

    assert resolved is not None
    assert verify(payload, signer.sign(payload), resolved) is True


def test_in_memory_resolver_can_hold_several_keys_of_different_types() -> None:
    signers = [
        Ed25519Signer.generate(),
        RSASigner.generate(key_size=2048),
        ECDSASigner.generate(curve="p256"),
    ]
    resolver = InMemoryKeyResolver(s.public_key for s in signers)

    assert all(resolver.resolve(s.key_id) is not None for s in signers)


def test_in_memory_resolver_reports_how_many_keys_it_holds() -> None:
    resolver = InMemoryKeyResolver([Ed25519Signer.generate().public_key])

    assert len(resolver) == 1


def test_in_memory_resolver_registering_the_same_key_twice_is_idempotent() -> None:
    signer = Ed25519Signer.generate()
    resolver = InMemoryKeyResolver()
    resolver.add(signer.public_key)
    resolver.add(signer.public_key)

    assert len(resolver) == 1


def test_in_memory_resolver_exposes_its_key_ids() -> None:
    signer = Ed25519Signer.generate()
    resolver = InMemoryKeyResolver([signer.public_key])

    assert resolver.key_ids() == frozenset({signer.key_id})


def test_in_memory_resolver_can_register_a_public_key_pem() -> None:
    signer = Ed25519Signer.generate()
    resolver = InMemoryKeyResolver()

    assert resolver.add_pem(signer.public_key_pem()) == signer.key_id


def test_in_memory_resolver_rejects_a_pem_that_is_not_a_public_key() -> None:
    resolver = InMemoryKeyResolver()

    with pytest.raises(ConfigurationError, match="could not be loaded"):
        resolver.add_pem("-----BEGIN PUBLIC KEY-----\nnope\n-----END PUBLIC KEY-----\n")


# ---------------------------------------------------------------------------------
# FileKeyResolver
# ---------------------------------------------------------------------------------


def _write_public_pem(directory: pathlib.Path, name: str, signer) -> None:
    (directory / name).write_text(signer.public_key_pem(), encoding="ascii")


def test_file_resolver_loads_a_public_key_pem_from_its_directory(tmp_path: pathlib.Path) -> None:
    signer = Ed25519Signer.generate()
    _write_public_pem(tmp_path, "signing.pem", signer)

    resolver = FileKeyResolver(tmp_path)

    assert resolver.resolve(signer.key_id) is not None


def test_file_resolver_indexes_by_derived_key_id_not_by_filename(
    tmp_path: pathlib.Path,
) -> None:
    signer = Ed25519Signer.generate()
    _write_public_pem(tmp_path, "totally-unrelated-name.pem", signer)

    resolver = FileKeyResolver(tmp_path)

    assert resolver.resolve(signer.key_id) is not None
    assert resolver.resolve("totally-unrelated-name") is None


def test_file_resolver_loads_every_pem_in_the_directory(tmp_path: pathlib.Path) -> None:
    signers = [Ed25519Signer.generate(), RSASigner.generate(key_size=2048)]
    for index, signer in enumerate(signers):
        _write_public_pem(tmp_path, f"key{index}.pem", signer)

    resolver = FileKeyResolver(tmp_path)

    assert all(resolver.resolve(s.key_id) is not None for s in signers)


def test_file_resolver_returns_none_for_an_unknown_key_id(tmp_path: pathlib.Path) -> None:
    _write_public_pem(tmp_path, "a.pem", Ed25519Signer.generate())

    assert FileKeyResolver(tmp_path).resolve("00" * 32) is None


def test_file_resolver_ignores_files_without_a_matching_suffix(
    tmp_path: pathlib.Path,
) -> None:
    _write_public_pem(tmp_path, "good.pem", Ed25519Signer.generate())
    (tmp_path / "README.md").write_text("not a key", encoding="ascii")

    assert len(FileKeyResolver(tmp_path)) == 1


def test_file_resolver_ignores_subdirectories(tmp_path: pathlib.Path) -> None:
    _write_public_pem(tmp_path, "good.pem", Ed25519Signer.generate())
    (tmp_path / "nested.pem").mkdir()

    assert len(FileKeyResolver(tmp_path)) == 1


def test_file_resolver_accepts_a_custom_suffix(tmp_path: pathlib.Path) -> None:
    signer = Ed25519Signer.generate()
    _write_public_pem(tmp_path, "key.pub", signer)

    resolver = FileKeyResolver(tmp_path, suffixes=(".pub",))

    assert resolver.resolve(signer.key_id) is not None


def test_file_resolver_fails_loudly_on_a_malformed_pem_rather_than_skipping_it(
    tmp_path: pathlib.Path,
) -> None:
    # A key that silently fails to load becomes UNKNOWN_KEY at verification time, which
    # reads as tampering. Refuse at startup instead.
    (tmp_path / "broken.pem").write_text("-----BEGIN PUBLIC KEY-----\nx\n", encoding="ascii")

    with pytest.raises(ConfigurationError, match="broken.pem"):
        FileKeyResolver(tmp_path)


def test_file_resolver_rejects_a_directory_that_does_not_exist(
    tmp_path: pathlib.Path,
) -> None:
    with pytest.raises(ConfigurationError, match="not a directory"):
        FileKeyResolver(tmp_path / "absent")


def test_file_resolver_rejects_a_path_that_is_a_file(tmp_path: pathlib.Path) -> None:
    target = tmp_path / "a-file"
    target.write_text("x", encoding="ascii")

    with pytest.raises(ConfigurationError, match="not a directory"):
        FileKeyResolver(target)


def test_file_resolver_accepts_an_empty_directory(tmp_path: pathlib.Path) -> None:
    resolver = FileKeyResolver(tmp_path)

    assert len(resolver) == 0
    assert resolver.resolve("00" * 32) is None


def test_file_resolver_reload_picks_up_a_key_added_after_construction(
    tmp_path: pathlib.Path,
) -> None:
    resolver = FileKeyResolver(tmp_path)
    signer = Ed25519Signer.generate()
    _write_public_pem(tmp_path, "late.pem", signer)

    assert resolver.resolve(signer.key_id) is None
    resolver.reload()
    assert resolver.resolve(signer.key_id) is not None


def test_file_resolver_reload_drops_a_key_removed_from_the_directory(
    tmp_path: pathlib.Path,
) -> None:
    signer = Ed25519Signer.generate()
    _write_public_pem(tmp_path, "gone.pem", signer)
    resolver = FileKeyResolver(tmp_path)

    (tmp_path / "gone.pem").unlink()
    resolver.reload()

    assert resolver.resolve(signer.key_id) is None


def test_file_resolver_accepts_a_string_path(tmp_path: pathlib.Path) -> None:
    signer = Ed25519Signer.generate()
    _write_public_pem(tmp_path, "a.pem", signer)

    assert FileKeyResolver(str(tmp_path)).resolve(signer.key_id) is not None


def test_a_key_resolved_from_disk_verifies_the_signature_of_its_signer(
    tmp_path: pathlib.Path,
) -> None:
    signer = RSASigner.generate(key_size=2048)
    _write_public_pem(tmp_path, "rsa.pem", signer)
    payload = signing_payload("ledger-a", "aa" * 32)

    resolved = FileKeyResolver(tmp_path).resolve(signer.key_id)

    assert resolved is not None
    assert verify(payload, signer.sign(payload), resolved) is True


# ---------------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------------


def test_in_memory_resolver_satisfies_the_key_resolver_protocol() -> None:
    assert isinstance(InMemoryKeyResolver(), KeyResolver)


def test_file_resolver_satisfies_the_key_resolver_protocol(tmp_path: pathlib.Path) -> None:
    assert isinstance(FileKeyResolver(tmp_path), KeyResolver)


def test_an_object_without_resolve_is_not_a_key_resolver() -> None:
    class NotAResolver:
        pass

    assert not isinstance(NotAResolver(), KeyResolver)


def test_file_resolver_exposes_the_key_ids_it_loaded(tmp_path: pathlib.Path) -> None:
    signer = Ed25519Signer.generate()
    _write_public_pem(tmp_path, "a.pem", signer)

    assert FileKeyResolver(tmp_path).key_ids() == frozenset({signer.key_id})
