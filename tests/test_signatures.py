"""Tests for ``signledger.crypto.signatures`` (architecture contract, section 6)."""

import base64
import os
import pathlib
import subprocess
import sys

import pytest

from signledger.core.exceptions import ConfigurationError, SignatureError, ValidationError
from signledger.crypto.signatures import (
    ECDSASigner,
    Ed25519Signer,
    RSASigner,
    Signer,
    _missing_dependency_error,
    get_private_key_pem,
    load_pem_private_key,
    signing_payload,
    verify,
)


def test_signing_payload_is_domain_tag_ledger_id_and_raw_hash_bytes() -> None:
    payload = signing_payload("ledger-a", "00ff10")

    assert payload == b"signledger/v2\x00ledger-a\x00\x00\xff\x10"


def test_signing_payload_rejects_empty_ledger_id() -> None:
    with pytest.raises(ValidationError, match="ledger_id"):
        signing_payload("", "00ff")


def test_signing_payload_rejects_nul_byte_in_ledger_id() -> None:
    # A NUL inside ledger_id would make the framing ambiguous: hash width varies by
    # algorithm, so ("a\x00bb", short_hash) could encode identically to ("a", long_hash).
    with pytest.raises(ValidationError, match="NUL"):
        signing_payload("led\x00ger", "00ff")


def test_signing_payload_rejects_empty_entry_hash() -> None:
    with pytest.raises(ValidationError, match="entry hash"):
        signing_payload("ledger-a", "")


def test_signing_payload_rejects_non_hexadecimal_entry_hash() -> None:
    with pytest.raises(ValidationError, match="hexadecimal"):
        signing_payload("ledger-a", "zzzz")


def test_signing_payload_rejects_odd_length_entry_hash() -> None:
    with pytest.raises(ValidationError, match="hexadecimal"):
        signing_payload("ledger-a", "abc")


def test_signing_payload_rejects_whitespace_inside_entry_hash() -> None:
    # bytes.fromhex() silently skips ASCII spaces, so "00 ff" and "00ff" would collide.
    with pytest.raises(ValidationError, match="hexadecimal"):
        signing_payload("ledger-a", "00 ff")


# ---------------------------------------------------------------------------------
# Ed25519
# ---------------------------------------------------------------------------------


def test_ed25519_signer_reports_the_contract_algorithm_string() -> None:
    assert Ed25519Signer.generate().algorithm == "ed25519"


def test_ed25519_signature_is_base64_text() -> None:
    signer = Ed25519Signer.generate()

    signature = signer.sign(b"payload")

    assert isinstance(signature, str)
    assert base64.b64decode(signature, validate=True)


def test_ed25519_signature_verifies_against_its_own_public_key() -> None:
    signer = Ed25519Signer.generate()
    payload = signing_payload("ledger-a", "aa" * 32)

    assert verify(payload, signer.sign(payload), signer.public_key) is True


def test_verify_rejects_a_signature_over_a_different_payload() -> None:
    signer = Ed25519Signer.generate()
    signature = signer.sign(signing_payload("ledger-a", "aa" * 32))

    assert verify(signing_payload("ledger-a", "bb" * 32), signature, signer.public_key) is False


def test_verify_rejects_a_signature_made_by_a_different_key() -> None:
    payload = signing_payload("ledger-a", "aa" * 32)
    signature = Ed25519Signer.generate().sign(payload)

    assert verify(payload, signature, Ed25519Signer.generate().public_key) is False


# ---------------------------------------------------------------------------------
# verify(): malformed input is False, never an exception
# ---------------------------------------------------------------------------------


def test_verify_returns_false_for_a_malformed_base64_signature() -> None:
    signer = Ed25519Signer.generate()

    assert verify(b"payload", "!!!not base64!!!", signer.public_key) is False


def test_verify_returns_false_for_a_base64_signature_with_bad_padding() -> None:
    signer = Ed25519Signer.generate()

    assert verify(b"payload", "AAAA=", signer.public_key) is False


def test_verify_returns_false_for_a_non_ascii_signature_string() -> None:
    signer = Ed25519Signer.generate()

    assert verify(b"payload", "äöü", signer.public_key) is False


def test_verify_returns_false_for_a_well_formed_base64_signature_of_wrong_length() -> None:
    signer = Ed25519Signer.generate()

    assert verify(b"payload", base64.b64encode(b"short").decode(), signer.public_key) is False


def test_verify_rejects_a_str_payload_rather_than_silently_failing() -> None:
    signer = Ed25519Signer.generate()
    signature = signer.sign(b"payload")

    with pytest.raises(TypeError, match="payload must be bytes"):
        verify("payload", signature, signer.public_key)  # type: ignore[arg-type]


def test_sign_rejects_a_str_payload_rather_than_silently_failing() -> None:
    # This is the exact v1 defect: the documented path called signer(hash_hex_str).
    with pytest.raises(TypeError, match="payload must be bytes"):
        Ed25519Signer.generate().sign("deadbeef")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------------
# key_id
# ---------------------------------------------------------------------------------


def test_key_id_is_a_sha256_hex_fingerprint() -> None:
    key_id = Ed25519Signer.generate().key_id

    assert len(key_id) == 64
    assert set(key_id) <= set("0123456789abcdef")


def test_key_id_is_stable_across_a_pem_round_trip_of_the_same_key() -> None:
    from cryptography.hazmat.primitives.serialization import load_pem_public_key

    from signledger.crypto.signatures import compute_key_id

    signer = Ed25519Signer.generate()
    reloaded = load_pem_public_key(signer.public_key_pem().encode("ascii"))

    assert compute_key_id(reloaded) == signer.key_id


def test_key_id_differs_between_two_distinct_keys() -> None:
    assert Ed25519Signer.generate().key_id != Ed25519Signer.generate().key_id


def test_public_key_pem_is_a_subject_public_key_info_pem() -> None:
    pem = Ed25519Signer.generate().public_key_pem()

    assert pem.startswith("-----BEGIN PUBLIC KEY-----")
    assert pem.rstrip().endswith("-----END PUBLIC KEY-----")


# ---------------------------------------------------------------------------------
# Domain separation
# ---------------------------------------------------------------------------------


def test_signature_does_not_verify_for_the_same_entry_hash_under_another_ledger_id() -> None:
    signer = Ed25519Signer.generate()
    entry_hash = "aa" * 32
    signature = signer.sign(signing_payload("ledger-a", entry_hash))

    assert verify(signing_payload("ledger-b", entry_hash), signature, signer.public_key) is False


def test_signing_payload_differs_between_ledgers_for_one_entry_hash() -> None:
    entry_hash = "aa" * 32

    assert signing_payload("ledger-a", entry_hash) != signing_payload("ledger-b", entry_hash)


def test_signing_payload_separator_prevents_a_ledger_id_hash_framing_collision() -> None:
    # 0x61 is "a", so the hash "6161bb" is the bytes b"aa\xbb". Without the NUL separator
    # b"ledger" + b"aa\xbb" and b"ledgeraa" + b"\xbb" are the same byte string, and one
    # signature would cover an entry in either ledger.
    assert signing_payload("ledger", "6161bb") != signing_payload("ledgeraa", "bb")


# ---------------------------------------------------------------------------------
# RSA-PSS / SHA-256
# ---------------------------------------------------------------------------------


def test_rsa_signer_reports_the_contract_algorithm_string() -> None:
    assert RSASigner.generate(key_size=2048).algorithm == "rsa-pss-sha256"


def test_rsa_signature_verifies_against_its_own_public_key() -> None:
    signer = RSASigner.generate(key_size=2048)
    payload = signing_payload("ledger-a", "aa" * 32)

    assert verify(payload, signer.sign(payload), signer.public_key) is True


def test_rsa_signature_does_not_verify_over_a_different_payload() -> None:
    signer = RSASigner.generate(key_size=2048)
    signature = signer.sign(signing_payload("ledger-a", "aa" * 32))

    assert verify(signing_payload("ledger-b", "aa" * 32), signature, signer.public_key) is False


def test_rsa_signer_rejects_a_key_below_the_minimum_size() -> None:
    from cryptography.hazmat.primitives.asymmetric import rsa

    weak = rsa.generate_private_key(public_exponent=65537, key_size=1024)

    with pytest.raises(ConfigurationError, match="2048"):
        RSASigner(weak)


def test_rsa_generate_rejects_a_key_size_below_the_minimum() -> None:
    with pytest.raises(ConfigurationError, match="2048"):
        RSASigner.generate(key_size=1024)


def test_rsa_pss_salt_length_is_not_pinned_by_the_verifier() -> None:
    # A signature made with the maximum salt length must still verify: salt length is not
    # a security parameter to pin, and pinning it would break interop with other signers.
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding

    signer = RSASigner.generate(key_size=2048)
    payload = signing_payload("ledger-a", "aa" * 32)
    max_salt = signer._private_key.sign(
        payload,
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
        hashes.SHA256(),
    )

    assert verify(payload, base64.b64encode(max_salt).decode(), signer.public_key) is True


# ---------------------------------------------------------------------------------
# ECDSA
# ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("curve", "algorithm"),
    [
        ("p256", "ecdsa-p256-sha256"),
        ("p384", "ecdsa-p384-sha384"),
        ("p521", "ecdsa-p521-sha512"),
    ],
)
def test_ecdsa_signer_reports_the_algorithm_for_its_curve(curve: str, algorithm: str) -> None:
    assert ECDSASigner.generate(curve=curve).algorithm == algorithm


@pytest.mark.parametrize("curve", ["p256", "p384", "p521"])
def test_ecdsa_signature_verifies_against_its_own_public_key(curve: str) -> None:
    signer = ECDSASigner.generate(curve=curve)
    payload = signing_payload("ledger-a", "aa" * 32)

    assert verify(payload, signer.sign(payload), signer.public_key) is True


@pytest.mark.parametrize("curve", ["p256", "p384", "p521"])
def test_ecdsa_signature_does_not_verify_over_a_different_payload(curve: str) -> None:
    signer = ECDSASigner.generate(curve=curve)
    signature = signer.sign(signing_payload("ledger-a", "aa" * 32))

    assert verify(signing_payload("ledger-b", "aa" * 32), signature, signer.public_key) is False


def test_ecdsa_signer_rejects_a_loaded_key_whose_curve_is_not_the_requested_one() -> None:
    # v1 silently kept the loaded key's curve, so an operator upgrading p256 -> p384 got
    # a no-op and believed they had strengthened the ledger.
    from cryptography.hazmat.primitives.asymmetric import ec

    p256_key = ec.generate_private_key(ec.SECP256R1())

    with pytest.raises(ConfigurationError, match="secp384r1"):
        ECDSASigner(p256_key, curve="p384")


def test_ecdsa_signer_accepts_a_loaded_key_whose_curve_matches() -> None:
    from cryptography.hazmat.primitives.asymmetric import ec

    signer = ECDSASigner(ec.generate_private_key(ec.SECP384R1()), curve="p384")

    assert signer.algorithm == "ecdsa-p384-sha384"


def test_ecdsa_signer_rejects_an_unknown_curve_name() -> None:
    with pytest.raises(ConfigurationError, match="unsupported ECDSA curve"):
        ECDSASigner.generate(curve="p192")


def test_ecdsa_signer_accepts_curve_aliases_for_one_curve() -> None:
    assert ECDSASigner.generate(curve="P-256").algorithm == ECDSASigner.generate(
        curve="secp256r1"
    ).algorithm


def test_verify_raises_for_an_elliptic_curve_key_on_an_unsupported_curve() -> None:
    # secp256k1 is a real curve cryptography supports; SignLedger does not sign with it,
    # and guessing a digest for it would be worse than refusing.
    from cryptography.hazmat.primitives.asymmetric import ec

    key = ec.generate_private_key(ec.SECP256K1()).public_key()

    with pytest.raises(SignatureError, match="secp256k1"):
        verify(b"payload", base64.b64encode(b"sig").decode(), key)


def test_verify_raises_for_a_key_type_that_cannot_sign() -> None:
    from cryptography.hazmat.primitives.asymmetric import x25519

    key = x25519.X25519PrivateKey.generate().public_key()

    with pytest.raises(SignatureError, match="unsupported public key type"):
        verify(b"payload", base64.b64encode(b"sig").decode(), key)


# ---------------------------------------------------------------------------------
# Private key serialization
# ---------------------------------------------------------------------------------


def test_get_private_key_pem_refuses_to_emit_plaintext_without_an_explicit_opt_in() -> None:
    # v1 hardcoded plaintext PKCS8, so every documented example wrote an unprotected
    # signing key to disk.
    signer = Ed25519Signer.generate()

    with pytest.raises(ConfigurationError, match="allow_unencrypted"):
        get_private_key_pem(signer._private_key)


def test_get_private_key_pem_encrypts_by_default_when_given_a_password() -> None:
    signer = Ed25519Signer.generate()

    pem = get_private_key_pem(signer._private_key, password=b"correct horse")

    assert pem.startswith("-----BEGIN ENCRYPTED PRIVATE KEY-----")


def test_get_private_key_pem_emits_plaintext_only_on_explicit_opt_in() -> None:
    signer = Ed25519Signer.generate()

    pem = get_private_key_pem(signer._private_key, allow_unencrypted=True)

    assert pem.startswith("-----BEGIN PRIVATE KEY-----")


def test_get_private_key_pem_rejects_a_password_together_with_the_plaintext_opt_in() -> None:
    signer = Ed25519Signer.generate()

    with pytest.raises(ConfigurationError, match="both"):
        get_private_key_pem(signer._private_key, password=b"pw", allow_unencrypted=True)


def test_get_private_key_pem_rejects_an_empty_password() -> None:
    signer = Ed25519Signer.generate()

    with pytest.raises(ConfigurationError, match="empty"):
        get_private_key_pem(signer._private_key, password=b"")


def test_encrypted_private_key_round_trips_to_the_same_key_id() -> None:
    signer = Ed25519Signer.generate()
    pem = get_private_key_pem(signer._private_key, password=b"correct horse")

    reloaded = Ed25519Signer(load_pem_private_key(pem.encode(), password=b"correct horse"))

    assert reloaded.key_id == signer.key_id


def test_a_signature_from_a_reloaded_encrypted_key_still_verifies() -> None:
    signer = Ed25519Signer.generate()
    pem = get_private_key_pem(signer._private_key, password=b"correct horse")
    reloaded = Ed25519Signer(load_pem_private_key(pem.encode(), password=b"correct horse"))
    payload = signing_payload("ledger-a", "aa" * 32)

    assert verify(payload, reloaded.sign(payload), signer.public_key) is True


def test_load_pem_private_key_fails_on_a_wrong_password() -> None:
    signer = Ed25519Signer.generate()
    pem = get_private_key_pem(signer._private_key, password=b"correct horse")

    with pytest.raises(ConfigurationError, match="could not be loaded"):
        load_pem_private_key(pem.encode(), password=b"wrong horse")


def test_load_pem_private_key_fails_when_an_encrypted_key_is_given_no_password() -> None:
    signer = Ed25519Signer.generate()
    pem = get_private_key_pem(signer._private_key, password=b"correct horse")

    with pytest.raises(ConfigurationError, match="password"):
        load_pem_private_key(pem.encode())


def test_load_pem_private_key_fails_when_a_password_is_given_for_a_plaintext_key() -> None:
    signer = Ed25519Signer.generate()
    pem = get_private_key_pem(signer._private_key, allow_unencrypted=True)

    with pytest.raises(ConfigurationError, match="password"):
        load_pem_private_key(pem.encode(), password=b"unexpected")


def test_load_pem_private_key_fails_on_data_that_is_not_a_pem() -> None:
    with pytest.raises(ConfigurationError, match="could not be loaded"):
        load_pem_private_key(b"not a pem at all")


def test_private_key_load_failure_never_echoes_the_password() -> None:
    # Contract section 10: no exception message may contain credentials.
    signer = Ed25519Signer.generate()
    pem = get_private_key_pem(signer._private_key, password=b"correct horse")

    with pytest.raises(ConfigurationError) as excinfo:
        load_pem_private_key(pem.encode(), password=b"swordfish")

    assert "swordfish" not in str(excinfo.value)
    assert "correct horse" not in str(excinfo.value)


# ---------------------------------------------------------------------------------
# from_pem
# ---------------------------------------------------------------------------------


def test_ed25519_signer_from_pem_round_trips_to_the_same_key_id() -> None:
    signer = Ed25519Signer.generate()
    pem = get_private_key_pem(signer._private_key, password=b"pw")

    assert Ed25519Signer.from_pem(pem, password=b"pw").key_id == signer.key_id


def test_rsa_signer_from_pem_round_trips_to_the_same_key_id() -> None:
    signer = RSASigner.generate(key_size=2048)
    pem = get_private_key_pem(signer._private_key, password=b"pw")

    assert RSASigner.from_pem(pem, password=b"pw").key_id == signer.key_id


def test_ecdsa_signer_from_pem_round_trips_to_the_same_key_id() -> None:
    signer = ECDSASigner.generate(curve="p384")
    pem = get_private_key_pem(signer._private_key, password=b"pw")

    assert ECDSASigner.from_pem(pem, password=b"pw", curve="p384").key_id == signer.key_id


def test_ecdsa_signer_from_pem_enforces_the_requested_curve() -> None:
    signer = ECDSASigner.generate(curve="p256")
    pem = get_private_key_pem(signer._private_key, password=b"pw")

    with pytest.raises(ConfigurationError, match="secp521r1"):
        ECDSASigner.from_pem(pem, password=b"pw", curve="p521")


def test_from_pem_accepts_bytes_as_well_as_text() -> None:
    signer = Ed25519Signer.generate()
    pem = get_private_key_pem(signer._private_key, password=b"pw")

    assert Ed25519Signer.from_pem(pem.encode("ascii"), password=b"pw").key_id == signer.key_id


def test_rsa_signer_rejects_an_ed25519_private_key() -> None:
    other = Ed25519Signer.generate()

    with pytest.raises(ConfigurationError, match="RSAPrivateKey"):
        RSASigner(other._private_key)


def test_ed25519_signer_rejects_an_elliptic_curve_private_key() -> None:
    other = ECDSASigner.generate(curve="p256")

    with pytest.raises(ConfigurationError, match="Ed25519PrivateKey"):
        Ed25519Signer(other._private_key)


def test_ecdsa_signer_rejects_an_rsa_private_key() -> None:
    other = RSASigner.generate(key_size=2048)

    with pytest.raises(ConfigurationError, match="EllipticCurvePrivateKey"):
        ECDSASigner(other._private_key)


# ---------------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "signer_factory",
    [
        Ed25519Signer.generate,
        lambda: RSASigner.generate(key_size=2048),
        lambda: ECDSASigner.generate(curve="p256"),
    ],
    ids=["ed25519", "rsa", "ecdsa"],
)
def test_every_shipped_signer_satisfies_the_signer_protocol(signer_factory) -> None:
    assert isinstance(signer_factory(), Signer)


def test_an_object_missing_sign_is_not_a_signer() -> None:
    class NotASigner:
        key_id = "x"
        algorithm = "y"

        def public_key_pem(self) -> str:
            return ""

    assert not isinstance(NotASigner(), Signer)


# ---------------------------------------------------------------------------------
# cryptography is an OPTIONAL extra
# ---------------------------------------------------------------------------------

# Run in a subprocess with a meta-path hook that makes `import cryptography` genuinely
# fail. Nothing inside signledger is mocked: the dependency really is absent.
_NO_CRYPTOGRAPHY_PROBE = """
import importlib, importlib.machinery, os, sys, types


class _Blocker:
    def find_spec(self, name, path=None, target=None):
        if name == "cryptography" or name.startswith("cryptography."):
            raise ImportError("No module named 'cryptography'")
        return None


for _name in [m for m in sys.modules if m.split(".")[0] == "cryptography"]:
    del sys.modules[_name]
sys.meta_path.insert(0, _Blocker())

root, snippet = sys.argv[1], sys.argv[2]
try:
    importlib.import_module("signledger")
except Exception:
    package = types.ModuleType("signledger")
    package.__path__ = [os.path.join(root, "signledger")]
    spec = importlib.machinery.ModuleSpec("signledger", None, is_package=True)
    spec.submodule_search_locations = package.__path__
    package.__spec__ = spec
    sys.modules["signledger"] = package
importlib.import_module("signledger.core.entry")

try:
    import cryptography
except ImportError:
    pass
else:
    raise AssertionError("the blocker did not work; cryptography is still importable")

exec(snippet)
"""


def _run_without_cryptography(snippet: str) -> "subprocess.CompletedProcess[str]":
    root = str(pathlib.Path(__file__).resolve().parent.parent)
    return subprocess.run(
        [sys.executable, "-c", _NO_CRYPTOGRAPHY_PROBE, root, snippet],
        capture_output=True,
        text=True,
        cwd=root,
        env={**os.environ, "PYTHONPATH": root},
        check=False,
    )


@pytest.mark.parametrize(
    "module",
    ["signledger.crypto.signatures", "signledger.crypto.keys"],
)
def test_module_still_imports_when_cryptography_is_absent(module: str) -> None:
    result = _run_without_cryptography(f"import {module}; print('imported')")

    assert result.returncode == 0, result.stderr
    assert "imported" in result.stdout


def test_signing_payload_still_works_when_cryptography_is_absent() -> None:
    # It is pure byte framing; a verifier should be able to build one with no extras.
    result = _run_without_cryptography(
        "from signledger.crypto.signatures import signing_payload\n"
        "expected = b'signledger/v2' + bytes([0]) + b'l' + bytes([0]) + bytes([0, 255])\n"
        "assert signing_payload('l', '00ff') == expected\n"
        "print('ok')"
    )

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


@pytest.mark.parametrize(
    ("label", "expression"),
    [
        ("ed25519", "Ed25519Signer.generate()"),
        ("rsa", "RSASigner.generate(key_size=2048)"),
        ("ecdsa", "ECDSASigner.generate(curve='p256')"),
        ("verify", "verify(b'p', 'AAAA', object())"),
        ("compute_key_id", "compute_key_id(object())"),
        ("load_pem_private_key", "load_pem_private_key(b'x')"),
        ("load_pem_public_key", "load_pem_public_key(b'x')"),
        ("get_private_key_pem", "get_private_key_pem(object(), password=b'pw')"),
    ],
)
def test_signature_entry_points_raise_a_clear_import_error_without_cryptography(
    label: str, expression: str
) -> None:
    result = _run_without_cryptography(
        "from signledger.crypto.signatures import *\n"
        "from signledger.crypto.signatures import compute_key_id, load_pem_public_key\n"
        "try:\n"
        f"    {expression}\n"
        "except ImportError as exc:\n"
        "    print('RAISED:', exc)\n"
        "else:\n"
        "    raise AssertionError('no ImportError was raised')\n"
    )

    assert result.returncode == 0, result.stderr
    assert "signledger[signatures]" in result.stdout


@pytest.mark.parametrize(
    ("label", "expression"),
    [
        ("in_memory", "InMemoryKeyResolver()"),
        ("file", "FileKeyResolver('.')"),
    ],
)
def test_key_resolvers_raise_a_clear_import_error_without_cryptography(
    label: str, expression: str
) -> None:
    result = _run_without_cryptography(
        "from signledger.crypto.keys import FileKeyResolver, InMemoryKeyResolver\n"
        "try:\n"
        f"    {expression}\n"
        "except ImportError as exc:\n"
        "    print('RAISED:', exc)\n"
        "else:\n"
        "    raise AssertionError('no ImportError was raised')\n"
    )

    assert result.returncode == 0, result.stderr
    assert "signledger[signatures]" in result.stdout


def test_missing_dependency_error_names_the_extra_and_keeps_the_original_cause() -> None:
    cause = ImportError("No module named 'cryptography'")

    error = _missing_dependency_error(cause)

    assert "signledger[signatures]" in str(error)
    assert error.__cause__ is cause
