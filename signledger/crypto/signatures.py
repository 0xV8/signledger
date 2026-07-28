"""Signing and verification (architecture contract, section 6).

Signing and verification are **symmetric**. A signer whose output nothing ever checks is
not a feature: v1 shipped ``Signer.verify()`` with zero call sites, and its documented
signing path raised ``TypeError`` because the caller passed a hex ``str`` to a method that
took ``bytes``. Every signer here is exercised end to end against :func:`verify`.

``cryptography`` is an optional extra (``pip install 'signledger[signatures]'``). Importing
this module without it succeeds; constructing a signer raises a clear ``ImportError``.
"""

import base64
import binascii
import hashlib
import re
from typing import TYPE_CHECKING, Any, Optional, Protocol, Union, cast, runtime_checkable

from signledger.core.exceptions import ConfigurationError, SignatureError, ValidationError

try:
    from cryptography.exceptions import InvalidSignature, UnsupportedAlgorithm
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec, ed25519, padding, rsa

    _IMPORT_ERROR: Optional[ImportError] = None
except ImportError as _exc:  # pragma: no cover - exercised by a subprocess probe
    _IMPORT_ERROR = _exc

if TYPE_CHECKING:  # pragma: no cover
    from cryptography.hazmat.primitives.asymmetric.ec import (
        EllipticCurve,
        EllipticCurvePrivateKey,
        EllipticCurvePublicKey,
    )
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
        Ed25519PublicKey,
    )
    from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey, RSAPublicKey
    from cryptography.hazmat.primitives.hashes import HashAlgorithm

    PublicKey = Union[RSAPublicKey, EllipticCurvePublicKey, Ed25519PublicKey]
    PrivateKey = Union[RSAPrivateKey, EllipticCurvePrivateKey, Ed25519PrivateKey]
else:
    PublicKey = Any
    PrivateKey = Any

__all__ = [
    "MIN_RSA_KEY_SIZE",
    "ECDSASigner",
    "Ed25519Signer",
    "PrivateKey",
    "PublicKey",
    "RSASigner",
    "Signer",
    "compute_key_id",
    "get_private_key_pem",
    "load_pem_private_key",
    "load_pem_public_key",
    "signing_payload",
    "verify",
]

#: Domain tag. Bound into every signed payload so a signature over an entry hash cannot be
#: replayed as a signature over anything else SignLedger ever signs.
_DOMAIN_TAG = b"signledger/v2"

#: An even-length run of hex digits and nothing else. Deliberately stricter than
#: ``bytes.fromhex``, which silently skips ASCII spaces - so "00 ff" and "00ff" would
#: produce one payload from two different strings.
_HEX_RE = re.compile(r"\A(?:[0-9a-fA-F]{2})+\Z")

#: Smallest RSA modulus this build will sign with. Below this the signature is not worth
#: the bytes it occupies, and a tamper-evidence claim resting on it would be false.
MIN_RSA_KEY_SIZE = 2048


def _missing_dependency_error(cause: BaseException) -> ImportError:
    """Build the error raised when the optional ``cryptography`` extra is not installed."""
    error = ImportError(
        "signledger signatures require the optional 'cryptography' dependency; "
        "install it with: pip install 'signledger[signatures]'"
    )
    error.__cause__ = cause
    return error


def _require_cryptography() -> None:
    if _IMPORT_ERROR is not None:  # pragma: no cover - exercised by a subprocess probe
        raise _missing_dependency_error(_IMPORT_ERROR)


def _require_bytes(payload: object) -> None:
    """Reject a ``str`` payload loudly. This is the exact shape of the v1 signing bug."""
    if not isinstance(payload, bytes):
        raise TypeError(
            f"payload must be bytes, not {type(payload).__name__}; build it with "
            "signing_payload(ledger_id, entry_hash_hex)"
        )


def signing_payload(ledger_id: str, entry_hash_hex: str) -> bytes:
    """Return the domain-separated bytes that a signature is computed over.

    ``b"signledger/v2\\x00" + ledger_id + b"\\x00" + raw_hash_bytes``

    The ledger id is inside the payload, so a signature harvested from one ledger does not
    verify against an entry carrying the same hash in another ledger. The hash is embedded
    as *raw bytes*, never as hex text.
    """
    if not ledger_id:
        raise ValidationError("ledger_id must not be empty")
    if "\x00" in ledger_id:
        raise ValidationError(
            "ledger_id must not contain a NUL byte; NUL is the field separator and an "
            "embedded one would make the signed payload ambiguous"
        )
    if not entry_hash_hex:
        raise ValidationError("entry hash must not be empty; seal the entry before signing")
    if not _HEX_RE.match(entry_hash_hex):
        raise ValidationError("entry hash must be an even-length hexadecimal string")
    return (
        _DOMAIN_TAG
        + b"\x00"
        + ledger_id.encode("utf-8")
        + b"\x00"
        + binascii.unhexlify(entry_hash_hex)
    )


def compute_key_id(public_key: PublicKey) -> str:
    """Return the SHA-256 fingerprint of ``public_key``'s DER SubjectPublicKeyInfo.

    Derived purely from the key material, so it is stable across a PEM round-trip and
    differs for different keys. A key id is always recomputed, never asserted by a caller.
    """
    _require_cryptography()
    der = public_key.public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return hashlib.sha256(der).hexdigest()


def verify(payload: bytes, signature: str, public_key: PublicKey) -> bool:
    """Return whether ``signature`` (base64) is valid over ``payload`` for ``public_key``.

    Only ``InvalidSignature`` and explicit base64 decode errors are caught. v1 wrapped this
    in ``except Exception``, which turned real bugs - a wrong key type, an unsupported
    curve - into a silent "signature did not verify" that read like an attack.
    """
    _require_cryptography()
    _require_bytes(payload)
    try:
        raw_signature = base64.b64decode(signature, validate=True)
    except ValueError:
        # The two documented failures of b64decode(validate=True), and nothing else:
        # binascii.Error (non-alphabet character or bad padding) - itself a ValueError -
        # and ValueError("string argument should contain only ASCII characters"), which
        # base64 raises in place of the UnicodeEncodeError it caught.
        return False

    try:
        if isinstance(public_key, ed25519.Ed25519PublicKey):
            public_key.verify(raw_signature, payload)
        elif isinstance(public_key, rsa.RSAPublicKey):
            public_key.verify(
                raw_signature,
                payload,
                # AUTO recovers the salt length from the signature. Salt length is not a
                # security parameter to pin, and pinning it would reject valid signatures
                # produced by other conforming implementations.
                padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.AUTO),
                hashes.SHA256(),
            )
        elif isinstance(public_key, ec.EllipticCurvePublicKey):
            public_key.verify(raw_signature, payload, ec.ECDSA(_ecdsa_hash_for(public_key.curve)))
        else:
            raise SignatureError(
                f"unsupported public key type {type(public_key).__name__}; "
                "expected an RSA, elliptic-curve or Ed25519 public key"
            )
    except InvalidSignature:
        return False
    return True


# ---------------------------------------------------------------------------------------
# Private key serialization
# ---------------------------------------------------------------------------------------


def load_pem_private_key(data: bytes, password: Optional[bytes] = None) -> PrivateKey:
    """Load a PEM private key, optionally decrypting it with ``password``.

    Every failure - wrong password, missing password, unexpected password, malformed PEM -
    becomes a :class:`ConfigurationError` whose message names no credential. The original
    exception is deliberately not chained: it is the one object in this call that a naive
    ``repr()`` in a log handler could carry key material out of.
    """
    _require_cryptography()
    try:
        return cast(PrivateKey, serialization.load_pem_private_key(data, password=password))
    except TypeError:
        # "Password was not given but private key is encrypted", and its inverse.
        raise ConfigurationError(
            "private key password mismatch: an encrypted key needs a password and a "
            "plaintext key must not be given one"
        ) from None
    except ValueError:
        raise ConfigurationError(
            "private key could not be loaded; the PEM may be malformed, the password may "
            "be wrong, or the key may use an unsupported encryption algorithm"
        ) from None


def load_pem_public_key(data: Union[str, bytes]) -> PublicKey:
    """Load a SubjectPublicKeyInfo PEM, reporting failure as a ``ConfigurationError``."""
    _require_cryptography()
    raw = data.encode("ascii") if isinstance(data, str) else data
    try:
        return cast(PublicKey, serialization.load_pem_public_key(raw))
    except (ValueError, UnsupportedAlgorithm):
        raise ConfigurationError(
            "public key could not be loaded; the PEM may be malformed or the key may use "
            "an algorithm this build does not support"
        ) from None


def get_private_key_pem(
    private_key: PrivateKey,
    password: Optional[bytes] = None,
    *,
    allow_unencrypted: bool = False,
) -> str:
    """Serialize ``private_key`` to a PKCS#8 PEM, **encrypted unless told otherwise**.

    v1 hardcoded ``NoEncryption``, so following its documentation wrote an unprotected
    signing key to disk. Here plaintext requires ``allow_unencrypted=True`` at the call
    site, where it is visible in review.
    """
    _require_cryptography()
    if password is not None and allow_unencrypted:
        raise ConfigurationError(
            "pass either a password or allow_unencrypted=True, not both; the intent of "
            "passing both is ambiguous"
        )
    if password is None and not allow_unencrypted:
        raise ConfigurationError(
            "refusing to write a plaintext private key: pass a password, or pass "
            "allow_unencrypted=True to opt in explicitly"
        )
    if password is not None and not password:
        raise ConfigurationError("private key password must not be empty")

    encryption: serialization.KeySerializationEncryption
    if password is None:
        encryption = serialization.NoEncryption()
    else:
        encryption = serialization.BestAvailableEncryption(password)
    return private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=encryption,
    ).decode("ascii")


# ---------------------------------------------------------------------------------------
# Curves
# ---------------------------------------------------------------------------------------

#: Curve name -> (contract algorithm string, digest name). NIST pairs each curve with a
#: digest of matching strength; mismatching them weakens the signature to the smaller of
#: the two, so the digest follows from the curve and is never a caller choice.
_CURVES = {
    "secp256r1": ("ecdsa-p256-sha256", "SHA256"),
    "secp384r1": ("ecdsa-p384-sha384", "SHA384"),
    "secp521r1": ("ecdsa-p521-sha512", "SHA512"),
}

#: Accepted spellings of the curves above, normalised to their ``cryptography`` names.
_CURVE_ALIASES = {
    "p256": "secp256r1",
    "p-256": "secp256r1",
    "prime256v1": "secp256r1",
    "secp256r1": "secp256r1",
    "p384": "secp384r1",
    "p-384": "secp384r1",
    "secp384r1": "secp384r1",
    "p521": "secp521r1",
    "p-521": "secp521r1",
    "secp521r1": "secp521r1",
}


def _normalise_curve(curve: str) -> str:
    try:
        return _CURVE_ALIASES[curve.lower()]
    except KeyError:
        raise ConfigurationError(
            f"unsupported ECDSA curve {curve!r}; choose one of "
            f"{sorted(set(_CURVE_ALIASES.values()))}"
        ) from None


def _ecdsa_hash_for(curve: "EllipticCurve") -> "HashAlgorithm":
    entry = _CURVES.get(curve.name)
    if entry is None:
        raise SignatureError(
            f"unsupported ECDSA curve {curve.name!r}; SignLedger signs with {sorted(_CURVES)}"
        )
    digest: HashAlgorithm = getattr(hashes, entry[1])()
    return digest


# ---------------------------------------------------------------------------------------
# Signers
# ---------------------------------------------------------------------------------------


@runtime_checkable
class Signer(Protocol):
    """The signing side of contract section 6."""

    key_id: str
    algorithm: str

    def sign(self, payload: bytes) -> str: ...

    def public_key_pem(self) -> str: ...


class _BaseSigner:
    """Shared key-identity plumbing. ``algorithm`` and ``sign`` belong to the subclass."""

    algorithm: str

    def __init__(self, private_key: PrivateKey) -> None:
        _require_cryptography()
        expected = self._key_type()
        if not isinstance(private_key, expected):
            raise ConfigurationError(
                f"{type(self).__name__} needs a {expected.__name__}, "
                f"got {type(private_key).__name__}"
            )
        self._private_key = private_key

    @classmethod
    def _key_type(cls) -> type:
        """The private key class this signer accepts. Resolved lazily: the cryptography
        classes do not exist when the optional extra is absent."""
        raise NotImplementedError

    @classmethod
    def _load(cls, pem: Union[str, bytes], password: Optional[bytes]) -> PrivateKey:
        _require_cryptography()
        data = pem.encode("ascii") if isinstance(pem, str) else pem
        return load_pem_private_key(data, password=password)

    @property
    def public_key(self) -> PublicKey:
        """The verifying key, for registering with a key resolver."""
        return self._private_key.public_key()

    @property
    def key_id(self) -> str:
        """SHA-256 fingerprint of the DER public key (contract section 6)."""
        return compute_key_id(self.public_key)

    def public_key_pem(self) -> str:
        """The public key as a SubjectPublicKeyInfo PEM."""
        return self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        ).decode("ascii")


class Ed25519Signer(_BaseSigner):
    """Signs with Ed25519 (contract algorithm string ``"ed25519"``)."""

    algorithm = "ed25519"

    @classmethod
    def _key_type(cls) -> type:
        return ed25519.Ed25519PrivateKey

    @classmethod
    def generate(cls) -> "Ed25519Signer":
        """Return a signer over a freshly generated Ed25519 key."""
        _require_cryptography()
        return cls(ed25519.Ed25519PrivateKey.generate())

    @classmethod
    def from_pem(
        cls, pem: Union[str, bytes], password: Optional[bytes] = None
    ) -> "Ed25519Signer":
        """Return a signer over the Ed25519 private key in ``pem``."""
        return cls(cast("Ed25519PrivateKey", cls._load(pem, password)))

    def sign(self, payload: bytes) -> str:
        _require_bytes(payload)
        key = cast("Ed25519PrivateKey", self._private_key)
        return base64.b64encode(key.sign(payload)).decode("ascii")


class RSASigner(_BaseSigner):
    """Signs with RSASSA-PSS over SHA-256 (contract string ``"rsa-pss-sha256"``)."""

    algorithm = "rsa-pss-sha256"

    @classmethod
    def _key_type(cls) -> type:
        return rsa.RSAPrivateKey

    def __init__(self, private_key: "RSAPrivateKey") -> None:
        super().__init__(private_key)
        if private_key.key_size < MIN_RSA_KEY_SIZE:
            raise ConfigurationError(
                f"RSA key size {private_key.key_size} is below the {MIN_RSA_KEY_SIZE}-bit "
                "minimum SignLedger will sign with"
            )

    @classmethod
    def generate(cls, key_size: int = 3072) -> "RSASigner":
        """Return a signer over a freshly generated RSA key."""
        _require_cryptography()
        if key_size < MIN_RSA_KEY_SIZE:
            raise ConfigurationError(
                f"RSA key size {key_size} is below the {MIN_RSA_KEY_SIZE}-bit minimum "
                "SignLedger will sign with"
            )
        return cls(rsa.generate_private_key(public_exponent=65537, key_size=key_size))

    @classmethod
    def from_pem(cls, pem: Union[str, bytes], password: Optional[bytes] = None) -> "RSASigner":
        """Return a signer over the RSA private key in ``pem``."""
        return cls(cast("RSAPrivateKey", cls._load(pem, password)))

    def sign(self, payload: bytes) -> str:
        _require_bytes(payload)
        key = cast("RSAPrivateKey", self._private_key)
        raw = key.sign(
            payload,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
            hashes.SHA256(),
        )
        return base64.b64encode(raw).decode("ascii")


class ECDSASigner(_BaseSigner):
    """Signs with ECDSA on P-256, P-384 or P-521, digest matched to the curve.

    The requested ``curve`` is **asserted** against the loaded key rather than inferred
    from it. v1 inferred it, so an operator who edited their config from p256 to p384 and
    restarted kept signing with the old p256 key and never learned the upgrade was a no-op.
    """

    @classmethod
    def _key_type(cls) -> type:
        return ec.EllipticCurvePrivateKey

    def __init__(self, private_key: "EllipticCurvePrivateKey", *, curve: str = "p256") -> None:
        super().__init__(private_key)
        requested = _normalise_curve(curve)
        actual = private_key.curve.name
        if actual != requested:
            raise ConfigurationError(
                f"ECDSA key is on curve {actual!r} but {requested!r} was requested; "
                "load a key on the requested curve rather than silently using the other one"
            )
        self.algorithm = _CURVES[requested][0]
        self._digest_name = _CURVES[requested][1]

    @classmethod
    def generate(cls, curve: str = "p256") -> "ECDSASigner":
        """Return a signer over a freshly generated key on ``curve``."""
        _require_cryptography()
        normalised = _normalise_curve(curve)
        curve_cls = getattr(ec, normalised.upper())
        return cls(ec.generate_private_key(curve_cls()), curve=normalised)

    @classmethod
    def from_pem(
        cls,
        pem: Union[str, bytes],
        password: Optional[bytes] = None,
        *,
        curve: str = "p256",
    ) -> "ECDSASigner":
        """Return a signer over the EC private key in ``pem``, asserting its curve."""
        return cls(cast("EllipticCurvePrivateKey", cls._load(pem, password)), curve=curve)

    def sign(self, payload: bytes) -> str:
        _require_bytes(payload)
        digest: HashAlgorithm = getattr(hashes, self._digest_name)()
        key = cast("EllipticCurvePrivateKey", self._private_key)
        raw = key.sign(payload, ec.ECDSA(digest))
        return base64.b64encode(raw).decode("ascii")
