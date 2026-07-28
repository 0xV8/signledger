"""Key resolution - turning a ``key_id`` back into a verifying key (contract section 6).

A signature is only evidence if the verifier can find the right public key *independently*
of the entry that carries the signature. That is what a resolver is for: the entry names a
``key_id`` (inside the hash preimage, so it cannot be swapped), and the resolver decides
which key that id means.

Key ids are always **derived** from key material by
:func:`~signledger.crypto.signatures.compute_key_id`, never accepted from a caller. A
resolver that let you file a key under an id of your choosing would let an attacker map a
victim's ``key_id`` onto their own key.

``cryptography`` is an optional extra; importing this module without it succeeds, and
constructing a resolver raises a clear ``ImportError``.
"""

import os
import pathlib
from typing import (
    Dict,
    FrozenSet,
    Iterable,
    Optional,
    Protocol,
    Sequence,
    Union,
    runtime_checkable,
)

from signledger.core.exceptions import ConfigurationError
from signledger.crypto.signatures import (
    PublicKey,
    _require_cryptography,
    compute_key_id,
    load_pem_public_key,
)

__all__ = [
    "DEFAULT_KEY_SUFFIXES",
    "FileKeyResolver",
    "InMemoryKeyResolver",
    "KeyResolver",
]

#: Filename suffixes :class:`FileKeyResolver` treats as key material by default.
DEFAULT_KEY_SUFFIXES = (".pem",)


@runtime_checkable
class KeyResolver(Protocol):
    """Resolves a ``key_id`` to a public key, or ``None`` when it is not known."""

    def resolve(self, key_id: str) -> Optional[PublicKey]: ...


class InMemoryKeyResolver:
    """Holds public keys in a dict, indexed by their derived fingerprints.

    Suited to tests and to processes that load their trusted keys at startup. The
    constructor takes *keys*, not a ``{key_id: key}`` mapping, precisely so no caller can
    introduce a wrong id -> key binding.
    """

    def __init__(self, public_keys: Optional[Iterable[PublicKey]] = None) -> None:
        _require_cryptography()
        self._keys: Dict[str, PublicKey] = {}
        for public_key in public_keys or ():
            self.add(public_key)

    def add(self, public_key: PublicKey) -> str:
        """Register ``public_key`` and return the ``key_id`` it was filed under."""
        key_id = compute_key_id(public_key)
        self._keys[key_id] = public_key
        return key_id

    def add_pem(self, pem: Union[str, bytes]) -> str:
        """Register the public key encoded in ``pem`` and return its ``key_id``."""
        return self.add(load_pem_public_key(pem))

    def resolve(self, key_id: str) -> Optional[PublicKey]:
        return self._keys.get(key_id)

    def key_ids(self) -> FrozenSet[str]:
        """The fingerprints this resolver currently knows."""
        return frozenset(self._keys)

    def __len__(self) -> int:
        return len(self._keys)


class FileKeyResolver:
    """Loads public key PEMs from a directory, indexed by derived fingerprint.

    The directory is scanned once, at construction, so a malformed key is reported while
    the process is starting rather than in the middle of a verification run - where it
    would surface as ``UNKNOWN_KEY`` and read like tampering. Call :meth:`reload` after
    rotating keys on disk.

    Filenames carry no meaning: a key resolves under its own fingerprint whatever the file
    is called, so renaming a file cannot silently re-point a ``key_id``.
    """

    def __init__(
        self,
        directory: Union[str, "os.PathLike[str]"],
        *,
        suffixes: Sequence[str] = DEFAULT_KEY_SUFFIXES,
    ) -> None:
        _require_cryptography()
        self.directory = pathlib.Path(directory)
        self.suffixes = tuple(suffixes)
        self._keys: Dict[str, PublicKey] = {}
        self.reload()

    def reload(self) -> None:
        """Re-scan the directory, replacing everything currently held."""
        if not self.directory.is_dir():
            raise ConfigurationError(f"key directory {self.directory.name!r} is not a directory")
        keys: Dict[str, PublicKey] = {}
        for path in sorted(self.directory.iterdir()):
            if path.suffix not in self.suffixes or not path.is_file():
                continue
            try:
                public_key = load_pem_public_key(path.read_bytes())
            except ConfigurationError as exc:
                raise ConfigurationError(f"{path.name}: {exc}") from None
            keys[compute_key_id(public_key)] = public_key
        self._keys = keys

    def resolve(self, key_id: str) -> Optional[PublicKey]:
        return self._keys.get(key_id)

    def key_ids(self) -> FrozenSet[str]:
        """The fingerprints loaded from disk at the last scan."""
        return frozenset(self._keys)

    def __len__(self) -> int:
        return len(self._keys)
