"""Cryptographic primitives: canonical serialization, the hash chain, Merkle trees
and signatures.

Two tiers, deliberately:

* **Eager** - ``canonical``, ``hashing`` and ``merkle`` depend on nothing outside the
  standard library, so they are imported directly.
* **Lazy** - ``signatures`` and ``keys`` need ``cryptography``, which is the optional
  ``signatures`` extra. They are resolved on first attribute access through the module
  ``__getattr__`` below, so ``import signledger.crypto`` never drags ``cryptography``
  into a base install.

The laziness is not a micro-optimisation. ``pip install signledger`` does not install
``cryptography``; an eager re-export here would make the *entire* package unimportable
for every user who did not ask for signing.
"""

from typing import Any, List

from signledger.crypto.canonical import (
    MAX_ENTRY_BYTES,
    MAX_NESTING_DEPTH,
    canonicalize,
    entry_preimage,
)
from signledger.crypto.hashing import (
    DEFAULT_ALGORITHM,
    SUPPORTED_ALGORITHMS,
    HashChain,
    genesis_hash,
)
from signledger.crypto.merkle import (
    LEAF_PREFIX,
    NODE_PREFIX,
    ROOT_PREFIX,
    ConsistencyProof,
    InclusionProof,
    MerkleTree,
    ProofStep,
    expected_path_length,
    leaf_hash,
    node_hash,
    verify_consistency,
    verify_inclusion,
)

#: Attribute name -> module it lives in, for the optional-dependency surface.
_LAZY_EXPORTS = {
    "ECDSASigner": "signledger.crypto.signatures",
    "Ed25519Signer": "signledger.crypto.signatures",
    "MIN_RSA_KEY_SIZE": "signledger.crypto.signatures",
    "PrivateKey": "signledger.crypto.signatures",
    "PublicKey": "signledger.crypto.signatures",
    "RSASigner": "signledger.crypto.signatures",
    "Signer": "signledger.crypto.signatures",
    "compute_key_id": "signledger.crypto.signatures",
    "get_private_key_pem": "signledger.crypto.signatures",
    "load_pem_private_key": "signledger.crypto.signatures",
    "load_pem_public_key": "signledger.crypto.signatures",
    "signing_payload": "signledger.crypto.signatures",
    "verify": "signledger.crypto.signatures",
    "DEFAULT_KEY_SUFFIXES": "signledger.crypto.keys",
    "FileKeyResolver": "signledger.crypto.keys",
    "InMemoryKeyResolver": "signledger.crypto.keys",
    "KeyResolver": "signledger.crypto.keys",
}


def __getattr__(name: str) -> Any:
    """Resolve the optional-dependency exports on first access (PEP 562).

    Unknown names still raise ``AttributeError``: a ``__getattr__`` that answered
    everything would turn every typo into a confusing failure somewhere else.
    """
    try:
        module_name = _LAZY_EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None

    import importlib

    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value  # cache, so this costs one lookup per process
    return value


def __dir__() -> List[str]:
    """Make the lazy names discoverable by ``dir()`` and tab completion."""
    return sorted(set(globals()) | set(_LAZY_EXPORTS))


__all__ = [
    "DEFAULT_ALGORITHM",
    "DEFAULT_KEY_SUFFIXES",
    "LEAF_PREFIX",
    "MAX_ENTRY_BYTES",
    "MAX_NESTING_DEPTH",
    "MIN_RSA_KEY_SIZE",
    "NODE_PREFIX",
    "ROOT_PREFIX",
    "SUPPORTED_ALGORITHMS",
    "ConsistencyProof",
    "ECDSASigner",
    "Ed25519Signer",
    "FileKeyResolver",
    "HashChain",
    "InMemoryKeyResolver",
    "InclusionProof",
    "KeyResolver",
    "MerkleTree",
    "PrivateKey",
    "ProofStep",
    "PublicKey",
    "RSASigner",
    "Signer",
    "canonicalize",
    "compute_key_id",
    "entry_preimage",
    "expected_path_length",
    "genesis_hash",
    "get_private_key_pem",
    "leaf_hash",
    "load_pem_private_key",
    "load_pem_public_key",
    "node_hash",
    "signing_payload",
    "verify",
    "verify_consistency",
    "verify_inclusion",
]
