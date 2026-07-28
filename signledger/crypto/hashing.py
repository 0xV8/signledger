"""The hash chain - digests over SLCS/1 preimages.

A bare hash chain is *not* tamper-evidence on its own: SHA-256 is public, so anyone with
write access to the store can edit a row and re-derive every digest after it. What this
module provides is the deterministic, versioned foundation that signatures (section 6)
and anchored checkpoints (section 8) build the real guarantee on top of.

``calculate_hash`` dispatches on ``entry.format_version``. Verification must keep working
against every historical format forever, so the encoder is chosen per entry rather than
assumed; a v3 encoder is added by registering it in ``_PREIMAGE_ENCODERS``, and old
ledgers keep verifying under the rules they were written with.
"""

import hashlib
import hmac
from datetime import datetime
from typing import TYPE_CHECKING, Callable, Dict

from signledger.core.exceptions import ConfigurationError, ValidationError
from signledger.crypto.canonical import canonicalize, entry_preimage

if TYPE_CHECKING:  # pragma: no cover
    # Import-time only. ``signledger.core.entry`` imports this package for the
    # canonicalizer, so importing Entry for real here would close the loop and break
    # every ``signledger.crypto.*`` import - the exact circular-import trap v1 fell into.
    from signledger.core.entry import Entry

__all__ = [
    "SUPPORTED_ALGORITHMS",
    "DEFAULT_ALGORITHM",
    "HashChain",
    "genesis_hash",
]

#: Digest algorithms this build accepts. Anything else is refused rather than guessed at.
SUPPORTED_ALGORITHMS = frozenset({"sha256", "sha512", "sha3_256", "sha3_512"})

DEFAULT_ALGORITHM = "sha256"

#: Domain tag that separates a genesis preimage from an entry preimage. No Entry preimage
#: carries a ``type`` key, so the two encodings cannot collide.
_GENESIS_TYPE = "signledger/genesis"

#: Preimage encoder per hash format version. Historical versions stay registered forever.
_PREIMAGE_ENCODERS: Dict[int, Callable[["Entry"], bytes]] = {
    2: entry_preimage,
}


class HashChain:
    """Computes and verifies the digest of an :class:`~signledger.core.entry.Entry`."""

    def __init__(self, algorithm: str = DEFAULT_ALGORITHM) -> None:
        if algorithm not in SUPPORTED_ALGORITHMS:
            raise ConfigurationError(
                f"unsupported hash algorithm {algorithm!r}; "
                f"choose one of {sorted(SUPPORTED_ALGORITHMS)}"
            )
        self.algorithm = algorithm

    def calculate_hash(self, entry: "Entry") -> str:
        """Return the hex digest of ``entry``'s preimage under its own format version."""
        encoder = _PREIMAGE_ENCODERS.get(entry.format_version)
        if encoder is None:
            raise ValidationError(
                f"format_version {entry.format_version} has no preimage encoder in this "
                f"build; supported: {sorted(_PREIMAGE_ENCODERS)}"
            )
        return _digest(self.algorithm, encoder(entry))

    def verify_hash(self, entry: "Entry") -> bool:
        """Return whether ``entry.hash`` is the digest of ``entry``.

        The comparison runs over UTF-8 bytes rather than ``str``: ``compare_digest``
        raises ``TypeError`` on a non-ASCII ``str``, and a tampered row must make a
        verifier answer False, not crash it.
        """
        expected = self.calculate_hash(entry).encode("utf-8")
        return hmac.compare_digest(expected, entry.hash.encode("utf-8"))


def genesis_hash(
    ledger_id: str,
    created_at: datetime,
    format_version: int = 2,
    *,
    algorithm: str = DEFAULT_ALGORITHM,
) -> str:
    """Return the anchor digest that entry 0 links back to.

    Verification asserts that entry 0's ``previous_hash`` equals this value, which is
    what stops an attacker replacing the prefix of a chain with one of their own.

    ``algorithm`` is keyword-only with a default, so the three-positional-argument form
    in the contract keeps working; pass it to match a ledger that is not using SHA-256,
    otherwise entry 0's ``previous_hash`` would be a different width from every other
    link in the chain.
    """
    if not ledger_id:
        raise ValidationError("ledger_id must not be empty")
    preimage = canonicalize(
        {
            "created_at": created_at,
            "format_version": format_version,
            "ledger_id": ledger_id,
            "type": _GENESIS_TYPE,
        }
    )
    return _digest(algorithm, preimage)


def _digest(algorithm: str, payload: bytes) -> str:
    if algorithm not in SUPPORTED_ALGORITHMS:
        raise ConfigurationError(
            f"unsupported hash algorithm {algorithm!r}; "
            f"choose one of {sorted(SUPPORTED_ALGORITHMS)}"
        )
    return hashlib.new(algorithm, payload).hexdigest()
