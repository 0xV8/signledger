"""RFC 6962 Merkle tree (architecture contract, section 7)."""

import hashlib
import hmac
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

from signledger.core.exceptions import ConfigurationError, ValidationError
from signledger.crypto.hashing import DEFAULT_ALGORITHM, SUPPORTED_ALGORITHMS

__all__ = [
    "LEAF_PREFIX",
    "NODE_PREFIX",
    "ROOT_PREFIX",
    "ConsistencyProof",
    "InclusionProof",
    "MerkleTree",
    "ProofStep",
    "expected_path_length",
    "leaf_hash",
    "node_hash",
    "verify_consistency",
    "verify_inclusion",
]

#: Domain tag for a leaf. RFC 6962 section 2.1.
LEAF_PREFIX = b"\x00"

#: Domain tag for an interior node. RFC 6962 section 2.1.
NODE_PREFIX = b"\x01"

#: Domain tag for the count-bound root. Contract section 7: "leaf count is bound into the
#: root". This is a SignLedger addition on top of RFC 6962, whose MTH commits to the leaf
#: count only implicitly through the tree shape - and not at all for a one-leaf tree,
#: where the MTH *is* a leaf digest.
ROOT_PREFIX = b"\x02"

#: Width of the big-endian leaf count in the root preimage.
_COUNT_BYTES = 8


def _digest(algorithm: str, payload: bytes) -> bytes:
    return hashlib.new(algorithm, payload).digest()


def _leaf_digest(algorithm: str, data_item: bytes) -> bytes:
    return _digest(algorithm, LEAF_PREFIX + data_item)


def _node_digest(algorithm: str, left: bytes, right: bytes) -> bytes:
    return _digest(algorithm, NODE_PREFIX + left + right)


def _digest_size(algorithm: str) -> int:
    return hashlib.new(algorithm).digest_size


def _unhex(value: str, width: int) -> Optional[bytes]:
    """Decode an attacker-supplied hex digest, or ``None`` if it is not one."""
    try:
        raw = bytes.fromhex(value)
    except (TypeError, ValueError):
        return None
    return raw if len(raw) == width else None


def _check_data_item(data_item: bytes) -> bytes:
    """Leaves are raw bytes only.

    Accepting ``str`` would put an implicit encoding choice inside the digest, and
    accepting ``bytearray`` would let a caller mutate a leaf after the tree was built.
    """
    if not isinstance(data_item, bytes):
        raise ValidationError(
            f"a Merkle leaf must be bytes, got {type(data_item).__name__}"
        )
    return data_item


def _check_leaf_index(tree_size: int, leaf_index: int) -> None:
    if tree_size < 1:
        raise ValidationError(f"a tree of {tree_size} leaves has no leaf to prove")
    if not 0 <= leaf_index < tree_size:
        raise ValidationError(
            f"leaf_index {leaf_index} is outside a tree of {tree_size} leaves"
        )


def _same_digest(left: str, right: str) -> bool:
    """Constant-time hex comparison (contract section 5 rule 7).

    Compares UTF-8 bytes rather than ``str``: ``compare_digest`` raises on a non-ASCII
    ``str``, and a tampered value must make a verifier answer False, never crash it.
    """
    return hmac.compare_digest(left.encode("utf-8"), right.encode("utf-8"))


def _check_algorithm(algorithm: str) -> str:
    if algorithm not in SUPPORTED_ALGORITHMS:
        raise ConfigurationError(
            f"unsupported hash algorithm {algorithm!r}; "
            f"choose one of {sorted(SUPPORTED_ALGORITHMS)}"
        )
    return algorithm


def _largest_power_of_two_below(n: int) -> int:
    """Return the largest power of two **strictly** less than ``n`` (``n >= 2``)."""
    return 1 << ((n - 1).bit_length() - 1)


def _bind_count(algorithm: str, leaf_count: int, tree_hash: bytes) -> bytes:
    """Return ``H(0x02 || uint64_be(leaf_count) || mth)``.

    Without this an inclusion proof says only "this item is *a* leaf of *a* tree with
    this shape"; with it the root a verifier is handed also pins how many leaves that
    tree has, so a proof cut from one tree size cannot be replayed against another.
    """
    return _digest(
        algorithm, ROOT_PREFIX + leaf_count.to_bytes(_COUNT_BYTES, "big") + tree_hash
    )


def _mth(algorithm: str, leaf_digests: Sequence[bytes]) -> bytes:
    """RFC 6962 section 2.1 Merkle Tree Hash over already-hashed leaves."""
    n = len(leaf_digests)
    if n == 0:
        return _digest(algorithm, b"")
    if n == 1:
        return leaf_digests[0]
    k = _largest_power_of_two_below(n)
    return _node_digest(
        algorithm, _mth(algorithm, leaf_digests[:k]), _mth(algorithm, leaf_digests[k:])
    )


def leaf_hash(data_item: bytes, algorithm: str = DEFAULT_ALGORITHM) -> str:
    """Return the hex leaf digest ``H(0x00 || data_item)``."""
    return _leaf_digest(_check_algorithm(algorithm), _check_data_item(data_item)).hex()


def node_hash(left: str, right: str, algorithm: str = DEFAULT_ALGORITHM) -> str:
    """Return the hex interior digest ``H(0x01 || left || right)``.

    ``left`` and ``right`` are hex, but they are hashed as **raw digest bytes**: hashing
    the hex text instead would diverge from every other RFC 6962 implementation.
    """
    return _node_digest(
        _check_algorithm(algorithm), bytes.fromhex(left), bytes.fromhex(right)
    ).hex()


@dataclass(frozen=True)
class ProofStep:
    """One level of an audit path: the sibling subtree digest and which side it is on."""

    sibling_hash: str
    sibling_is_left: bool


@dataclass(frozen=True)
class InclusionProof:
    """An RFC 6962 audit path.

    Deliberately carries **no root and no leaf digest**. Both are the verifier's job:
    the root comes from the caller as trusted input, and the leaf digest is re-derived
    from the data item. v1 stored both on the proof and checked the proof against
    itself, which authenticated every forgery.
    """

    leaf_index: int
    tree_size: int
    path: Tuple[ProofStep, ...]
    algorithm: str


@dataclass(frozen=True)
class ConsistencyProof:
    """An RFC 6962 consistency proof: the old tree is a prefix of the new one.

    Carries no roots, for the same reason :class:`InclusionProof` carries none - both
    roots are the caller's trusted input. ``path`` is the RFC 6962 SUBPROOF array with
    the old tree hash materialised when ``old_size`` is a power of two, so the proof can
    be checked against the two count-bound roots and nothing else.
    """

    old_size: int
    new_size: int
    path: Tuple[str, ...]
    algorithm: str


def _audit_path(
    algorithm: str, leaf_digests: Sequence[bytes], index: int
) -> Tuple[ProofStep, ...]:
    """RFC 6962 section 2.1.1 PATH(index, leaf_digests), ordered leaf-upwards."""
    n = len(leaf_digests)
    if n == 1:
        return ()
    k = _largest_power_of_two_below(n)
    if index < k:
        sibling = _mth(algorithm, leaf_digests[k:])
        return _audit_path(algorithm, leaf_digests[:k], index) + (
            ProofStep(sibling.hex(), sibling_is_left=False),
        )
    sibling = _mth(algorithm, leaf_digests[:k])
    return _audit_path(algorithm, leaf_digests[k:], index - k) + (
        ProofStep(sibling.hex(), sibling_is_left=True),
    )


def _audit_directions(tree_size: int, leaf_index: int) -> Tuple[bool, ...]:
    """The side each audit-path sibling must sit on, derived from the tree shape alone.

    This is what binds ``leaf_index`` to the proof. Verification recomputes the sequence
    from ``(tree_size, leaf_index)`` and refuses a path that does not match it, so the
    index cannot be edited to relocate an item within the tree.
    """
    if tree_size == 1:
        return ()
    k = _largest_power_of_two_below(tree_size)
    if leaf_index < k:
        return _audit_directions(k, leaf_index) + (False,)
    return _audit_directions(tree_size - k, leaf_index - k) + (True,)


def expected_path_length(tree_size: int, leaf_index: int) -> int:
    """Return how many audit-path steps a proof for ``leaf_index`` must carry.

    For a full tree this is ``ceil(log2(tree_size))`` for every leaf. For a ragged tree
    a promoted leaf sits above the deepest level and needs fewer steps, so the contract's
    ``ceil(log2(n))`` is the *longest* path rather than every path.
    """
    _check_leaf_index(tree_size, leaf_index)
    return len(_audit_directions(tree_size, leaf_index))


def _subproof(
    algorithm: str, leaf_digests: Sequence[bytes], m: int, complete: bool
) -> Tuple[bytes, ...]:
    """RFC 6962 section 2.1.2 SUBPROOF(m, leaf_digests, complete)."""
    n = len(leaf_digests)
    if m == n:
        # ``complete`` means the old tree is exactly this subtree, so the verifier
        # already holds its hash and the RFC omits it.
        return () if complete else (_mth(algorithm, leaf_digests),)
    k = _largest_power_of_two_below(n)
    if m <= k:
        return _subproof(algorithm, leaf_digests[:k], m, complete) + (
            _mth(algorithm, leaf_digests[k:]),
        )
    return _subproof(algorithm, leaf_digests[k:], m - k, False) + (
        _mth(algorithm, leaf_digests[:k]),
    )


def _is_power_of_two(n: int) -> bool:
    return n & (n - 1) == 0


class MerkleTree:
    """An immutable RFC 6962 Merkle tree over a fixed list of data items."""

    def __init__(
        self, leaves: Iterable[bytes], algorithm: str = DEFAULT_ALGORITHM
    ) -> None:
        self._algorithm = _check_algorithm(algorithm)
        self._leaf_digests: Tuple[bytes, ...] = tuple(
            _leaf_digest(self._algorithm, _check_data_item(item)) for item in leaves
        )
        # The tree is immutable, so the hash is computed once here rather than on every
        # property access.
        self._tree_hash = _mth(self._algorithm, self._leaf_digests)

    @property
    def algorithm(self) -> str:
        """The digest algorithm every node in this tree is built with."""
        return self._algorithm

    @property
    def leaf_count(self) -> int:
        """The number of data items in the tree."""
        return len(self._leaf_digests)

    @property
    def merkle_tree_hash(self) -> str:
        """The bare RFC 6962 MTH, for interoperability with other implementations.

        Use :attr:`root` for anything security-bearing: this value does not commit to
        the leaf count.
        """
        return self._tree_hash.hex()

    @property
    def root(self) -> str:
        """The SignLedger root: the RFC 6962 MTH with the leaf count bound in."""
        return _bind_count(self._algorithm, self.leaf_count, self._tree_hash).hex()

    def generate_inclusion_proof(self, leaf_index: int) -> InclusionProof:
        """Return the audit path proving the item at ``leaf_index`` is in this tree."""
        _check_leaf_index(self.leaf_count, leaf_index)
        return InclusionProof(
            leaf_index=leaf_index,
            tree_size=self.leaf_count,
            path=_audit_path(self._algorithm, self._leaf_digests, leaf_index),
            algorithm=self._algorithm,
        )

    def generate_consistency_proof(self, old_size: int) -> ConsistencyProof:
        """Return the proof that the first ``old_size`` leaves are a prefix of this tree.

        ``old_size == 0`` is refused rather than answered with an empty proof: nothing
        can be demonstrated about an empty tree, and an empty list is exactly what v1
        returned for every call.
        """
        if old_size < 1:
            raise ValidationError(
                f"old_size must be at least 1 to prove anything, got {old_size}"
            )
        if old_size > self.leaf_count:
            raise ValidationError(
                f"old_size {old_size} exceeds the tree's {self.leaf_count} leaves"
            )
        path: Tuple[bytes, ...] = ()
        if old_size < self.leaf_count:
            path = _subproof(self._algorithm, self._leaf_digests, old_size, True)
            if _is_power_of_two(old_size):
                # RFC 9162 section 2.1.4.2 step 2, done here rather than in the verifier:
                # a count-bound root cannot be unwrapped back into this tree hash.
                path = (_mth(self._algorithm, self._leaf_digests[:old_size]), *path)
        return ConsistencyProof(
            old_size=old_size,
            new_size=self.leaf_count,
            path=tuple(node.hex() for node in path),
            algorithm=self._algorithm,
        )


def verify_inclusion(
    proof: InclusionProof, data_item: bytes, expected_root: str
) -> bool:
    """Return whether ``data_item`` is the leaf ``proof`` claims, under ``expected_root``.

    ``expected_root`` is a **required caller argument** and is the only trusted input.
    Nothing is read off ``proof`` that the proof could use to authenticate itself: the
    leaf digest is recomputed from ``data_item``, and the root is recomputed from that
    leaf and compared against what the caller already believed.

    Malformed *proofs* answer False rather than raising: a proof is attacker-controlled
    input, and a verifier that crashes on a hostile path is a denial of service. A
    malformed ``expected_root`` or ``data_item`` does raise, because those are the
    caller's own arguments and silently answering "not included" would hide the bug.
    """
    if not isinstance(proof, InclusionProof):
        raise ValidationError(
            f"proof must be an InclusionProof, got {type(proof).__name__}"
        )
    if not isinstance(data_item, bytes):
        raise ValidationError(
            f"data_item must be bytes, got {type(data_item).__name__}"
        )
    if not isinstance(expected_root, str):
        raise ValidationError(
            f"expected_root must be a hex str, got {type(expected_root).__name__}"
        )
    if not expected_root:
        raise ValidationError("expected_root must not be empty")

    algorithm = proof.algorithm
    if algorithm not in SUPPORTED_ALGORITHMS:
        return False
    if proof.tree_size < 1:
        return False
    if not 0 <= proof.leaf_index < proof.tree_size:
        return False

    # The shape a proof for this (tree_size, leaf_index) is *required* to have. Checking
    # it is what stops leaf_index being edited to relocate an item, and what rejects
    # truncated and extended paths instead of letting them chain to some other root.
    # Comparing the whole direction tuple pins the path *length* and every direction at
    # once: one step per level that has a sibling - ceil(log2(tree_size)) of them for a
    # full tree - each on the side the tree shape requires. A truncated, extended,
    # re-sided or relocated path is rejected here, before a single digest is computed.
    directions = _audit_directions(proof.tree_size, proof.leaf_index)
    if not all(isinstance(step, ProofStep) for step in proof.path):
        return False
    if tuple(step.sibling_is_left for step in proof.path) != directions:
        return False

    width = _digest_size(algorithm)
    node = _leaf_digest(algorithm, data_item)
    for step, sibling_is_left in zip(proof.path, directions):
        sibling = _unhex(step.sibling_hash, width)
        if sibling is None:
            return False
        node = (
            _node_digest(algorithm, sibling, node)
            if sibling_is_left
            else _node_digest(algorithm, node, sibling)
        )
    computed = _bind_count(algorithm, proof.tree_size, node).hex()
    return _same_digest(computed, expected_root)


def _replay_consistency(
    algorithm: str, path: Sequence[bytes], old_size: int, new_size: int
) -> Optional[Tuple[bytes, bytes]]:
    """RFC 9162 section 2.1.4.2 steps 3-7: rebuild both tree hashes from the proof.

    Returns ``(old_tree_hash, new_tree_hash)``, or ``None`` if the proof is structurally
    unusable. Deriving both ends from the proof - rather than walking one and trusting
    the other - is what makes this a verifier and not a formality.
    """
    if not path or old_size < 1:
        # `old_size < 1` is a LIVENESS guard, not only a correctness one, and it is
        # repeated here deliberately even though `verify_consistency` already checks it.
        # At old_size == 0, `fn` is -1, which is the fixed point of arithmetic right shift:
        # -1 is odd so the loop below continues, and -1 >> 1 is -1 again, forever. A
        # mutation sweep found two single-character edits to the caller's guard that turn
        # an attacker-supplied ConsistencyProof into a wedged verifier thread rather than a
        # False. One guard between untrusted input and a hang is not enough for the only
        # loop in this file whose bound comes from its arguments.
        return None
    fn, sn = old_size - 1, new_size - 1
    while fn & 1:
        fn >>= 1
        sn >>= 1
    old_hash = new_hash = path[0]
    for node in path[1:]:
        if sn == 0:
            return None
        if fn & 1 or fn == sn:
            old_hash = _node_digest(algorithm, node, old_hash)
            new_hash = _node_digest(algorithm, node, new_hash)
            # `fn != 0` was here and is provably dead: this branch is entered only when
            # `fn & 1 or fn == sn`, and `sn == 0` has already returned above, so `fn` is
            # never 0 at this point; and right-shifting a nonzero positive int always
            # reaches an odd value before 0. Verified exhaustively over every valid
            # consistency proof up to 32 leaves plus 20,000 hostile size pairs - both
            # spellings agree on every one. Dead code in a verifier is a line no test can
            # ever justify, so it is gone rather than covered.
            while not fn & 1:
                fn >>= 1
                sn >>= 1
        else:
            new_hash = _node_digest(algorithm, new_hash, node)
        fn >>= 1
        sn >>= 1
    if sn != 0:
        return None
    return old_hash, new_hash


def verify_consistency(
    proof: ConsistencyProof, old_root: str, new_root: str
) -> bool:
    """Return whether ``proof`` shows the ``old_root`` tree is a prefix of ``new_root``.

    Both roots are **required caller arguments**; neither is read off the proof. An
    empty path never proves anything about a tree that grew.
    """
    if not isinstance(proof, ConsistencyProof):
        raise ValidationError(
            f"proof must be a ConsistencyProof, got {type(proof).__name__}"
        )
    for name, value in (("old_root", old_root), ("new_root", new_root)):
        if not isinstance(value, str):
            raise ValidationError(
                f"{name} must be a hex str, got {type(value).__name__}"
            )
        if not value:
            raise ValidationError(f"{name} must not be empty")

    algorithm = proof.algorithm
    if algorithm not in SUPPORTED_ALGORITHMS:
        return False
    old_size, new_size = proof.old_size, proof.new_size
    if old_size < 1 or new_size < old_size:
        return False

    if old_size == new_size:
        # Nothing grew, so the proof must be empty and the two roots must be the one root.
        return not proof.path and _same_digest(old_root, new_root)

    width = _digest_size(algorithm)
    path = []
    for element in proof.path:
        node = _unhex(element, width)
        if node is None:
            return False
        path.append(node)

    replayed = _replay_consistency(algorithm, path, old_size, new_size)
    if replayed is None:
        return False
    old_hash, new_hash = replayed
    return _same_digest(
        _bind_count(algorithm, old_size, old_hash).hex(), old_root
    ) and _same_digest(_bind_count(algorithm, new_size, new_hash).hex(), new_root)
