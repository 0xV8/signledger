"""Chain verification (architecture contract, section 5)."""

import hmac
from dataclasses import dataclass
from enum import Enum
from typing import Any, FrozenSet, List, Optional, Tuple

from signledger.backends.base import CAP_APPEND_ONLY_ENFORCED
from signledger.core.entry import CURRENT_FORMAT_VERSION
from signledger.core.exceptions import SignLedgerError, ValidationError

__all__ = [
    "AssuranceLevel",
    "FailureKind",
    "VerificationFailure",
    "VerificationReport",
    "verify_chain",
]


class AssuranceLevel(str, Enum):
    """How much a given deployment's evidence is actually worth."""

    NONE = "none"
    CHAIN_ONLY = "chain_only"
    SIGNED = "signed"
    ANCHORED = "anchored"
    WORM = "worm"


class FailureKind(str, Enum):
    """The kinds of damage verification can report."""

    BROKEN_LINK = "broken_link"
    BAD_HASH = "bad_hash"
    BAD_SIGNATURE = "bad_signature"
    MISSING_SIGNATURE = "missing_signature"
    SEQUENCE_GAP = "sequence_gap"
    GENESIS_MISMATCH = "genesis_mismatch"
    HEAD_MISMATCH = "head_mismatch"
    COUNT_MISMATCH = "count_mismatch"
    UNKNOWN_KEY = "unknown_key"
    FORMAT_UNSUPPORTED = "format_unsupported"

    # A checkpoint is the only thing that sees a truncated tail, which makes it the next
    # thing an attacker goes for. It was signed and never verified, so these exist to say
    # precisely which part of it did not hold up rather than collapsing into HEAD_MISMATCH.
    CHECKPOINT_UNSIGNED = "checkpoint_unsigned"
    CHECKPOINT_UNKNOWN_KEY = "checkpoint_unknown_key"
    CHECKPOINT_BAD_SIGNATURE = "checkpoint_bad_signature"
    CHECKPOINT_LEDGER_MISMATCH = "checkpoint_ledger_mismatch"
    CHECKPOINT_FORMAT_UNSUPPORTED = "checkpoint_format_unsupported"
    CHECKPOINT_MERKLE_MISMATCH = "checkpoint_merkle_mismatch"


@dataclass(frozen=True)
class VerificationFailure:
    """One piece of damage, located precisely enough to act on."""

    kind: FailureKind
    sequence: Optional[int]
    entry_id: Optional[str]
    expected: Optional[str]
    actual: Optional[str]
    message: str


@dataclass(frozen=True)
class VerificationReport:
    """The result of a verification run. Never an exception, always a report."""

    ok: bool
    entries_verified: int
    first_sequence: Optional[int]
    last_sequence: Optional[int]
    failures: Tuple[VerificationFailure, ...]
    assurance: AssuranceLevel
    checkpoint_used: Optional[Any]


def _same_digest(left: str, right: str) -> bool:
    """Contract section 5 rule 7: every digest comparison is constant-time.

    The comparison runs over UTF-8 bytes because ``compare_digest`` raises ``TypeError``
    on a non-ASCII ``str``, and a tampered row has to make a verifier answer *no* rather
    than crash it.
    """
    return hmac.compare_digest(left.encode("utf-8"), right.encode("utf-8"))


def _check_signature(entry: Any, key_resolver: Any) -> Optional[VerificationFailure]:
    """Return the failure this entry's signature earns, or ``None`` if it verifies.

    Every branch fails **closed**. A missing signature is not a pass, an unresolvable
    ``key_id`` is not a pass, and a digest that cannot be framed into a signing payload is
    not a pass - v1 answered True for all three.
    """
    if entry.signature is None:
        return VerificationFailure(
            kind=FailureKind.MISSING_SIGNATURE,
            sequence=entry.sequence,
            entry_id=entry.entry_id,
            expected="a signature",
            actual=None,
            message=f"sequence {entry.sequence} carries no signature",
        )
    if entry.key_id is None:
        return VerificationFailure(
            kind=FailureKind.UNKNOWN_KEY,
            sequence=entry.sequence,
            entry_id=entry.entry_id,
            expected="a key_id",
            actual=None,
            message=f"sequence {entry.sequence} is signed but names no key",
        )

    public_key = None if key_resolver is None else key_resolver.resolve(entry.key_id)
    if public_key is None:
        return VerificationFailure(
            kind=FailureKind.UNKNOWN_KEY,
            sequence=entry.sequence,
            entry_id=entry.entry_id,
            expected=entry.key_id,
            actual=None,
            message=f"sequence {entry.sequence} names a key this resolver cannot supply",
        )

    # Deferred: ``signledger.crypto.signatures`` imports ``cryptography``, the optional
    # ``signatures`` extra. A module-scope import here would load it during every
    # `import signledger`, because the ledger imports this module. Control only reaches
    # this point when an entry actually carries a signature to check.
    from signledger.crypto.signatures import signing_payload, verify

    try:
        payload = signing_payload(entry.ledger_id, entry.hash)
    except ValidationError:
        # A tampered digest that is not even hex. No signature can cover it, and raising
        # here would let an attacker turn verification into a crash.
        return VerificationFailure(
            kind=FailureKind.BAD_SIGNATURE,
            sequence=entry.sequence,
            entry_id=entry.entry_id,
            expected=entry.key_id,
            actual=None,
            message=(
                f"sequence {entry.sequence} has a digest no signing payload can be "
                f"framed from"
            ),
        )

    if not verify(payload, entry.signature, public_key):
        return VerificationFailure(
            kind=FailureKind.BAD_SIGNATURE,
            sequence=entry.sequence,
            entry_id=entry.entry_id,
            expected=entry.key_id,
            actual=None,
            message=f"sequence {entry.sequence} has a signature that does not verify",
        )
    return None


def _check_checkpoint_authenticity(
    checkpoint: Any,
    *,
    ledger_id: Optional[str],
    enable_signatures: bool,
    key_resolver: Any,
) -> List[VerificationFailure]:
    """Establish that this checkpoint is *this* ledger's, and that nobody wrote it.

    Until this existed a checkpoint was believed on sight. Every field it carries was
    signed at ``Ledger.checkpoint`` and then read back without ever being checked, so
    rewriting the anchor file - or, cheaper, one ``UPDATE`` on a checkpoints table that
    carries no append-only trigger - restored ``ok=True`` on a chain with its tail cut
    off, and *raised* the reported assurance while doing it.

    The order matters. A checkpoint describing another ledger, or written in a format
    this build cannot read, is rejected before its contents are compared against
    anything: comparing them would be answering a question about the wrong chain.
    """
    failures: List[VerificationFailure] = []

    if ledger_id is not None and checkpoint.ledger_id != ledger_id:
        failures.append(
            VerificationFailure(
                kind=FailureKind.CHECKPOINT_LEDGER_MISMATCH,
                sequence=checkpoint.sequence,
                entry_id=None,
                expected=ledger_id,
                actual=checkpoint.ledger_id,
                message=(
                    "the checkpoint describes a different ledger; it is not evidence "
                    "about this chain"
                ),
            )
        )

    if checkpoint.format_version > CURRENT_FORMAT_VERSION:
        failures.append(
            VerificationFailure(
                kind=FailureKind.CHECKPOINT_FORMAT_UNSUPPORTED,
                sequence=checkpoint.sequence,
                entry_id=None,
                expected=str(CURRENT_FORMAT_VERSION),
                actual=str(checkpoint.format_version),
                message=(
                    f"the checkpoint is in format_version {checkpoint.format_version}, "
                    f"which this build cannot interpret"
                ),
            )
        )

    if failures or not enable_signatures:
        # With signing off there is nothing to verify and saying so is honest: assurance
        # tops out at CHAIN_ONLY, so no caller can mistake this for an authenticated
        # checkpoint.
        return failures

    if checkpoint.signature is None or checkpoint.key_id is None:
        return [
            VerificationFailure(
                kind=FailureKind.CHECKPOINT_UNSIGNED,
                sequence=checkpoint.sequence,
                entry_id=None,
                expected="a signature and a key_id",
                actual=None,
                message=(
                    f"the checkpoint at sequence {checkpoint.sequence} carries no "
                    f"signature, and this ledger signs its checkpoints"
                ),
            )
        ]

    public_key = None if key_resolver is None else key_resolver.resolve(checkpoint.key_id)
    if public_key is None:
        return [
            VerificationFailure(
                kind=FailureKind.CHECKPOINT_UNKNOWN_KEY,
                sequence=checkpoint.sequence,
                entry_id=None,
                expected=checkpoint.key_id,
                actual=None,
                message=(
                    f"the checkpoint at sequence {checkpoint.sequence} names a key this "
                    f"resolver cannot supply"
                ),
            )
        ]

    # Deferred for the same reason as the entry path: importing ``cryptography`` at module
    # scope would load the optional extra on every ``import signledger``.
    from signledger.anchor.sinks import checkpoint_signing_payload
    from signledger.crypto.signatures import verify

    try:
        payload = checkpoint_signing_payload(checkpoint)
    except (SignLedgerError, TypeError, ValueError):
        # A doctored field that cannot even be canonicalised - an anchor file is JSON, and
        # JSON carries a lone surrogate happily while UTF-8 cannot encode one. Raising here
        # would let an attacker turn verification into a crash, which denies the answer as
        # effectively as a forged pass.
        #
        # ``SignLedgerError`` and not ``ValidationError``: canonicalisation raises
        # ``CanonicalizationError``, which is a sibling of ``ValidationError`` under
        # ``SignLedgerError``, not a subclass of it. Catching the narrower one looked
        # careful and caught nothing.
        return [
            VerificationFailure(
                kind=FailureKind.CHECKPOINT_BAD_SIGNATURE,
                sequence=checkpoint.sequence,
                entry_id=None,
                expected=checkpoint.key_id,
                actual=None,
                message=(
                    f"the checkpoint at sequence {checkpoint.sequence} holds a field no "
                    f"signing payload can be framed from"
                ),
            )
        ]

    if not verify(payload, checkpoint.signature, public_key):
        return [
            VerificationFailure(
                kind=FailureKind.CHECKPOINT_BAD_SIGNATURE,
                sequence=checkpoint.sequence,
                entry_id=None,
                expected=checkpoint.key_id,
                actual=None,
                message=(
                    f"the checkpoint at sequence {checkpoint.sequence} has a signature "
                    f"that does not verify"
                ),
            )
        ]
    return []


def _check_merkle_root(backend: Any, checkpoint: Any, chain: Any) -> List[VerificationFailure]:
    """Recompute the tree the checkpoint claims to summarise.

    ``merkle_root`` was computed, signed and published, and then never recomputed - so a
    forged checkpoint could carry any 32 bytes at all. The window is the entries up to and
    including ``checkpoint.sequence``: ``end_seq`` is inclusive, and the root over that
    prefix is stable as the chain grows past it, which is what lets this run against a
    live ledger rather than only a frozen one.
    """
    from signledger.crypto.merkle import MerkleTree

    try:
        leaves = [
            bytes.fromhex(entry.hash)
            for entry in backend.iter_entries(end_seq=checkpoint.sequence)
        ]
    except ValueError:
        # A tampered digest that is not hex, somewhere at or below the checkpointed
        # sequence. The entry-level BAD_HASH check reports that damage precisely, and a
        # tree cannot be built over it; reporting a Merkle mismatch as well would name a
        # symptom rather than the cause.
        return []

    # No `if not leaves` guard: this function runs only when the head check passed, which
    # means `get_by_sequence(checkpoint.sequence)` returned an entry, which means the
    # window below it is never empty. A branch no input can reach is a line no test can
    # justify, and dead code in a verifier is worse than none.
    recomputed = MerkleTree(leaves, algorithm=chain.algorithm).root
    if _same_digest(recomputed, checkpoint.merkle_root):
        return []
    return [
        VerificationFailure(
            kind=FailureKind.CHECKPOINT_MERKLE_MISMATCH,
            sequence=checkpoint.sequence,
            entry_id=None,
            expected=checkpoint.merkle_root,
            actual=recomputed,
            message=(
                f"the entries up to sequence {checkpoint.sequence} do not build the "
                f"Merkle root the checkpoint recorded for them"
            ),
        )
    ]


def _check_checkpoint(
    backend: Any,
    checkpoint: Any,
    *,
    chain: Any = None,
    ledger_id: Optional[str] = None,
    enable_signatures: bool = False,
    key_resolver: Any = None,
) -> List[VerificationFailure]:
    """Compare the store against a checkpoint taken earlier.

    This is the only mechanism that sees a **truncated tail** (contract section 8). Lop
    the last N entries off a chain and what remains is internally flawless - every link
    matches, every digest recomputes. The chain cannot know it used to be longer. A
    checkpoint can, because it recorded the head and the count at a point in time.

    Growth past the checkpoint is normal and is *not* a failure; the comparison is made
    against the entry that sat at ``checkpoint.sequence``, never against today's head.

    A checkpoint is only worth comparing against once it has been shown to be this
    ledger's and unmodified, so authenticity is established first and the rest is skipped
    when it fails - a forged checkpoint's head and count describe whatever the forger
    chose, and reporting them as mismatches would bury the reason under its symptoms.
    """
    authenticity = _check_checkpoint_authenticity(
        checkpoint,
        ledger_id=ledger_id,
        enable_signatures=enable_signatures,
        key_resolver=key_resolver,
    )
    if authenticity:
        return authenticity

    failures: List[VerificationFailure] = []

    anchored_entry = backend.get_by_sequence(checkpoint.sequence)
    if anchored_entry is None:
        failures.append(
            VerificationFailure(
                kind=FailureKind.HEAD_MISMATCH,
                sequence=checkpoint.sequence,
                entry_id=None,
                expected=checkpoint.head_hash,
                actual=None,
                message=(
                    f"the checkpoint covers sequence {checkpoint.sequence}, which is no "
                    f"longer in the store"
                ),
            )
        )
    elif not _same_digest(anchored_entry.hash, checkpoint.head_hash):
        failures.append(
            VerificationFailure(
                kind=FailureKind.HEAD_MISMATCH,
                sequence=checkpoint.sequence,
                entry_id=anchored_entry.entry_id,
                expected=checkpoint.head_hash,
                actual=anchored_entry.hash,
                message=(
                    f"sequence {checkpoint.sequence} no longer has the digest the "
                    f"checkpoint recorded for it"
                ),
            )
        )

    stored = backend.count()
    if stored < checkpoint.entry_count:
        failures.append(
            VerificationFailure(
                kind=FailureKind.COUNT_MISMATCH,
                sequence=checkpoint.sequence,
                entry_id=None,
                expected=str(checkpoint.entry_count),
                actual=str(stored),
                message=(
                    f"the checkpoint counted {checkpoint.entry_count} entries; the store "
                    f"now holds {stored}"
                ),
            )
        )

    if chain is not None and not failures:
        # Only once the head and the count agree. A truncated tail fails all three checks
        # at once, and a report listing three symptoms of one cause is harder to act on
        # than one listing the two that locate it. Where this earns its place is the case
        # the other two cannot see: a head and a count that both reconcile over a tree that
        # does not - a substituted entry, or a ``merkle_root`` somebody chose.
        failures.extend(_check_merkle_root(backend, checkpoint, chain))
    return failures


def _assurance(
    failures: List[VerificationFailure],
    *,
    enable_signatures: bool,
    anchored: bool,
    capabilities: FrozenSet[str],
) -> AssuranceLevel:
    """Derive what this deployment's evidence is actually worth.

    Computed from what was observed, never configured. Each level is earned by the one
    below it: a capability token cannot upgrade assurance that signatures never granted,
    which is why an append-only store with no signatures still reports ``CHAIN_ONLY``.
    """
    if failures:
        return AssuranceLevel.NONE
    if not enable_signatures:
        return AssuranceLevel.CHAIN_ONLY
    if not anchored:
        return AssuranceLevel.SIGNED
    if CAP_APPEND_ONLY_ENFORCED in capabilities:
        return AssuranceLevel.WORM
    return AssuranceLevel.ANCHORED


def verify_chain(
    backend: Any,
    *,
    chain: Any,
    genesis: str,
    from_sequence: Optional[int] = None,
    enable_signatures: bool = False,
    key_resolver: Any = None,
    checkpoint: Any = None,
    anchored: bool = False,
    capabilities: FrozenSet[str] = frozenset(),
    ledger_id: Optional[str] = None,
) -> VerificationReport:
    """Verify the entries the backend holds and report everything wrong with them.

    Reads always come from ``backend``. There is no cache path here on purpose: v1 served
    unverified cache hits, so on-disk tampering was invisible to the only function whose
    job was to find it.
    """
    entries = list(backend.iter_entries(start_seq=from_sequence))
    failures: List[VerificationFailure] = []

    # Seeded below from the genesis hash, or - for a partial run starting past entry 0 -
    # from the start entry's own ``previous_hash`` (contract section 5 rule 6). v1's
    # equivalent seeded it with ``None`` and raised on every healthy ledger.
    expected_previous: Optional[str] = None
    expected_sequence = 0 if from_sequence is None else from_sequence

    for index, entry in enumerate(entries):
        if entry.sequence != expected_sequence:
            failures.append(
                VerificationFailure(
                    kind=FailureKind.SEQUENCE_GAP,
                    sequence=entry.sequence,
                    entry_id=entry.entry_id,
                    expected=str(expected_sequence),
                    actual=str(entry.sequence),
                    message=(
                        f"expected sequence {expected_sequence} but found "
                        f"{entry.sequence}; entries are missing or reordered"
                    ),
                )
            )
        expected_sequence = entry.sequence + 1

        if index == 0:
            expected_previous = genesis if entry.sequence == 0 else entry.previous_hash

        if not _same_digest(entry.previous_hash, expected_previous or ""):
            genesis_link = entry.sequence == 0
            failures.append(
                VerificationFailure(
                    kind=(
                        FailureKind.GENESIS_MISMATCH if genesis_link else FailureKind.BROKEN_LINK
                    ),
                    sequence=entry.sequence,
                    entry_id=entry.entry_id,
                    expected=expected_previous,
                    actual=entry.previous_hash,
                    message=(
                        "entry 0 does not link to this ledger's genesis hash"
                        if genesis_link
                        else f"sequence {entry.sequence} does not link to sequence "
                        f"{entry.sequence - 1}"
                    ),
                )
            )
        # Continue from what is actually stored, so one break is reported once instead of
        # cascading into a failure for every entry that follows it.
        expected_previous = entry.hash

        try:
            recomputed = chain.calculate_hash(entry)
        except ValidationError:
            # No encoder for this format_version. Reporting BAD_HASH instead would be a
            # guess: nothing in this build knows what the digest was supposed to be.
            failures.append(
                VerificationFailure(
                    kind=FailureKind.FORMAT_UNSUPPORTED,
                    sequence=entry.sequence,
                    entry_id=entry.entry_id,
                    expected=str(CURRENT_FORMAT_VERSION),
                    actual=str(entry.format_version),
                    message=(
                        f"sequence {entry.sequence} is in format_version "
                        f"{entry.format_version}, which this build cannot hash"
                    ),
                )
            )
            continue

        if not _same_digest(recomputed, entry.hash):
            failures.append(
                VerificationFailure(
                    kind=FailureKind.BAD_HASH,
                    sequence=entry.sequence,
                    entry_id=entry.entry_id,
                    expected=recomputed,
                    actual=entry.hash,
                    message=f"sequence {entry.sequence} does not hash to its stored digest",
                )
            )

        if enable_signatures:
            signature_failure = _check_signature(entry, key_resolver)
            if signature_failure is not None:
                failures.append(signature_failure)

    if not entries:
        # Contract section 5 rule 1. v1's CLI printed a success tick here, which is the
        # most dangerous answer a verifier can give: "nothing to check" is not "intact".
        failures.append(
            VerificationFailure(
                kind=FailureKind.COUNT_MISMATCH,
                sequence=None,
                entry_id=None,
                expected=">=1",
                actual="0",
                message="no entries were verified; an empty range is never evidence of integrity",
            )
        )

    if checkpoint is not None:
        failures.extend(
            _check_checkpoint(
                backend,
                checkpoint,
                chain=chain,
                ledger_id=ledger_id,
                enable_signatures=enable_signatures,
                key_resolver=key_resolver,
            )
        )

    return VerificationReport(
        ok=not failures,
        entries_verified=len(entries),
        first_sequence=entries[0].sequence if entries else None,
        last_sequence=entries[-1].sequence if entries else None,
        failures=tuple(failures),
        assurance=_assurance(
            failures,
            enable_signatures=enable_signatures,
            anchored=anchored,
            capabilities=capabilities,
        ),
        checkpoint_used=checkpoint,
    )
