"""Anchor sinks - publishing a chain head somewhere the adversary does not control.

Contract section 8. This is the half of tamper-evidence that a hash chain cannot supply.
SHA-256 is public, so an insider who can edit a row can re-derive every digest after it
and leave a chain that verifies perfectly. And a *truncated* tail leaves no trace at all:
what remains is genuinely self-consistent. The only way either becomes visible is if the
head and the entry count were recorded somewhere else, earlier.

Two sinks ship:

* :class:`FileAnchorSink` - append-only JSONL, fsynced on publish. Point it at a
  different filesystem, a different host, or a replicated volume; an anchor stored in the
  same database as the ledger protects against nothing.
* :class:`NullAnchorSink` - discards. For tests and for wiring a ledger that has not
  chosen a sink yet. It deliberately never hands a checkpoint back, so it can never
  confer ``ANCHORED`` assurance.

``HTTPAnchorSink`` and notarization services are extension points: implement
:class:`AnchorSink` and any ledger can use it.
"""

import json
import os
import threading
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Protocol, runtime_checkable

from signledger.anchor.checkpoint import Checkpoint
from signledger.core.exceptions import AnchorError
from signledger.crypto.canonical import canonicalize

__all__ = [
    "AnchorReceipt",
    "AnchorSink",
    "CHECKPOINT_TYPE_TAG",
    "FileAnchorSink",
    "NullAnchorSink",
    "checkpoint_signing_payload",
]

#: Domain tag separating a checkpoint preimage from the genesis preimage. Both are
#: canonical JSON objects, so without distinct tags one could be replayed as the other.
CHECKPOINT_TYPE_TAG = "signledger/checkpoint"

#: The fields written to, and read back from, an anchor record.
_RECORD_FIELDS = (
    "ledger_id",
    "sequence",
    "head_hash",
    "entry_count",
    "merkle_root",
    "timestamp",
    "signature",
    "key_id",
    "format_version",
)


@dataclass(frozen=True)
class AnchorReceipt:
    """Proof that a checkpoint was handed to a sink, and where it landed."""

    sink: str
    ledger_id: str
    sequence: int
    published_at: datetime
    location: Optional[str] = None


@runtime_checkable
class AnchorSink(Protocol):
    """Somewhere a chain head can be published and read back from."""

    name: str

    def publish(self, cp: Checkpoint) -> AnchorReceipt: ...

    def fetch_latest(self, ledger_id: str) -> Optional[Checkpoint]: ...


def checkpoint_signing_payload(cp: Checkpoint) -> bytes:
    """Return the canonical bytes a checkpoint's signature covers.

    Every field except ``signature`` itself is included - a payload that covered the
    signature could never be produced, because the signature does not exist until after
    the payload has been signed.

    ``key_id`` **is** covered, for the same reason it is covered in an entry preimage: an
    attacker who could re-point a checkpoint at a key they control would only need to
    supply their own signature to make a forged head look anchored.
    """
    return canonicalize(
        {
            "type": CHECKPOINT_TYPE_TAG,
            "ledger_id": cp.ledger_id,
            "sequence": cp.sequence,
            "head_hash": cp.head_hash,
            "entry_count": cp.entry_count,
            "merkle_root": cp.merkle_root,
            "timestamp": cp.timestamp,
            "key_id": cp.key_id,
            "format_version": cp.format_version,
        }
    )


class NullAnchorSink:
    """A sink that accepts checkpoints and keeps none of them.

    ``fetch_latest`` always answers ``None``. That is not a limitation to work around: a
    sink that stores nothing outside the process has anchored nothing, and reporting
    ``ANCHORED`` for it would be exactly the kind of unearned assurance this rebuild
    exists to remove.
    """

    name = "null"

    def publish(self, cp: Checkpoint) -> AnchorReceipt:
        return AnchorReceipt(
            sink=self.name,
            ledger_id=cp.ledger_id,
            sequence=cp.sequence,
            published_at=datetime.now(timezone.utc),
            location=None,
        )

    def fetch_latest(self, ledger_id: str) -> Optional[Checkpoint]:
        return None


class FileAnchorSink:
    """Append-only JSONL on disk, fsynced on every publish.

    One JSON object per line, carrying exactly the :class:`Checkpoint` fields, so a
    third-party implementation can audit the anchor without this library.

    A malformed line is an **error**, never something to skip past: silently ignoring it
    would let an attacker delete an anchor simply by corrupting it.
    """

    name = "file"

    def __init__(self, path: str) -> None:
        self.path = path
        self._lock = threading.Lock()

    def publish(self, cp: Checkpoint) -> AnchorReceipt:
        """Append ``cp`` to the anchor file and flush it all the way to the disk."""
        line = json.dumps(_to_record(cp), sort_keys=True, ensure_ascii=False) + "\n"
        try:
            with self._lock, open(self.path, "a", encoding="utf-8") as handle:
                handle.write(line)
                # flush() only reaches the OS buffer; fsync() is what survives a crash.
                handle.flush()
                os.fsync(handle.fileno())
        except OSError as err:
            # The path can carry a directory layout an operator would rather not see in a
            # log, so only the error type is reported (contract section 10).
            raise AnchorError(
                f"could not publish to the anchor file: {type(err).__name__}"
            ) from err
        return AnchorReceipt(
            sink=self.name,
            ledger_id=cp.ledger_id,
            sequence=cp.sequence,
            published_at=datetime.now(timezone.utc),
            location=self.path,
        )

    def fetch_latest(self, ledger_id: str) -> Optional[Checkpoint]:
        """Return the highest-sequence checkpoint published for ``ledger_id``."""
        try:
            with open(self.path, encoding="utf-8") as handle:
                lines = handle.read().splitlines()
        except FileNotFoundError:
            # Nothing has been published yet. That is a state, not a failure.
            return None
        except OSError as err:
            raise AnchorError(f"could not read the anchor file: {type(err).__name__}") from err

        newest: Optional[Checkpoint] = None
        for number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            candidate = _from_record(_decode(line, number), number)
            if candidate.ledger_id != ledger_id:
                continue
            if newest is None or candidate.sequence > newest.sequence:
                newest = candidate
        return newest


def _to_record(cp: Checkpoint) -> Dict[str, Any]:
    record = {field: getattr(cp, field) for field in _RECORD_FIELDS}
    record["timestamp"] = cp.timestamp.astimezone(timezone.utc).isoformat()
    return record


def _decode(line: str, number: int) -> Dict[str, Any]:
    try:
        decoded = json.loads(line)
    except ValueError as err:
        raise AnchorError(f"anchor record {number} is not valid JSON") from err
    if not isinstance(decoded, dict):
        raise AnchorError(f"anchor record {number} is a {type(decoded).__name__}, not an object")
    return decoded


def _from_record(record: Dict[str, Any], number: int) -> Checkpoint:
    missing = [field for field in _RECORD_FIELDS if field not in record]
    if missing:
        raise AnchorError(f"anchor record {number} is missing: {', '.join(sorted(missing))}")
    restored = {field: record[field] for field in _RECORD_FIELDS}
    try:
        moment = datetime.fromisoformat(restored["timestamp"])
    except (TypeError, ValueError) as err:
        raise AnchorError(f"anchor record {number} has an unreadable timestamp") from err
    return replace(Checkpoint(**restored), timestamp=moment)
