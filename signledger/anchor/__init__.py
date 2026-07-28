"""External anchoring - checkpoints and the sinks that publish them.

A checkpoint records a chain head and an entry count. Publishing it through an
:class:`~signledger.anchor.sinks.AnchorSink` is what turns a silent deletion into a
detectable one, because the record lives somewhere the adversary does not control.

Nothing here imports ``cryptography``: :func:`checkpoint_signing_payload` only *builds*
the bytes a signature will cover, so the anchor package stays usable without the
``signatures`` extra.
"""

from signledger.anchor.checkpoint import Checkpoint
from signledger.anchor.sinks import (
    CHECKPOINT_TYPE_TAG,
    AnchorReceipt,
    AnchorSink,
    FileAnchorSink,
    NullAnchorSink,
    checkpoint_signing_payload,
)

__all__ = [
    "CHECKPOINT_TYPE_TAG",
    "AnchorReceipt",
    "AnchorSink",
    "Checkpoint",
    "FileAnchorSink",
    "NullAnchorSink",
    "checkpoint_signing_payload",
]
