"""The Django audit middleware.

One ledger entry per request, plus an optional row in the ORM index so the trail can be
searched with the ORM and the admin. The ledger is the evidence; the index is a
convenience, and it is never allowed to be the reason a request fails.
"""

import logging
from typing import Any, Callable, Dict, Optional, Tuple

from django.db import IntegrityError, transaction

from signledger.contrib import FAIL_CLOSED, AuditRecorder, build_audit_record
from signledger.contrib.django import _resolve, get_ledger, get_settings
from signledger.contrib.django.models import AuditIndex
from signledger.contrib.redaction import RedactionPolicy

__all__ = ["INTEGRATION", "SignLedgerAuditMiddleware"]

#: Name recorded in every entry this integration writes.
INTEGRATION = "django"

#: A child of the ``signledger`` logger, so one logging config covers the whole library.
logger = logging.getLogger("signledger.contrib.django")


class SignLedgerAuditMiddleware:
    """Records every request that is not explicitly skipped.

    Configured entirely from ``settings.SIGNLEDGER`` - see
    :mod:`signledger.contrib.django` - so a deployment changes behaviour in one place
    rather than by editing the ``MIDDLEWARE`` line.
    """

    def __init__(self, get_response: Any) -> None:
        self.get_response = get_response
        config = get_settings()
        self.redaction = RedactionPolicy(
            capture_body=bool(config["CAPTURE_BODY"]),
            max_body_bytes=int(config["MAX_BODY_BYTES"]),
            trusted_proxy_count=int(config["TRUSTED_PROXY_COUNT"]),
            extra_sensitive_fields=config["EXTRA_SENSITIVE_FIELDS"],
        )
        # Raises ConfigurationError here, at startup, rather than on the first request
        # that happens to fail months later.
        self.recorder = AuditRecorder(
            None,
            redaction=self.redaction,
            audit_failure_policy=config.get("AUDIT_FAILURE_POLICY", FAIL_CLOSED),
        )
        self.skip_paths: Tuple[str, ...] = tuple(config["SKIP_PATHS"])
        self.index_entries = bool(config["INDEX_ENTRIES"])
        # Resolved once, here, and through the same function ``LEDGER_FACTORY`` uses, so
        # a dotted path works in both. It used to be stored verbatim, which meant a
        # perfectly ordinary settings value raised ``TypeError: 'str' object is not
        # callable`` on every request while ``manage.py check`` reported nothing.
        resolver = config["USER_RESOLVER"]
        self.user_resolver: Optional[Callable[[Any], Any]] = (
            None if resolver is None else _resolve(resolver, "USER_RESOLVER")
        )

    def __call__(self, request: Any) -> Any:
        if any(request.path.startswith(prefix) for prefix in self.skip_paths):
            return self.get_response(request)

        # Decided and read *before* the view runs. Django caches ``request.body``, so
        # the view still gets its own copy - auditing must not consume the request.
        body, reason = self._body(request)

        response = self.get_response(request)

        record = build_audit_record(
            integration=INTEGRATION,
            method=request.method,
            path=request.path,
            status=response.status_code,
            client_ip=self.redaction.client_ip(
                peer=request.META.get("REMOTE_ADDR"),
                forwarded_for=request.META.get("HTTP_X_FORWARDED_FOR"),
            ),
            headers=self.redaction.headers(dict(request.headers)),
            query=self.redaction.query(dict(request.GET.lists())),
            body=body,
            body_reason=reason,
            user=self._user(request),
        )
        # Looked up per request, built once. The lookup has to be late rather than done
        # in __init__: a ledger built before a WSGI server forks would hand every worker
        # the *same* SQLite handle. Deployments call reset_ledger() in a post-fork hook,
        # and only a late lookup notices. get_ledger() memoises, so this is a dict read
        # and always the same object within a process.
        self.recorder.ledger = get_ledger()
        entry = self.recorder.record(record, {"integration": INTEGRATION})
        if entry is not None and self.index_entries:
            self._index(entry, record)
        return response

    # -- internals ---------------------------------------------------------------------

    def _body(self, request: Any) -> Tuple[Any, str]:
        decision = self.redaction.body_decision(
            content_length=_content_length(request),
            content_type=request.content_type or None,
        )
        if not decision.capture:
            return None, decision.reason
        return self.redaction.decode_body(request.body), "captured"

    def _user(self, request: Any) -> Optional[str]:
        """The acting identity, from ``USER_RESOLVER`` if a deployment named one.

        Falls back to ``request.user`` when ``django.contrib.auth`` is installed, and
        records the **primary key** rather than the username: a pk is stable across a
        rename, and a username is far more likely to be personal data.
        """
        if self.user_resolver is not None:
            resolved = self.user_resolver(request)
            return None if resolved is None else str(resolved)
        user = getattr(request, "user", None)
        if user is None or not getattr(user, "is_authenticated", False):
            return None
        return str(user.pk)

    def _index(self, entry: Any, record: Dict[str, Any]) -> None:
        """Mirror the entry into the ORM index.

        This runs *after* the ledger entry has been written, and the ledger is immutable,
        so this method must never raise. An exception here would hand the client an
        outcome the tamper-evident log has already recorded, permanently, as a different
        one - a 500 the log insists was a 200. For a library whose entire product is
        "this record is trustworthy", that is the worst thing it can produce, and it is
        why the failure policy governs the ledger write and nothing else.

        The failure is counted on the recorder, named in the log, and the request is
        served. That is the promise in this module's docstring: the index is a
        convenience, and it is never allowed to be the reason a request fails.
        """
        row = {
            "ledger_id": entry.ledger_id,
            "sequence": entry.sequence,
            "entry_id": entry.entry_id,
            "entry_hash": entry.hash,
            "timestamp": entry.timestamp,
            "method": record["method"],
            "path": record["path"][:512],
            "status": record["status"],
            "actor": record["user"],
        }
        try:
            self._write_index_row(row)
        except Exception as err:  # noqa: BLE001 - counted and logged, never swallowed
            self.recorder.audit_failures += 1
            self.recorder.last_audit_error = type(err).__name__
            # The exception *type* only. A database driver quotes the row it refused,
            # and that row names an actor and a path: contract section 10.
            logger.warning(
                "the audit index row could not be written (%s). The ledger entry is "
                "unaffected - AuditIndex is a queryable mirror and never the evidence - "
                "so the request was served as it was answered.",
                type(err).__name__,
            )

    def _write_index_row(self, row: Dict[str, Any]) -> None:
        """Insert the row, replacing one left behind by a store that no longer exists.

        ``UNIQUE (ledger_id, sequence)`` is what makes this table an index rather than a
        pile of rows, and it is also what a deployment collides with the moment
        ``LEDGER_FACTORY`` is repointed at a fresh store - a new file, a restored backup,
        a wiped staging database - while ``ledger_id`` keeps the one stable value the
        README asks every worker to share. The new chain restarts at sequence 0 on top of
        rows mirroring the old one. Those rows are a stale mirror of a store that no
        longer exists; the entry being written now is the one that holds that position.
        Replacing it is recovery, not loss: the ledger holds every entry either way.

        The insert runs inside ``atomic()`` so a rejected one is rolled back to a
        savepoint rather than left in the connection's lap. Anything that wraps a request
        in a transaction - a consumer's own ``TestCase``, a transaction middleware ahead
        of this one - would otherwise have that transaction poisoned by the very
        collision this method exists to survive, and on PostgreSQL every later statement
        in it fails outright.
        """
        try:
            with transaction.atomic():
                AuditIndex.objects.create(**row)
        except IntegrityError:
            with transaction.atomic():
                AuditIndex.objects.filter(
                    ledger_id=row["ledger_id"], sequence=row["sequence"]
                ).delete()
                AuditIndex.objects.create(**row)
            logger.warning(
                "replaced a stale audit index row at %s#%s: it mirrored an entry from a "
                "store that no longer holds that position. Repointing LEDGER_FACTORY at "
                "a fresh store while keeping the same ledger_id does exactly this. The "
                "ledger is unaffected. Delete the AuditIndex rows for this ledger_id to "
                "clear the rest of the stale mirror, or give the new store its own id.",
                row["ledger_id"],
                row["sequence"],
            )


def _content_length(request: Any) -> Optional[int]:
    """The declared length, or ``None`` when it is absent or not a number."""
    raw = request.META.get("CONTENT_LENGTH")
    if raw in (None, ""):
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        # An unparseable length is no length at all, and capture will refuse it.
        return None
