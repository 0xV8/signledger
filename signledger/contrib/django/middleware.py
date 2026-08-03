"""The Django audit middleware.

One ledger entry per request, plus an optional row in the ORM index so the trail can be
searched with the ORM and the admin. The ledger is the evidence; the index is a
convenience, and it is never allowed to be the reason a request fails.
"""

import functools
import logging
import sys
import uuid
from typing import Any, Callable, Dict, Optional, Tuple

from django.core.signals import got_request_exception
from django.db import IntegrityError, transaction

from signledger.contrib import (
    COMPLETION_FINAL,
    COMPLETION_PROVISIONAL,
    FAIL_CLOSED,
    KIND_COMPLETION,
    KIND_REQUEST,
    OUTCOME_COMPLETE,
    OUTCOME_INCOMPLETE,
    AuditRecorder,
    SettlingBody,
    build_audit_record,
    is_skipped,
    will_send_body,
)
from signledger.contrib.django import _resolve, get_ledger, get_settings
from signledger.contrib.django.models import AuditIndex
from signledger.contrib.redaction import RedactionPolicy

__all__ = ["INTEGRATION", "SignLedgerAuditMiddleware"]

#: Name recorded in every entry this integration writes.
INTEGRATION = "django"

#: A child of the ``signledger`` logger, so one logging config covers the whole library.
logger = logging.getLogger("signledger.contrib.django")

#: Where :func:`_note_exception` leaves the exception name for ``__call__`` to read.
_ERROR_ATTR = "_signledger_error"


def _note_exception(sender: Any, request: Any = None, **kwargs: Any) -> None:
    """Remember what failed, while it is still knowable.

    Django's exception middleware turns an unhandled exception into a 500 response
    *inside* ``get_response``, so the ``except`` in ``__call__`` never sees the ordinary
    case - the entry was written for a 500 that named no cause, and
    ``WHERE error IS NOT NULL`` came back structurally zero however many views had blown
    up. This signal covers that path; the ``except`` covers the propagating one.

    ``got_request_exception`` does not pass the exception, by design: receivers are
    expected to read it from ``sys.exc_info()``, which is still set while the signal is
    being sent.
    """
    exception = sys.exc_info()[1]
    if exception is not None and request is not None:
        setattr(request, _ERROR_ATTR, type(exception).__name__)


# Module-level and stateless, with a dispatch_uid, so that however many middleware
# instances Django builds there is exactly one receiver and nothing holds a reference to
# a handler that has gone away.
got_request_exception.connect(_note_exception, dispatch_uid="signledger.audit")


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
        if is_skipped(request.path, self.skip_paths):
            return self.get_response(request)

        # Decided and read *before* the view runs. Django caches ``request.body``, so
        # the view still gets its own copy - auditing must not consume the request.
        body, reason = self._body(request)
        shape = {
            "client_ip": self.redaction.client_ip(
                peer=request.META.get("REMOTE_ADDR"),
                forwarded_for=request.META.get("HTTP_X_FORWARDED_FOR"),
            ),
            "headers": self.redaction.headers(dict(request.headers)),
            "query": self.redaction.query(dict(request.GET.lists())),
            "body": body,
            "body_reason": reason,
            "user": self._user(request),
        }
        context: Dict[str, Any] = {
            "request_id": uuid.uuid4().hex,
            "method": request.method,
            "path": request.path,
            "shape": shape,
        }

        try:
            response = self.get_response(request)
        except Exception as err:
            # The failed requests are the ones an audit trail exists for. Without this,
            # DEBUG_PROPAGATE_EXCEPTIONS served a real 500 over TCP while the ledger
            # gained nothing at all - and check_integrity() went on reporting ok about a
            # chain with no memory of the request.
            #
            # Deliberately not ``process_exception``: on Django 5.2 that hook fires *and*
            # the exception still propagates past get_response, so recording in both
            # places double-counts. It is also never called for an exception raised by a
            # middleware below this one, which this ``except`` does catch.
            self._record(context, status=500, error=type(err).__name__)
            raise

        streamed = self._will_stream(request, response)
        self._record(
            context,
            status=response.status_code,
            error=getattr(request, _ERROR_ATTR, None),
            completion=COMPLETION_PROVISIONAL if streamed else COMPLETION_FINAL,
        )
        if streamed:
            response.streaming_content = SettlingBody(
                response.streaming_content,
                functools.partial(self._settle, context, response.status_code),
            )
        return response

    # -- recording ---------------------------------------------------------------------

    def _record(
        self,
        context: Dict[str, Any],
        *,
        status: Optional[int],
        error: Optional[str] = None,
        completion: str = COMPLETION_FINAL,
        kind: str = KIND_REQUEST,
        outcome: Optional[str] = None,
    ) -> Any:
        record = build_audit_record(
            integration=INTEGRATION,
            method=context["method"],
            path=context["path"],
            status=status,
            error=error,
            kind=kind,
            request_id=context["request_id"],
            completion=completion,
            outcome=outcome,
            **context["shape"],
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
        return entry

    def _settle(
        self,
        context: Dict[str, Any],
        status: int,
        complete: bool,
        error: Optional[str],
    ) -> None:
        """Record how a promised response actually ended.

        No failing closed here: the client has had the response since the first chunk, so
        there is nothing left to refuse. The recorder counts and names the failure, and it
        is logged, because unlike Flask's extension this middleware instance is owned by
        Django's ``BaseHandler`` and is unreachable from application code - a counter
        nobody can read is not an observable.
        """
        try:
            self._record(
                context,
                status=status,
                error=error,
                kind=KIND_COMPLETION,
                completion=COMPLETION_FINAL,
                outcome=OUTCOME_COMPLETE if complete else OUTCOME_INCOMPLETE,
            )
        except Exception as err:  # noqa: BLE001 - counted and logged, never swallowed
            self.recorder.audit_failures += 1
            self.recorder.last_audit_error = type(err).__name__
            logger.warning(
                "a streamed response could not be settled (%s). Its request entry stays "
                "completion=provisional, which is the honest record: the status was "
                "promised and how it ended is not known.",
                type(err).__name__,
            )

    def _will_stream(self, request: Any, response: Any) -> bool:
        """Whether this response's body still has to be produced.

        ``FileResponse`` is deliberately excluded. Assigning a wrapper to
        ``streaming_content`` sets ``file_to_stream`` to ``None``, and Django's WSGI
        handler takes the ``wsgi.file_wrapper`` zero-copy path only while that attribute
        is set - so settling a download would quietly take every download off ``sendfile``.
        Where ``file_wrapper`` *is* used the server iterates the raw file object and never
        enters the wrapper at all, which would settle a fully delivered file as
        ``incomplete``. It is recorded ``final``, on the same footing as Flask's
        ``send_file``: a body sent under a declared length shows its own truncation to the
        client as a short read.
        """
        if not getattr(response, "streaming", False):
            return False
        if getattr(response, "file_to_stream", None) is not None:
            return False
        return will_send_body(request.method, response.status_code)

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
        try:
            # Built *inside* the try. It used to sit above it, reading record["user"] and
            # three of its neighbours, which made the "never raises" promise false for any
            # record shaped differently from the one call site that existed - and the
            # first settlement record would then have taken every streamed request down
            # with a KeyError, on the default INDEX_ENTRIES=True.
            self._write_index_row(
                {
                    "ledger_id": entry.ledger_id,
                    "sequence": entry.sequence,
                    "entry_id": entry.entry_id,
                    "entry_hash": entry.hash,
                    "timestamp": entry.timestamp,
                    "method": record["method"],
                    "path": record["path"][:512],
                    "status": record["status"],
                    "actor": record["user"],
                    "kind": record["kind"],
                    "completion": record["completion"],
                    "request_id": record["request_id"],
                }
            )
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
