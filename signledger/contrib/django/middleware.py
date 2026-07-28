"""The Django audit middleware.

One ledger entry per request, plus an optional row in the ORM index so the trail can be
searched with the ORM and the admin. The ledger is the evidence; the index is a
convenience, and it is never allowed to be the reason a request fails.
"""

from typing import Any, Dict, Optional, Tuple

from signledger.contrib import FAIL_CLOSED, AuditRecorder, build_audit_record
from signledger.contrib.django import get_ledger, get_settings
from signledger.contrib.django.models import AuditIndex
from signledger.contrib.redaction import RedactionPolicy

__all__ = ["INTEGRATION", "SignLedgerAuditMiddleware"]

#: Name recorded in every entry this integration writes.
INTEGRATION = "django"


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
        self.user_resolver = config["USER_RESOLVER"]

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
        """The acting identity.

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

        Wrapped so that an index failure follows the same policy as a ledger failure.
        Under ``fail_open`` it is counted and the request proceeds: the evidence is
        already safely in the ledger, and a convenience table must never be the reason
        a request 500s.
        """
        try:
            AuditIndex.objects.create(
                ledger_id=entry.ledger_id,
                sequence=entry.sequence,
                entry_id=entry.entry_id,
                entry_hash=entry.hash,
                timestamp=entry.timestamp,
                method=record["method"],
                path=record["path"][:512],
                status=record["status"],
                actor=record["user"],
            )
        except Exception as err:  # noqa: BLE001 - re-raised or counted, never swallowed
            self.recorder.audit_failures += 1
            self.recorder.last_audit_error = type(err).__name__
            if self.recorder.fail_closed:
                raise


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
