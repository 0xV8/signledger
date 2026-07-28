"""Django integration.

Add the app and the middleware::

    INSTALLED_APPS = [..., "signledger.contrib.django"]
    MIDDLEWARE = [..., "signledger.contrib.django.middleware.SignLedgerAuditMiddleware"]

    SIGNLEDGER = {
        "LEDGER_FACTORY": "myapp.audit.build_ledger",   # required
        "CAPTURE_BODY": False,
        "TRUSTED_PROXY_COUNT": 0,
        "AUDIT_FAILURE_POLICY": "fail_closed",
    }

then ``manage.py migrate``.

``LEDGER_FACTORY`` is **required and has no default**. A default would have to be an
in-memory ledger, and an audit trail that silently evaporates when a worker restarts is
worse than no audit trail: it looks like it is working.
"""

from typing import Any, Callable, Dict, Optional

from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
from django.utils.module_loading import import_string

from signledger.core.exceptions import ConfigurationError

__all__ = [
    "SETTINGS_KEY",
    "SignLedgerDjangoConfigurationError",
    "get_ledger",
    "get_settings",
    "reset_ledger",
]

#: The Django settings key this integration reads.
SETTINGS_KEY = "SIGNLEDGER"

#: Defaults for everything except ``LEDGER_FACTORY``, which must be supplied.
DEFAULTS: Dict[str, Any] = {
    "AUDIT_FAILURE_POLICY": "fail_closed",
    "CAPTURE_BODY": False,
    "MAX_BODY_BYTES": 8192,
    "TRUSTED_PROXY_COUNT": 0,
    "EXTRA_SENSITIVE_FIELDS": (),
    "SKIP_PATHS": (),
    "INDEX_ENTRIES": True,
    "USER_RESOLVER": None,
}


class SignLedgerDjangoConfigurationError(ConfigurationError, ImproperlyConfigured):  # type: ignore[misc]
    """A misconfigured SignLedger Django app.

    Inherits from both hierarchies deliberately. A Django developer reaches for
    ``ImproperlyConfigured`` and gets Django's startup reporting for free; a SignLedger
    user wraps the library in ``ConfigurationError``. Making callers guess which one to
    catch would be a papercut in the exact place people are already confused.
    """


#: The process-wide ledger. Built once: a factory called per request would mint a new
#: identity each time and fork the chain on every single request.
_LEDGER: Optional[Any] = None


def get_settings() -> Dict[str, Any]:
    """The ``SIGNLEDGER`` settings dict, merged over :data:`DEFAULTS`."""
    configured = getattr(settings, SETTINGS_KEY, None) or {}
    merged = dict(DEFAULTS)
    merged.update(configured)
    return merged


def get_ledger() -> Any:
    """The application's ledger, built once from ``LEDGER_FACTORY``.

    The factory may be a callable or a dotted path to one.
    """
    global _LEDGER
    if _LEDGER is None:
        _LEDGER = _build_ledger()
    return _LEDGER


def reset_ledger() -> None:
    """Forget the memoised ledger. Intended for tests and for worker re-forking."""
    global _LEDGER
    _LEDGER = None


def _build_ledger() -> Any:
    factory = get_settings().get("LEDGER_FACTORY")
    if factory is None:
        raise SignLedgerDjangoConfigurationError(
            f"{SETTINGS_KEY}['LEDGER_FACTORY'] is required. There is deliberately no "
            "default: an audit trail that silently disappears when a worker restarts "
            "is worse than none, because it looks like it is working."
        )
    return _resolve(factory)()


def _resolve(factory: Any) -> Callable[[], Any]:
    if callable(factory):
        direct: Callable[[], Any] = factory
        return direct
    try:
        resolved: Callable[[], Any] = import_string(factory)
    except ImportError as err:
        # Names the setting, never the traceback's file paths.
        raise SignLedgerDjangoConfigurationError(
            f"{SETTINGS_KEY}['LEDGER_FACTORY'] could not be imported"
        ) from err
    return resolved
