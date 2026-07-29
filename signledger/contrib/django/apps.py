"""The Django ``AppConfig`` that makes this a real, installable application.

It also carries the app's deploy-time system checks. ``SIGNLEDGER["LEDGER_FACTORY"]`` is
required and has no default, and a deployment that leaves it out is fatally broken: every
request raises. The middleware looks the ledger up per request on purpose - a ledger built
before a WSGI server forks would hand every worker the same handle - so nothing before the
first request would otherwise notice. A system check is the one place that can, and both
``manage.py check`` and ``manage.py migrate`` (which declares
``requires_system_checks = "__all__"``) run it, so a broken release stops at deploy time
rather than at the first byte of production traffic.

``USER_RESOLVER`` gets the identical treatment, for the identical reason. It is optional,
but a misconfigured one is exactly as fatal as a missing factory - every request raises -
and it used to have no check at all.
"""

from typing import Any, List, Optional, Sequence

from django.apps import AppConfig
from django.core.checks import Error, Tags, register

# ``_resolve`` is imported rather than reimplemented on purpose. A check that decides
# "importable" by different rules than the code that later does the importing is worse
# than no check at all: it would go green on exactly the configuration that then fails.
from signledger.contrib.django import (
    SETTINGS_KEY,
    SignLedgerDjangoConfigurationError,
    _resolve,
    get_settings,
)

__all__ = ["SignLedgerConfig", "check_ledger_factory", "check_user_resolver"]

#: Stable ids. These are the handles a deployment puts in ``SILENCED_SYSTEM_CHECKS``,
#: so they are public API and must not be renumbered.
MISSING_FACTORY_ID = "signledger_django.E001"
UNIMPORTABLE_FACTORY_ID = "signledger_django.E002"
WRONG_TYPE_FACTORY_ID = "signledger_django.E003"
UNIMPORTABLE_RESOLVER_ID = "signledger_django.E004"
WRONG_TYPE_RESOLVER_ID = "signledger_django.E005"

_HINT = (
    f"Set {SETTINGS_KEY}['LEDGER_FACTORY'] to a zero-argument callable, or to the dotted "
    "path of one, that returns a signledger Ledger. Build the Ledger inside the factory "
    "rather than at import time, so each forked worker gets its own."
)

_RESOLVER_HINT = (
    f"Set {SETTINGS_KEY}['USER_RESOLVER'] to a callable taking the request, or to the "
    "dotted path of one, or leave it out to record the primary key of request.user."
)


def check_ledger_factory(
    app_configs: Optional[Sequence[Any]] = None, **kwargs: Any
) -> List[Any]:
    """Report a fatally misconfigured ``LEDGER_FACTORY`` as a system check error.

    The factory is *resolved* and deliberately never *called*. Checks run in the master
    process, before any fork; building the ledger here would open the audit store in the
    one process that must not hold it.

    Messages name the setting and, at most, the offending Python type. They never echo
    the configured value back - contract section 10 - because a factory path is a
    settings value like any other and settings values are where credentials live.
    """
    factory = get_settings().get("LEDGER_FACTORY")
    if factory is None:
        return [
            Error(
                f"{SETTINGS_KEY}['LEDGER_FACTORY'] is required and has no default.",
                hint=_HINT,
                id=MISSING_FACTORY_ID,
            )
        ]
    if not callable(factory) and not isinstance(factory, str):
        return [
            Error(
                f"{SETTINGS_KEY}['LEDGER_FACTORY'] must be a callable or a dotted "
                f"import path, not {type(factory).__name__}.",
                hint=_HINT,
                id=WRONG_TYPE_FACTORY_ID,
            )
        ]
    try:
        _resolve(factory)
    except SignLedgerDjangoConfigurationError:
        return [
            Error(
                f"{SETTINGS_KEY}['LEDGER_FACTORY'] could not be imported.",
                hint=_HINT,
                id=UNIMPORTABLE_FACTORY_ID,
            )
        ]
    return []


def check_user_resolver(
    app_configs: Optional[Sequence[Any]] = None, **kwargs: Any
) -> List[Any]:
    """Report a ``USER_RESOLVER`` that cannot be called as a system check error.

    The setting is optional, so absent is not an error. Anything else is resolved by the
    same :func:`~signledger.contrib.django._resolve` the middleware uses and, like the
    factory, never called: a resolver runs against a request, and a check has none.

    Without this, a dotted path - the natural thing to write in a settings module, and
    the form ``LEDGER_FACTORY`` has always accepted - was stored verbatim, reported
    nothing here, and raised ``TypeError: 'str' object is not callable`` on every single
    request that reached the middleware.
    """
    resolver = get_settings().get("USER_RESOLVER")
    if resolver is None:
        return []
    if not callable(resolver) and not isinstance(resolver, str):
        return [
            Error(
                f"{SETTINGS_KEY}['USER_RESOLVER'] must be a callable or a dotted "
                f"import path, not {type(resolver).__name__}.",
                hint=_RESOLVER_HINT,
                id=WRONG_TYPE_RESOLVER_ID,
            )
        ]
    try:
        _resolve(resolver, "USER_RESOLVER")
    except SignLedgerDjangoConfigurationError:
        return [
            Error(
                f"{SETTINGS_KEY}['USER_RESOLVER'] could not be imported.",
                hint=_RESOLVER_HINT,
                id=UNIMPORTABLE_RESOLVER_ID,
            )
        ]
    return []


class SignLedgerConfig(AppConfig):  # type: ignore[misc]
    """Registers the audit index model, its shipped migration, and the config check.

    ``label`` is set explicitly. Django otherwise derives it from the last component of
    ``name``, which here is ``django`` - a label that collides with anything else called
    the same and reads as nonsense in ``migrate`` output. Every model in this app pins
    the same value in its ``Meta``, so the two can never drift apart.
    """

    name = "signledger.contrib.django"
    label = "signledger_django"
    verbose_name = "SignLedger audit"
    default_auto_field = "django.db.models.BigAutoField"

    def ready(self) -> None:
        """Register this app's checks with Django's check framework.

        Tagged ``compatibility`` so ``manage.py check -t compatibility`` includes them.
        ``registered_checks`` is a set, so a second ``ready()`` - autoreload, a test
        harness calling ``django.setup()`` again - re-registers nothing.
        """
        register(check_ledger_factory, Tags.compatibility)
        register(check_user_resolver, Tags.compatibility)
