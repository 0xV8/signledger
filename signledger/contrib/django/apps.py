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

The third check is the only warning here rather than an error: ``MAX_BODY_BYTES`` and
Django's ``DATA_UPLOAD_MAX_MEMORY_SIZE`` describe the same limit and were never compared,
so a deployment could ask for body capture and get markers instead. Survivable, and
therefore not allowed to stop a deploy - but not something an operator should have to
discover from the ledger.
"""

from typing import Any, List, Optional, Sequence

from django.apps import AppConfig
from django.conf import settings
from django.core.checks import Error, Tags, Warning, register

# ``_resolve`` is imported rather than reimplemented on purpose. A check that decides
# "importable" by different rules than the code that later does the importing is worse
# than no check at all: it would go green on exactly the configuration that then fails.
from signledger.contrib.django import (
    SETTINGS_KEY,
    SignLedgerDjangoConfigurationError,
    _resolve,
    get_settings,
)

__all__ = [
    "SignLedgerConfig",
    "check_body_capture_limits",
    "check_ledger_factory",
    "check_user_resolver",
]

#: Stable ids. These are the handles a deployment puts in ``SILENCED_SYSTEM_CHECKS``,
#: so they are public API and must not be renumbered.
MISSING_FACTORY_ID = "signledger_django.E001"
UNIMPORTABLE_FACTORY_ID = "signledger_django.E002"
WRONG_TYPE_FACTORY_ID = "signledger_django.E003"
UNIMPORTABLE_RESOLVER_ID = "signledger_django.E004"
WRONG_TYPE_RESOLVER_ID = "signledger_django.E005"
INCOMPATIBLE_BODY_LIMITS_ID = "signledger_django.W001"

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


def check_body_capture_limits(
    app_configs: Optional[Sequence[Any]] = None, **kwargs: Any
) -> List[Any]:
    """Report two body-size limits that cannot both be satisfied.

    ``SIGNLEDGER['MAX_BODY_BYTES']`` and Django's ``DATA_UPLOAD_MAX_MEMORY_SIZE`` are
    independent numbers that describe the same thing, and nothing has ever compared them.
    Set the Django one lower - hardening it is standard advice - and every request whose
    body falls between the two passes SignLedger's check and is then refused by Django on
    the very line that reads it. The middleware no longer turns that into an unaudited
    400, but it still cannot capture those bodies, so a deployment that asked for
    ``CAPTURE_BODY`` is silently getting less than it configured. That is worth saying out
    loud once, at deploy time, rather than leaving an operator to infer it from markers.

    A ``Warning`` and not an ``Error``: the configuration is survivable, and a check that
    can stop ``manage.py migrate`` has to be reserved for the ones that are not.

    Both numbers are named in the message, unlike every other check in this module. The
    rule they follow is the same one - contract section 10 forbids echoing a settings
    *value* back, because that is where credentials live - and a byte count is the one
    kind of settings value that cannot carry one. Naming them is also the entire point:
    the operator's next action is to reconcile two numbers, and a message that withheld
    them would send them looking for both.
    """
    config = get_settings()
    if not config.get("CAPTURE_BODY"):
        return []
    django_limit = getattr(settings, "DATA_UPLOAD_MAX_MEMORY_SIZE", None)
    if django_limit is None:
        return []
    max_body_bytes = int(config["MAX_BODY_BYTES"])
    if max_body_bytes <= django_limit:
        return []
    return [
        Warning(
            f"{SETTINGS_KEY}['MAX_BODY_BYTES'] is {max_body_bytes} but "
            f"DATA_UPLOAD_MAX_MEMORY_SIZE is {django_limit}, so a body between the two "
            "is admitted by SignLedger and then refused by Django.",
            hint=(
                f"Lower {SETTINGS_KEY}['MAX_BODY_BYTES'] to at most "
                f"{django_limit}, or raise DATA_UPLOAD_MAX_MEMORY_SIZE. Until then those "
                "requests are audited with body [OMITTED:over-request-limit], and any "
                "view that reads the body still gets Django's 400."
            ),
            id=INCOMPATIBLE_BODY_LIMITS_ID,
        )
    ]


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
        register(check_body_capture_limits, Tags.compatibility)
