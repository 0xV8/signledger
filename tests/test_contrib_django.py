"""Tests for :mod:`signledger.contrib.django`, driven through a real Django stack.

Django is configured here in-process against an in-memory SQLite database, the app is
put in ``INSTALLED_APPS``, its shipped migration is applied, and requests go through
``django.test.Client``. Nothing is mocked.

v1's Django "integration" was a directory that shadowed its own module, so none of it
executed - there was no ``AppConfig``, no model, no migration, and no way to install it.
The test that matters most here is
:func:`test_makemigrations_detects_no_missing_migrations`: a shipped app whose migration
does not match its models breaks every ``manage.py migrate`` that installs it.
"""

import ast
import asyncio
import contextlib
import gc
import json
import logging
import pathlib
import warnings
from typing import Any, AsyncIterator, Dict, Iterator, List

import django
import pytest
from django.conf import settings

#: The audit middleware, by path, so a test can rebuild ``MIDDLEWARE`` around it.
AUDIT_MIDDLEWARE = "signledger.contrib.django.middleware.SignLedgerAuditMiddleware"

if not settings.configured:
    settings.configure(
        DEBUG=False,
        DEBUG_PROPAGATE_EXCEPTIONS=False,
        SECRET_KEY="signledger-test-key",
        ALLOWED_HOSTS=["*"],
        USE_TZ=True,
        # A shared-cache in-memory database rather than a bare ``":memory:"``, which is
        # private to the connection that opened it. Django's connections are thread-local,
        # and under ASGI the sync middleware and the settlement both run in the
        # thread-sensitive executor rather than in the thread that ran ``migrate`` - so
        # with a private database every ORM write from there hits a second, empty file and
        # comes back "no such table". That is a property of this harness and not of the
        # code under test, and it would have hidden whether the audit index mirror works
        # under ASGI at all. This is the same URI Django's own test runner uses for
        # in-memory databases.
        DATABASES={
            "default": {
                "ENGINE": "django.db.backends.sqlite3",
                "NAME": "file:signledger_test?mode=memory&cache=shared",
                "OPTIONS": {"uri": True},
            }
        },
        # ``django.contrib.admin`` and its dependencies are installed deliberately.
        # ``models.py`` promises an operator "can use the ORM and the Django admin to
        # find a record", and the only way to know that promise holds is to reach the
        # admin the way an operator does: a real superuser, a real URL, a real response.
        INSTALLED_APPS=[
            "django.contrib.contenttypes",
            "django.contrib.auth",
            "django.contrib.sessions",
            "django.contrib.messages",
            "django.contrib.admin",
            "signledger.contrib.django",
        ],
        MIDDLEWARE=[
            "django.contrib.sessions.middleware.SessionMiddleware",
            "django.contrib.auth.middleware.AuthenticationMiddleware",
            "django.contrib.messages.middleware.MessageMiddleware",
            AUDIT_MIDDLEWARE,
        ],
        TEMPLATES=[
            {
                "BACKEND": "django.template.backends.django.DjangoTemplates",
                "APP_DIRS": True,
                "OPTIONS": {
                    "context_processors": [
                        "django.template.context_processors.request",
                        "django.contrib.auth.context_processors.auth",
                        "django.contrib.messages.context_processors.messages",
                    ]
                },
            }
        ],
        STATIC_URL="/static/",
        ROOT_URLCONF="test_contrib_django",
        DEFAULT_AUTO_FIELD="django.db.models.BigAutoField",
    )
    django.setup()

from django.contrib import admin  # noqa: E402
from django.core.exceptions import ImproperlyConfigured  # noqa: E402
from django.core.management import call_command  # noqa: E402
from django.core.management.base import SystemCheckError  # noqa: E402
from django.db import connection, transaction  # noqa: E402
from django.http import (  # noqa: E402
    HttpResponse,
    JsonResponse,
    StreamingHttpResponse,
)
from django.test import Client, override_settings  # noqa: E402
from django.urls import path, reverse  # noqa: E402

from signledger.backends.memory import MemoryBackend  # noqa: E402
from signledger.contrib import FAIL_CLOSED, FAIL_OPEN  # noqa: E402
from signledger.contrib.django import (  # noqa: E402
    SignLedgerDjangoConfigurationError,
    get_ledger,
    reset_ledger,
)
from signledger.contrib.django.admin import AuditIndexAdmin, register  # noqa: E402
from signledger.contrib.django.apps import SignLedgerConfig  # noqa: E402
from signledger.contrib.django.middleware import (  # noqa: E402
    SignLedgerAuditMiddleware,
)
from signledger.contrib.django.models import AuditIndex  # noqa: E402
from signledger.contrib.redaction import REDACTED  # noqa: E402
from signledger.core.exceptions import ConfigurationError  # noqa: E402
from signledger.core.ledger import Ledger  # noqa: E402

# --------------------------------------------------------------------------------------
# A minimal but real Django application
# --------------------------------------------------------------------------------------


def login_view(request: Any) -> HttpResponse:
    return JsonResponse({"ok": True})


def items_view(request: Any) -> HttpResponse:
    return JsonResponse({"items": []})


def boom_view(request: Any) -> HttpResponse:
    raise RuntimeError("view exploded")


def health_view(request: Any) -> HttpResponse:
    return JsonResponse({"ok": True})


def echo_view(request: Any) -> HttpResponse:
    return JsonResponse({"seen": json.loads(request.body.decode("utf-8"))})


urlpatterns = [
    path("login", login_view),
    path("items", items_view),
    path("boom", boom_view),
    path("health", health_view),
    path("echo", echo_view),
    # Mounted exactly as ``startproject`` mounts it. ``admin.site.urls`` is built from
    # whatever autodiscovery registered during ``django.setup()`` above, so this line is
    # only useful if the package really does ship an ``admin`` module.
    path("admin/", admin.site.urls),
]


# --------------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------------

#: The ledger the factory below hands out, so a test can inspect it afterwards.
CURRENT: Dict[str, Any] = {}


def build_ledger() -> Ledger:
    """A ``LEDGER_FACTORY`` callable, exactly as a deployment would write one."""
    ledger = Ledger(backend=MemoryBackend(), ledger_id="django-test")
    CURRENT["ledger"] = ledger
    return ledger


def resolve_actor(request: Any) -> str:
    """A ``USER_RESOLVER`` reachable by dotted path, the way a settings module names it.

    Naming it rather than importing it is the whole point: importing a callable into
    ``settings.py`` forces its module - and everything that module imports - to load
    while the settings are still being read.
    """
    CURRENT.setdefault("resolver_calls", []).append(request)
    return "actor-from-path"


@pytest.fixture(scope="session", autouse=True)
def django_database() -> Iterator[None]:
    """Apply the app's shipped migration to the in-memory database.

    Run under a valid configuration, and with ``skip_checks=False`` so this is the same
    code path ``manage.py migrate`` takes - ``call_command`` alone defaults
    ``skip_checks`` to ``True``, which the command line never does. ``migrate`` declares
    ``requires_system_checks = "__all__"``, so this also proves the app's own check does
    not break the migration a deployment has to run.
    """
    with audited():
        call_command("migrate", verbosity=0, run_syncdb=False, skip_checks=False)
    yield


@pytest.fixture(autouse=True)
def fresh_ledger() -> Iterator[None]:
    """Every test gets its own ledger and an empty index table."""
    reset_ledger()
    CURRENT.clear()
    AuditIndex.objects.all().delete()
    yield
    reset_ledger()
    CURRENT.clear()


def audited(**options: Any) -> Any:
    """``override_settings`` for the SIGNLEDGER dict, with a working factory."""
    config: Dict[str, Any] = {"LEDGER_FACTORY": build_ledger}
    config.update(options)
    return override_settings(SIGNLEDGER=config)


def entries() -> List[Any]:
    return list(get_ledger().iter_entries())


@pytest.fixture
def audited_ledger() -> Iterator[Any]:
    """A configured ledger plus an empty mirror, for the streaming and failure tests."""
    with audited():
        AuditIndex.objects.all().delete()
        yield get_ledger()
        AuditIndex.objects.all().delete()
    reset_ledger()
    CURRENT.clear()


# ======================================================================================
# It has to be a real, installable Django app
# ======================================================================================


def test_the_app_config_declares_an_explicit_label() -> None:
    """Without one, Django derives the label from the last dotted component - ``django``.

    That would collide with any other app and reads as nonsense in ``migrate`` output.
    """
    assert SignLedgerConfig.label == "signledger_django"


def test_the_app_config_points_at_this_package() -> None:
    assert SignLedgerConfig.name == "signledger.contrib.django"


def test_the_app_is_registered_with_django() -> None:
    from django.apps import apps

    assert apps.get_app_config("signledger_django") is not None


def test_makemigrations_detects_no_missing_migrations() -> None:
    """A shipped app whose migration does not match its models breaks every
    ``manage.py migrate`` that installs it."""
    try:
        with audited():
            call_command(
                "makemigrations",
                "signledger_django",
                check=True,
                dry_run=True,
                verbosity=0,
                skip_checks=False,
            )
    except SystemExit as exit_code:  # pragma: no cover - only on a real mismatch
        pytest.fail(f"models and shipped migrations disagree (exit {exit_code})")


def test_the_initial_migration_ships_inside_the_package() -> None:
    """A migration that only exists in the repo is not a migration users get."""
    from signledger.contrib.django.migrations import __path__ as migrations_path

    assert (pathlib.Path(migrations_path[0]) / "0001_initial.py").is_file()


def test_the_index_table_exists_after_migrating() -> None:
    assert AuditIndex.objects.count() == 0


def test_the_index_is_ordered_by_sequence_and_never_by_timestamp() -> None:
    """Contract section 3: ``sequence`` is the ordering authority.

    v1 ordered by wall clock, so equal timestamps came back reversed and an NTP
    step-back made a pristine ledger look reordered.
    """
    assert AuditIndex._meta.ordering == ["sequence"]


def test_the_index_stores_no_payload_columns() -> None:
    """The index exists to be queried, which means it must hold nothing sensitive."""
    columns = {field.name for field in AuditIndex._meta.get_fields()}
    assert not columns & {"data", "metadata", "body", "headers", "payload"}


# ======================================================================================
# Configuration
# ======================================================================================


def test_a_missing_ledger_factory_is_a_configuration_error() -> None:
    """No silent in-memory default: audit records would vanish on restart."""
    with override_settings(SIGNLEDGER={}), pytest.raises(ConfigurationError):
        get_ledger()


def test_a_missing_signledger_setting_entirely_is_a_configuration_error() -> None:
    with override_settings(SIGNLEDGER=None), pytest.raises(ConfigurationError):
        get_ledger()


def test_the_configuration_error_is_also_a_django_improperly_configured() -> None:
    """Django developers catch ``ImproperlyConfigured``; SignLedger users catch
    ``ConfigurationError``. This is one object, so both work."""
    assert issubclass(SignLedgerDjangoConfigurationError, ImproperlyConfigured)
    assert issubclass(SignLedgerDjangoConfigurationError, ConfigurationError)


def test_a_ledger_factory_given_as_a_dotted_path_is_imported() -> None:
    with override_settings(
        SIGNLEDGER={"LEDGER_FACTORY": "test_contrib_django.build_ledger"}
    ):
        assert isinstance(get_ledger(), Ledger)


def test_a_ledger_factory_dotted_path_that_does_not_resolve_is_reported() -> None:
    with override_settings(SIGNLEDGER={"LEDGER_FACTORY": "nope.nothing"}), pytest.raises(
        ConfigurationError
    ):
        get_ledger()


def test_the_ledger_is_built_once_and_reused() -> None:
    """A factory called per request would fork the chain on every request."""
    with audited():
        assert get_ledger() is get_ledger()


def test_resetting_releases_the_memoised_ledger() -> None:
    with audited():
        first = get_ledger()
        reset_ledger()
        assert get_ledger() is not first


def test_an_unknown_audit_failure_policy_is_rejected() -> None:
    with audited(AUDIT_FAILURE_POLICY="maybe"), pytest.raises(ConfigurationError):
        Client().get("/items")


def test_a_ledger_factory_that_is_neither_callable_nor_a_path_is_reported() -> None:
    """A built ``Ledger`` assigned straight to the setting has no ``rsplit``, so
    ``import_string`` used to die with ``AttributeError`` - the one exception a
    ``ConfigurationError`` handler does not catch."""
    with override_settings(
        SIGNLEDGER={"LEDGER_FACTORY": Ledger(backend=MemoryBackend())}
    ), pytest.raises(ConfigurationError):
        get_ledger()


# --------------------------------------------------------------------------------------
# USER_RESOLVER
#
# ``LEDGER_FACTORY`` takes a dotted path or a callable. ``USER_RESOLVER`` took a callable
# only: a dotted path was stored verbatim, passed ``manage.py check``, and raised
# ``TypeError: 'str' object is not callable`` on every single request. The inconsistency
# *is* the trap, because a dotted path is the natural thing to write in a settings module.
# --------------------------------------------------------------------------------------


def test_a_user_resolver_given_as_a_dotted_path_is_imported() -> None:
    with audited(USER_RESOLVER="test_contrib_django.resolve_actor"):
        Client().get("/items")
        assert entries()[0].data["user"] == "actor-from-path"


def test_a_user_resolver_dotted_path_that_does_not_resolve_is_reported() -> None:
    with audited(USER_RESOLVER="myapp.audit.no_such_resolver"), pytest.raises(
        ConfigurationError
    ):
        Client().get("/items")


def test_a_user_resolver_that_is_neither_callable_nor_a_path_is_reported() -> None:
    with audited(USER_RESOLVER=42), pytest.raises(ConfigurationError):
        Client().get("/items")


def test_a_user_resolver_error_names_the_setting_it_came_from() -> None:
    """``LEDGER_FACTORY`` and ``USER_RESOLVER`` resolve through the same code. An error
    naming the wrong setting sends the reader to the wrong line of ``settings.py``."""
    with audited(USER_RESOLVER="myapp.audit.no_such_resolver"), pytest.raises(
        ConfigurationError
    ) as caught:
        Client().get("/items")
    assert "USER_RESOLVER" in str(caught.value)


# ======================================================================================
# Django system checks
#
# A missing LEDGER_FACTORY is fatal: every single request 500s. Before this section
# existed, ``manage.py check`` and ``manage.py migrate`` both reported success and the
# operator found out from production traffic. The deliberately-late ``get_ledger()`` in
# the middleware stays late - it has to, so a post-fork ``reset_ledger()`` is noticed -
# so the *only* place this can be caught before traffic is a system check.
# ======================================================================================


def test_manage_py_check_fails_when_the_ledger_factory_is_missing() -> None:
    """The headline of Defect G1: this used to exit 0 and 500 on every request."""
    with override_settings(SIGNLEDGER={}), pytest.raises(SystemCheckError) as caught:
        call_command("check")
    assert "LEDGER_FACTORY" in str(caught.value)


def test_manage_py_check_fails_when_signledger_is_not_configured_at_all() -> None:
    with override_settings(SIGNLEDGER=None), pytest.raises(SystemCheckError):
        call_command("check")


def test_manage_py_check_fails_when_the_factory_path_does_not_import() -> None:
    """A typo in the dotted path is as fatal as leaving the setting out."""
    with override_settings(
        SIGNLEDGER={"LEDGER_FACTORY": "myapp.audit.does_not_exist"}
    ), pytest.raises(SystemCheckError):
        call_command("check")


def test_manage_py_check_fails_when_the_factory_is_neither_callable_nor_a_path() -> None:
    """A built ledger assigned straight to the setting is a real mistake to make.

    It is not callable and it has no ``rsplit``, so left to ``import_string`` it would
    crash the check framework itself with an ``AttributeError``.
    """
    with override_settings(
        SIGNLEDGER={"LEDGER_FACTORY": Ledger(backend=MemoryBackend())}
    ), pytest.raises(SystemCheckError):
        call_command("check")


def test_manage_py_migrate_also_refuses_a_missing_ledger_factory() -> None:
    """``migrate`` declares ``requires_system_checks = "__all__"``.

    Deploy scripts run ``migrate`` far more reliably than they run ``check``, so this is
    the one that actually stops a broken release. ``skip_checks=False`` because
    ``call_command`` defaults it to ``True`` while ``manage.py`` leaves it ``False``;
    without it this asserts nothing about what a deployment actually runs.
    """
    with override_settings(SIGNLEDGER={}), pytest.raises(SystemCheckError):
        call_command("migrate", verbosity=0, run_syncdb=False, skip_checks=False)


def test_manage_py_check_passes_with_a_working_factory() -> None:
    with audited():
        call_command("check")


def test_manage_py_check_passes_with_a_dotted_path_factory() -> None:
    with override_settings(
        SIGNLEDGER={"LEDGER_FACTORY": "test_contrib_django.build_ledger"}
    ):
        call_command("check")


def test_the_check_resolves_the_factory_without_calling_it() -> None:
    """Checking must not build a ledger.

    ``manage.py check`` runs in the master process, before any fork. Opening the audit
    store here is the very hazard the late ``get_ledger()`` exists to avoid.
    """
    with audited():
        call_command("check")
        assert "ledger" not in CURRENT


def test_the_check_runs_under_the_compatibility_tag() -> None:
    """Registered with a tag so ``manage.py check -t compatibility`` includes it."""
    with override_settings(SIGNLEDGER={}), pytest.raises(SystemCheckError):
        call_command("check", tags=["compatibility"])


def test_a_missing_factory_reports_one_silenceable_error_id() -> None:
    """The id is the handle for ``SILENCED_SYSTEM_CHECKS``, so it is part of the API."""
    from signledger.contrib.django.apps import check_ledger_factory

    with override_settings(SIGNLEDGER={}):
        assert [error.id for error in check_ledger_factory(None)] == [
            "signledger_django.E001"
        ]


def test_an_unimportable_factory_has_its_own_error_id() -> None:
    from signledger.contrib.django.apps import check_ledger_factory

    with override_settings(SIGNLEDGER={"LEDGER_FACTORY": "myapp.audit.nope"}):
        assert [error.id for error in check_ledger_factory(None)] == [
            "signledger_django.E002"
        ]


def test_a_factory_of_the_wrong_type_has_its_own_error_id() -> None:
    from signledger.contrib.django.apps import check_ledger_factory

    with override_settings(SIGNLEDGER={"LEDGER_FACTORY": 42}):
        assert [error.id for error in check_ledger_factory(None)] == [
            "signledger_django.E003"
        ]


def test_the_check_reports_nothing_at_all_for_a_good_configuration() -> None:
    from signledger.contrib.django.apps import check_ledger_factory

    with audited():
        assert check_ledger_factory(None) == []


def test_the_check_never_echoes_the_configured_value_back() -> None:
    """Contract section 10: a settings value can be a DSN with a password in it."""
    from signledger.contrib.django.apps import check_ledger_factory

    secret = "myapp.audit.postgres://user:hunter2@db/audit"
    with override_settings(SIGNLEDGER={"LEDGER_FACTORY": secret}):
        rendered = " ".join(str(error) for error in check_ledger_factory(None))
    assert "hunter2" not in rendered


# --------------------------------------------------------------------------------------
# ... and the same treatment for USER_RESOLVER, which had none: a dotted path there
# passed every check there was and then 500ed every request.
# --------------------------------------------------------------------------------------


def test_manage_py_check_fails_when_the_user_resolver_does_not_import() -> None:
    with audited(USER_RESOLVER="myapp.audit.no_such_resolver"), pytest.raises(
        SystemCheckError
    ) as caught:
        call_command("check")
    assert "USER_RESOLVER" in str(caught.value)


def test_manage_py_check_fails_when_the_user_resolver_is_the_wrong_type() -> None:
    with audited(USER_RESOLVER=42), pytest.raises(SystemCheckError):
        call_command("check")


def test_manage_py_check_passes_with_a_dotted_path_user_resolver() -> None:
    with audited(USER_RESOLVER="test_contrib_django.resolve_actor"):
        call_command("check")


def test_an_unimportable_user_resolver_has_its_own_error_id() -> None:
    from signledger.contrib.django.apps import check_user_resolver

    with audited(USER_RESOLVER="myapp.audit.no_such_resolver"):
        assert [error.id for error in check_user_resolver(None)] == [
            "signledger_django.E004"
        ]


def test_a_user_resolver_of_the_wrong_type_has_its_own_error_id() -> None:
    from signledger.contrib.django.apps import check_user_resolver

    with audited(USER_RESOLVER=42):
        assert [error.id for error in check_user_resolver(None)] == [
            "signledger_django.E005"
        ]


def test_the_user_resolver_check_reports_nothing_when_there_is_none() -> None:
    """It is optional, and absent is the default."""
    from signledger.contrib.django.apps import check_user_resolver

    with audited():
        assert check_user_resolver(None) == []


def test_the_user_resolver_check_reports_nothing_for_a_good_configuration() -> None:
    from signledger.contrib.django.apps import check_user_resolver

    with audited(USER_RESOLVER="test_contrib_django.resolve_actor"):
        assert check_user_resolver(None) == []


def test_the_user_resolver_check_resolves_it_without_calling_it() -> None:
    """Checks run in the master process against a request that does not exist."""
    with audited(USER_RESOLVER="test_contrib_django.resolve_actor"):
        call_command("check")
    assert "resolver_calls" not in CURRENT


def test_the_user_resolver_check_never_echoes_the_configured_value_back() -> None:
    from signledger.contrib.django.apps import check_user_resolver

    secret = "myapp.audit.postgres://user:hunter2@db/audit"
    with audited(USER_RESOLVER=secret):
        rendered = " ".join(str(error) for error in check_user_resolver(None))
    assert "hunter2" not in rendered


# ======================================================================================
# The headline requirement: credentials must not reach the ledger
# ======================================================================================


def test_an_authorization_header_and_a_password_body_leak_neither_into_the_ledger() -> None:
    with audited(CAPTURE_BODY=True):
        Client().post(
            "/login",
            data=json.dumps({"user": "ada", "password": "hunter2"}),
            content_type="application/json",
            HTTP_AUTHORIZATION="Bearer sk-live-51H8xQsecret",
        )
        stored = json.dumps(entries()[0].model_dump(), default=str)
    assert "sk-live-51H8xQsecret" not in stored
    assert "hunter2" not in stored


def test_the_authorization_header_is_recorded_as_the_marker() -> None:
    with audited():
        Client().get("/items", HTTP_AUTHORIZATION="Bearer secret-value")
        assert entries()[0].data["headers"]["authorization"] == REDACTED


def test_a_cookie_is_never_recorded() -> None:
    with audited():
        client = Client()
        client.cookies["sessionid"] = "abc123"
        client.get("/items")
        assert entries()[0].data["headers"]["cookie"] == REDACTED


def test_a_query_parameter_carrying_an_api_key_is_redacted() -> None:
    """Every value stays a list, redacted or not: arity is auditable, the value is not."""
    with audited():
        Client().get("/items?page=2&api_key=sk-live-1")
        query = entries()[0].data["query"]
        assert (query["api_key"], query["page"]) == ([REDACTED], ["2"])


# ======================================================================================
# What gets recorded
# ======================================================================================


def test_one_entry_is_appended_per_request() -> None:
    with audited():
        Client().get("/items")
        Client().get("/items")
        assert get_ledger().count() == 2


def test_the_recorded_status_is_the_one_the_client_received() -> None:
    with audited():
        response = Client().get("/items")
        assert (response.status_code, entries()[0].data["status"]) == (200, 200)


def test_the_recorded_method_and_path_match_the_request() -> None:
    with audited():
        Client().post("/login", data={"a": "1"})
        data = entries()[0].data
        assert (data["method"], data["path"]) == ("POST", "/login")


def test_a_view_that_raises_is_still_audited() -> None:
    with audited():
        response = Client(raise_request_exception=False).get("/boom")
        assert (response.status_code, entries()[0].data["status"]) == (500, 500)


def test_the_integration_names_itself_in_the_record() -> None:
    with audited():
        Client().get("/items")
        assert entries()[0].data["integration"] == "django"


def test_the_chain_still_verifies_after_a_series_of_audited_requests() -> None:
    with audited():
        for _ in range(5):
            Client().get("/items")
        assert get_ledger().check_integrity().ok


def test_the_resolved_user_is_recorded() -> None:
    with audited(USER_RESOLVER=lambda request: "user-42"):
        Client().get("/items")
        assert entries()[0].data["user"] == "user-42"


def test_no_user_is_recorded_when_the_request_has_none() -> None:
    """No ``AuthenticationMiddleware``, so ``request.user`` does not exist at all.

    Not a hypothetical: an API-only deployment with token auth commonly runs without it,
    and the middleware must not assume the attribute is there.
    """
    with audited(), override_settings(MIDDLEWARE=[AUDIT_MIDDLEWARE]):
        Client().get("/items")
        assert entries()[0].data["user"] is None


def test_an_anonymous_django_user_is_recorded_as_no_user() -> None:
    """The real ``AnonymousUser`` this time, attached by real auth middleware."""
    with audited():
        Client().get("/items")
        assert entries()[0].data["user"] is None


# ======================================================================================
# The ORM index
# ======================================================================================


def test_the_index_row_mirrors_the_ledger_entry() -> None:
    with audited():
        Client().get("/items")
        entry = entries()[0]
        row = AuditIndex.objects.get()
        assert (row.entry_id, row.sequence, row.entry_hash) == (
            entry.entry_id,
            entry.sequence,
            entry.hash,
        )


def test_the_index_records_the_request_shape() -> None:
    with audited():
        Client().get("/items")
        row = AuditIndex.objects.get()
        assert (row.method, row.path, row.status) == ("GET", "/items", 200)


def test_indexing_can_be_switched_off() -> None:
    """The ledger is the evidence; the index is a convenience some deployments skip."""
    with audited(INDEX_ENTRIES=False):
        Client().get("/items")
        assert AuditIndex.objects.count() == 0
        assert get_ledger().count() == 1


def test_the_index_is_written_for_every_request() -> None:
    with audited():
        for _ in range(3):
            Client().get("/items")
        assert AuditIndex.objects.count() == 3


# ======================================================================================
# The index is a convenience and is never the reason a request fails
#
# The ledger entry is written *first*, and only then is the index row. So an index write
# that re-raised - which it did under the default ``fail_closed`` - turned a request the
# ledger had already recorded, permanently and immutably, as a 200 into a 500 the client
# actually received. A tamper-evident log asserting an outcome that did not happen is the
# most damaging thing this library can produce, and the module docstring already promised
# it could not: "the index is a convenience, and it is never allowed to be the reason a
# request fails."
# ======================================================================================


@contextlib.contextmanager
def no_index_table() -> Iterator[None]:
    """Really drop the index table for the duration of the block.

    A real ``OperationalError`` from real SQLite, and - unlike a colliding row - not
    something the middleware can recover from. The table is dropped and rebuilt with
    Django's own schema editor from the same model, so nothing here is simulated.
    """
    with connection.schema_editor() as editor:
        editor.delete_model(AuditIndex)
    try:
        yield
    finally:
        with connection.schema_editor() as editor:
            editor.create_model(AuditIndex)


def test_the_ledger_never_records_an_outcome_the_client_did_not_receive() -> None:
    """The headline of D1, stated as the invariant it broke."""
    with audited(), no_index_table():
        response = Client(raise_request_exception=False).get("/items")
        assert entries()[0].data["status"] == response.status_code


@pytest.mark.parametrize("policy", [FAIL_CLOSED, FAIL_OPEN])
def test_an_index_failure_serves_the_real_response_under_either_policy(
    policy: str,
) -> None:
    """``audit_failure_policy`` governs the *ledger* write. The index is not evidence,
    so no setting may make it able to fail a request."""
    with audited(AUDIT_FAILURE_POLICY=policy), no_index_table():
        assert Client().get("/items").status_code == 200


def test_the_evidence_is_still_written_when_the_mirror_cannot_be() -> None:
    """And the fix must not be "index first": that would lose the entry instead."""
    with audited(), no_index_table():
        Client().get("/items")
        assert get_ledger().count() == 1


def test_an_index_failure_is_logged_rather_than_silently_dropped(caplog: Any) -> None:
    """``fail_open`` used to swallow it whole: no row, no exception, no trace of either."""
    with audited(AUDIT_FAILURE_POLICY=FAIL_OPEN), no_index_table():
        with caplog.at_level(logging.WARNING, logger="signledger"):
            Client().get("/items")
    assert [record.levelno for record in caplog.records] == [logging.WARNING]
    assert "OperationalError" in caplog.text


def test_an_index_failure_log_never_carries_the_request_payload(caplog: Any) -> None:
    """Contract section 10. The record being mirrored holds a redacted body, but the
    exception a database raises can quote the row it refused."""
    with audited(CAPTURE_BODY=True), no_index_table():
        with caplog.at_level(logging.WARNING, logger="signledger"):
            Client().post(
                "/login",
                data=json.dumps({"user": "ada", "password": "hunter2"}),
                content_type="application/json",
                HTTP_AUTHORIZATION="Bearer sk-live-51H8xQsecret",
            )
    rendered = caplog.text + repr([record.__dict__ for record in caplog.records])
    assert caplog.records, "the failure was not reported at all"
    assert "hunter2" not in rendered
    assert "sk-live-51H8xQsecret" not in rendered


def test_the_index_docstring_does_not_contradict_the_module_promise() -> None:
    """The defect was two docstrings saying opposite things, with the code implementing
    the damaging one. A method that conditions on the failure policy is the wrong one."""
    method_doc = (SignLedgerAuditMiddleware._index.__doc__ or "").lower()
    assert "never" in method_doc
    assert FAIL_CLOSED not in method_doc
    assert FAIL_OPEN not in method_doc


# ======================================================================================
# A stale index row is recoverable, not a reason to refuse traffic
#
# ``AuditIndex`` carries UNIQUE (ledger_id, sequence). The README Deployment section says
# to read one stable ``ledger_id`` from configuration and hand it to every worker. Repoint
# ``LEDGER_FACTORY`` at a fresh store - a new file, a restored backup, a wiped staging
# database - while keeping that id, and the new chain starts again at sequence 0 on top of
# index rows mirroring a store that no longer exists. The Django consumer took 307
# consecutive 500s doing exactly what the README instructs.
# ======================================================================================


def test_repointing_the_store_under_a_stable_ledger_id_still_serves_traffic() -> None:
    with audited():
        Client().get("/items")
        reset_ledger()  # a fresh store, the same configured ledger_id
        assert Client(raise_request_exception=False).get("/items").status_code == 200


def test_a_stale_index_row_is_replaced_by_the_entry_that_now_holds_the_position() -> None:
    with audited():
        Client().get("/items")
        stale = AuditIndex.objects.get()
        reset_ledger()
        Client(raise_request_exception=False).get("/items")
        rows = AuditIndex.objects.filter(ledger_id="django-test", sequence=0)
        assert [row.entry_id for row in rows] == [entries()[0].entry_id]
        assert rows[0].entry_id != stale.entry_id


def test_the_index_keeps_mirroring_after_a_stale_row_was_replaced() -> None:
    """A failed statement can leave a connection unusable. The next request must be
    indexed normally, and the whole table must mirror the store that is live now."""
    with audited():
        Client().get("/items")
        reset_ledger()
        Client(raise_request_exception=False).get("/items")
        Client().get("/items")
        assert [row.sequence for row in AuditIndex.objects.all()] == [0, 1]
        assert [row.entry_id for row in AuditIndex.objects.all()] == [
            entry.entry_id for entry in entries()
        ]


def test_a_stale_index_row_is_replaced_inside_an_enclosing_transaction() -> None:
    """A consumer's own ``TestCase`` wraps every request it makes in a transaction, and
    a transaction middleware ahead of this one does the same in production. A rejected
    INSERT that is not rolled back to a savepoint takes that transaction down with it."""
    with audited():
        Client().get("/items")
        reset_ledger()
        with transaction.atomic():
            Client(raise_request_exception=False).get("/items")
            rows = AuditIndex.objects.filter(ledger_id="django-test", sequence=0)
            assert [row.entry_id for row in rows] == [entries()[0].entry_id]


def test_replacing_a_stale_index_row_says_what_happened_and_what_to_do(
    caplog: Any,
) -> None:
    """Refusing the request was the old answer, and it explained nothing to anyone."""
    with audited():
        Client().get("/items")
        reset_ledger()
        with caplog.at_level(logging.WARNING, logger="signledger"):
            Client(raise_request_exception=False).get("/items")
    assert [record.levelno for record in caplog.records] == [logging.WARNING]
    assert "LEDGER_FACTORY" in caplog.text
    assert "django-test" in caplog.text


# ======================================================================================
# Body capture
# ======================================================================================


def test_the_body_is_not_recorded_by_default() -> None:
    with audited():
        Client().post(
            "/login", data=json.dumps({"user": "ada"}), content_type="application/json"
        )
        assert entries()[0].data["body"] == "[OMITTED:disabled]"


def test_an_enabled_body_capture_records_the_redacted_body() -> None:
    with audited(CAPTURE_BODY=True):
        Client().post(
            "/login",
            data=json.dumps({"user": "ada", "password": "hunter2"}),
            content_type="application/json",
        )
        assert entries()[0].data["body"] == {"user": "ada", "password": REDACTED}


def test_a_body_over_the_limit_is_recorded_as_omitted() -> None:
    with audited(CAPTURE_BODY=True, MAX_BODY_BYTES=16):
        Client().post(
            "/login",
            data=json.dumps({"user": "a" * 200}),
            content_type="application/json",
        )
        assert entries()[0].data["body"] == "[OMITTED:too-large]"


def test_the_view_can_still_read_a_body_the_middleware_captured() -> None:
    with audited(CAPTURE_BODY=True):
        response = Client().post(
            "/echo", data=json.dumps({"user": "ada"}), content_type="application/json"
        )
        assert response.json() == {"seen": {"user": "ada"}}


# ======================================================================================
# audit_failure_policy
# ======================================================================================


def test_fail_closed_is_the_default_and_turns_a_write_failure_into_a_500() -> None:
    with audited():
        Client().get("/items")
        get_ledger().backend.close()
        assert Client(raise_request_exception=False).get("/items").status_code == 500


def test_fail_open_still_serves_the_real_response() -> None:
    with audited(AUDIT_FAILURE_POLICY=FAIL_OPEN):
        Client().get("/items")
        get_ledger().backend.close()
        assert Client().get("/items").status_code == 200


def test_a_fail_open_index_write_does_not_break_the_request() -> None:
    """The index is a convenience. It must never be the reason a request fails."""
    with audited(AUDIT_FAILURE_POLICY=FAIL_OPEN):
        Client().get("/items")
        assert AuditIndex.objects.count() == 1


# ======================================================================================
# Trusted proxies
# ======================================================================================


def test_x_forwarded_for_is_ignored_when_no_proxies_are_trusted() -> None:
    with audited():
        Client().get("/items", HTTP_X_FORWARDED_FOR="1.2.3.4", REMOTE_ADDR="10.0.0.9")
        assert entries()[0].data["client_ip"] == "10.0.0.9"


def test_the_client_ip_is_taken_from_the_trusted_position_in_the_chain() -> None:
    with audited(TRUSTED_PROXY_COUNT=1):
        Client().get(
            "/items",
            HTTP_X_FORWARDED_FOR="evil, 203.0.113.7",
            REMOTE_ADDR="10.0.0.9",
        )
        assert entries()[0].data["client_ip"] == "203.0.113.7"


# ======================================================================================
# skip_paths and extra fields
# ======================================================================================


def test_a_skipped_path_writes_no_entry() -> None:
    with audited(SKIP_PATHS=["/health"]):
        Client().get("/health")
        assert get_ledger().count() == 0


def test_a_path_that_is_not_skipped_still_writes() -> None:
    with audited(SKIP_PATHS=["/health"]):
        Client().get("/items")
        assert get_ledger().count() == 1


def test_extra_sensitive_fields_from_settings_are_applied() -> None:
    with audited(CAPTURE_BODY=True, EXTRA_SENSITIVE_FIELDS=["patient_name"]):
        Client().post(
            "/login",
            data=json.dumps({"patient_name": "zsofia"}),
            content_type="application/json",
        )
        assert entries()[0].data["body"]["patient_name"] == REDACTED


# ======================================================================================
# Identity, index failures and malformed headers
# ======================================================================================


class Actor:
    """A stand-in for whatever ``django.contrib.auth`` would attach.

    Real middleware attaching a real attribute - the auth app is not installed here, so
    there is nothing else to hang ``request.user`` off.
    """

    def __init__(self, pk: Any, authenticated: bool) -> None:
        self.pk = pk
        self.is_authenticated = authenticated


def attach_signed_in_user(get_response: Any) -> Any:
    def middleware(request: Any) -> Any:
        request.user = Actor(pk=77, authenticated=True)
        return get_response(request)

    return middleware


def attach_anonymous_user(get_response: Any) -> Any:
    def middleware(request: Any) -> Any:
        request.user = Actor(pk=None, authenticated=False)
        return get_response(request)

    return middleware


def test_a_signed_in_user_is_recorded_by_primary_key() -> None:
    """The pk, not the username: it survives a rename and is less likely to be personal
    data than the name someone chose."""
    with audited(), override_settings(
        MIDDLEWARE=["test_contrib_django.attach_signed_in_user", AUDIT_MIDDLEWARE]
    ):
        Client().get("/items")
        assert entries()[0].data["user"] == "77"


def test_an_anonymous_user_is_recorded_as_no_user() -> None:
    """``AnonymousUser`` has a ``pk`` of ``None``; recording it would invent an actor."""
    with audited(), override_settings(
        MIDDLEWARE=["test_contrib_django.attach_anonymous_user", AUDIT_MIDDLEWARE]
    ):
        Client().get("/items")
        assert entries()[0].data["user"] is None


def test_an_explicit_resolver_wins_over_the_request_user() -> None:
    with audited(USER_RESOLVER=lambda request: None), override_settings(
        MIDDLEWARE=["test_contrib_django.attach_signed_in_user", AUDIT_MIDDLEWARE]
    ):
        Client().get("/items")
        assert entries()[0].data["user"] is None


def test_an_unparseable_content_length_is_treated_as_absent() -> None:
    with audited(CAPTURE_BODY=True):
        Client().post(
            "/login",
            data=json.dumps({"user": "ada"}),
            content_type="application/json",
            CONTENT_LENGTH="ten",
        )
        assert entries()[0].data["body"] == "[OMITTED:no-content-length"  "]"


def test_an_index_row_names_its_chain_position_when_printed() -> None:
    """The admin list view and every log line go through ``__str__``."""
    with audited():
        Client().get("/items")
        assert str(AuditIndex.objects.get()).endswith("#0")


# ======================================================================================
# The Django admin
#
# ``models.py`` says the index exists "so an operator can use the ORM and the Django
# admin to find a record", and no ``admin.py`` shipped: ``AuditIndex`` was registered
# with no admin site and its URL 404ed. An operator with a UI is the reason the mirror
# table is worth having, so the promise is kept - read-only, because every row mirrors
# an append-only log that nobody may edit through a web form.
# ======================================================================================

#: The changelist an operator lands on. ``reverse()`` only knows this name if the model
#: is registered, so every use of it is also an assertion that it is.
CHANGELIST = "admin:signledger_django_auditindex_changelist"


@pytest.fixture
def operator() -> Client:
    """A logged-in superuser, the way an operator actually reaches the admin."""
    from django.contrib.auth.models import User

    User.objects.all().delete()
    user = User.objects.create(username="operator", is_staff=True, is_superuser=True)
    client = Client()
    client.force_login(user)
    return client


def test_the_audit_index_is_registered_with_the_admin() -> None:
    """Autodiscovery imported it during ``django.setup()``; nothing here helped it."""
    assert admin.site.is_registered(AuditIndex)


def test_the_admin_url_for_the_index_resolves() -> None:
    assert reverse(CHANGELIST) == "/admin/signledger_django/auditindex/"


def test_an_operator_can_find_a_recorded_request_in_the_admin(operator: Client) -> None:
    with audited():
        Client().get("/items")
        response = operator.get(reverse(CHANGELIST))
    assert response.status_code == 200
    assert "/items" in response.content.decode("utf-8")


def test_an_operator_can_open_the_row_for_a_recorded_request(operator: Client) -> None:
    with audited():
        Client().get("/items")
        row = AuditIndex.objects.get()
        url = reverse("admin:signledger_django_auditindex_change", args=[row.pk])
        assert operator.get(url).status_code == 200


def test_the_admin_refuses_to_add_a_row(operator: Client) -> None:
    """There is no such thing as an audit entry that was not audited."""
    with audited():
        url = reverse("admin:signledger_django_auditindex_add")
        assert operator.get(url).status_code == 403


def test_the_admin_refuses_to_change_a_row(operator: Client) -> None:
    with audited():
        Client().get("/items")
        row = AuditIndex.objects.get()
        url = reverse("admin:signledger_django_auditindex_change", args=[row.pk])
        assert operator.post(url, {"status": "500"}).status_code == 403
        # The operator's own request is audited too, so this names the row it aimed at.
        assert AuditIndex.objects.get(pk=row.pk).status == 200


def test_the_admin_refuses_to_delete_a_row(operator: Client) -> None:
    with audited():
        Client().get("/items")
        row = AuditIndex.objects.get()
        url = reverse("admin:signledger_django_auditindex_delete", args=[row.pk])
        assert operator.post(url, {"post": "yes"}).status_code == 403
        assert AuditIndex.objects.filter(pk=row.pk).exists()


def test_the_model_admin_grants_no_write_permission_to_anyone() -> None:
    """Not even a superuser, and not for any object: the answer never depends on who
    is asking, because the row is a mirror of something that cannot change."""
    from django.test import RequestFactory

    model_admin = AuditIndexAdmin(AuditIndex, admin.site)
    request = RequestFactory().get("/admin/")
    assert not model_admin.has_add_permission(request)
    assert not model_admin.has_change_permission(request)
    assert not model_admin.has_delete_permission(request)


def test_every_field_is_read_only_in_the_admin() -> None:
    """Belt and braces with the permissions above: a field that is not listed here
    renders as an editable widget the moment someone adds a change permission back."""
    from django.test import RequestFactory

    model_admin = AuditIndexAdmin(AuditIndex, admin.site)
    request = RequestFactory().get("/admin/")
    assert set(model_admin.get_readonly_fields(request)) == {
        field.name for field in AuditIndex._meta.fields
    }


def test_registering_on_a_second_admin_site_works() -> None:
    """A deployment with its own ``AdminSite`` is normal, and ``register()`` is the
    documented way to reach it."""
    site = admin.AdminSite()
    assert register(site) is True
    assert site.is_registered(AuditIndex)


def test_registering_twice_is_not_an_error() -> None:
    """``admin.site.register`` raises ``AlreadyRegistered``. Autodiscovery, an explicit
    call and an autoreload can all land on the same site."""
    site = admin.AdminSite()
    register(site)
    assert register(site) is True


def test_nothing_is_registered_when_the_admin_app_is_not_installed() -> None:
    """The module must import in a project without ``django.contrib.admin``: touching
    ``admin.site`` there raises ``LookupError: No installed app with label 'admin'``,
    and ``tests/test_packaging.py`` imports every module in the package in exactly such
    a project."""
    site = admin.AdminSite()
    with override_settings(
        INSTALLED_APPS=["django.contrib.contenttypes", "signledger.contrib.django"]
    ):
        assert register(site) is False
    assert not site.is_registered(AuditIndex)


# ======================================================================================
# The fork hazard, documented in every integration rather than just this one
#
# This app's middleware has always carried a comment explaining why the ledger is looked
# up per request: a ledger built before a WSGI server forks hands every worker the same
# handle. Flask and FastAPI have the identical hazard and said nothing about it. Measured
# with PostgreSQL under ``gunicorn --preload -w 4``: protocol desync, 60-second hangs,
# and a ledger holding more entries than there were successful responses.
#
# These assertions live in the Django test module because that is where the warning that
# was copied outward already had a home. They read the shipped source with ``ast`` rather
# than importing, so they hold whether or not Flask and FastAPI are installed.
# ======================================================================================

#: Every integration whose ledger can outlive a fork, and therefore must warn about it.
FORK_EXPOSED_MODULES = ["flask.py", "fastapi.py"]

CONTRIB = pathlib.Path(__file__).resolve().parent.parent / "signledger" / "contrib"


def module_docstring(filename: str) -> str:
    source = (CONTRIB / filename).read_text(encoding="utf-8")
    return (ast.get_docstring(ast.parse(source)) or "").lower()


@pytest.mark.parametrize("filename", FORK_EXPOSED_MODULES)
def test_the_integration_names_the_fork_hazard(filename: str) -> None:
    assert "fork" in module_docstring(filename)


@pytest.mark.parametrize("filename", FORK_EXPOSED_MODULES)
def test_the_integration_names_the_flag_that_causes_it(filename: str) -> None:
    """``--preload`` is the switch that turns this from unlikely into guaranteed."""
    assert "preload" in module_docstring(filename)


@pytest.mark.parametrize("filename", FORK_EXPOSED_MODULES)
def test_the_fork_warning_says_what_to_do_instead(filename: str) -> None:
    """A warning that only says "don't" leaves the reader with no next step."""
    doc = module_docstring(filename)
    assert "post-fork" in doc or "per worker" in doc or "per-worker" in doc


# The cross-framework shape check that used to live here is now
# ``tests/test_contrib_parity.py``. It compared ``set(...)`` of the *keys* of ``entries[0]``
# between Flask and Django, for one 200 GET, with FastAPI absent, through in-process test
# clients - which never close a response iterable, and closing it is the only moment a
# streamed response's outcome is knowable. It could not have caught any of the divergences
# that shipped, all of which were about values, later entries, or streaming.


# ======================================================================================
# Streaming, failure, and what the mirror can express
# ======================================================================================


def stream_view(request: Any) -> HttpResponse:
    def body() -> Iterator[bytes]:
        yield b'{"row": 0}\n'
        yield b'{"row": 1}\n'

    return StreamingHttpResponse(body(), content_type="application/x-ndjson")


def broken_stream_view(request: Any) -> HttpResponse:
    def body() -> Iterator[bytes]:
        yield b'{"row": 0}\n'
        raise RuntimeError("feed died mid-stream")

    return StreamingHttpResponse(body(), content_type="application/x-ndjson")


urlpatterns += [
    path("stream", stream_view),
    path("broken-stream", broken_stream_view),
]


def run_wsgi(path_info: str, method: str = "GET") -> List[bytes]:
    """Drive one request the way a conforming WSGI server does.

    ``django.test.Client`` is not one: it does not close the response iterable, which is
    the only moment at which a streamed response's outcome is knowable. Every defect this
    section covers was invisible through the test client and visible over a real socket,
    so the drive has to be real even though the transport is not.
    """
    from django.core.handlers.wsgi import WSGIHandler
    from django.test import RequestFactory

    environ = RequestFactory().generic(method, path_info).environ
    iterable = WSGIHandler()(environ, lambda status, headers, exc_info=None: None)
    chunks: List[bytes] = []
    try:
        for chunk in iterable:
            chunks.append(chunk)
    except RuntimeError:
        # The generator raising is the scenario, not a test failure.
        pass
    finally:
        close = getattr(iterable, "close", None)
        if close is not None:
            close()
    return chunks


def data_of(ledger: Ledger) -> List[Dict[str, Any]]:
    return [entry.data for entry in ledger.iter_entries()]


def test_a_completed_stream_is_recorded_provisional_then_settled(audited_ledger: Ledger) -> None:
    """Django had no ``response.completion`` code path at all: no
    ``COMPLETION_PROVISIONAL``, no ``KIND_COMPLETION``, no ``_settle``. The machinery the
    other two were rebuilt with was simply never added here."""
    run_wsgi("/stream")

    assert [row["kind"] for row in data_of(audited_ledger)] == [
        "request",
        "response.completion",
    ]


def test_a_completed_stream_settles_as_complete(audited_ledger: Ledger) -> None:
    run_wsgi("/stream")

    assert data_of(audited_ledger)[1]["outcome"] == "complete"


def test_a_stream_that_dies_is_not_recorded_as_a_finished_200(
    audited_ledger: Ledger,
) -> None:
    """The blocker. A generator raising on its second chunk produced exactly one entry
    reading ``status=200, completion="final", error=None`` - the ledger asserting a
    success the client never received. Under ``wsgiref`` the client sees a clean EOF, so
    both ends believed it."""
    run_wsgi("/broken-stream")

    rows = data_of(audited_ledger)
    assert rows[0]["completion"] == "provisional"


def test_a_stream_that_dies_settles_as_incomplete(audited_ledger: Ledger) -> None:
    run_wsgi("/broken-stream")

    assert data_of(audited_ledger)[1]["outcome"] == "incomplete"


def test_a_stream_that_dies_names_the_exception(audited_ledger: Ledger) -> None:
    run_wsgi("/broken-stream")

    assert data_of(audited_ledger)[1]["error"] == "RuntimeError"


def test_a_view_that_raises_names_the_exception(audited_ledger: Ledger) -> None:
    """``error=`` reached ``build_audit_record`` from one of three call sites, so the key
    was absent from every Django entry and ``WHERE error IS NOT NULL`` was structurally
    zero however many views had blown up."""
    Client(raise_request_exception=False).get("/boom")

    assert data_of(audited_ledger)[0]["error"] == "RuntimeError"


@override_settings(DEBUG_PROPAGATE_EXCEPTIONS=True)
def test_a_propagating_exception_is_still_audited(audited_ledger: Ledger) -> None:
    """A real 500 was served and the ledger gained nothing: nothing incremented
    ``audit_failures``, and ``check_integrity()`` stayed ``ok`` on a chain with no memory
    of the request."""
    with contextlib.suppress(RuntimeError):
        Client(raise_request_exception=False).get("/boom")

    assert len(data_of(audited_ledger)) == 1


@override_settings(DEBUG_PROPAGATE_EXCEPTIONS=True)
def test_a_propagating_exception_is_recorded_as_a_500(audited_ledger: Ledger) -> None:
    with contextlib.suppress(RuntimeError):
        Client(raise_request_exception=False).get("/boom")

    assert data_of(audited_ledger)[0]["status"] == 500


def test_the_mirror_can_say_which_entries_are_provisional(audited_ledger: Ledger) -> None:
    """``models.py`` is documented as the queryable mirror and ships a read-only admin
    over it, so the ORM *is* the surface Django users are told to use. It had no ``kind``,
    ``completion`` or ``request_id`` column, which made it structurally incapable of
    answering the one question v2.0.0 added."""
    run_wsgi("/stream")

    assert AuditIndex.objects.filter(completion="provisional").count() == 1


def test_the_mirror_can_correlate_a_settlement_with_its_request(
    audited_ledger: Ledger,
) -> None:
    run_wsgi("/stream")

    rows = list(AuditIndex.objects.order_by("sequence"))
    assert rows[0].request_id == rows[1].request_id


def test_indexing_a_settlement_record_does_not_break_the_request(
    audited_ledger: Ledger,
) -> None:
    """``_index`` promises in its own docstring that it must never raise, then built the
    row dict - reading ``record["user"]`` among others - *outside* the ``try``. With
    ``INDEX_ENTRIES`` defaulting to True, the first settlement record would have taken
    every streamed request down with a ``KeyError``."""
    middleware = SignLedgerAuditMiddleware(lambda request: HttpResponse("ok"))
    middleware.recorder.ledger = audited_ledger
    entry = audited_ledger.append({"probe": True})

    middleware._index(entry, {})

    assert middleware.recorder.audit_failures == 1


# ======================================================================================
# Reading the body for audit is never the reason a request fails unaudited
#
# ``RedactionPolicy.max_body_bytes`` and Django's ``DATA_UPLOAD_MAX_MEMORY_SIZE`` are two
# independent limits with nothing holding them together. Harden the Django one below
# MAX_BODY_BYTES - or raise MAX_BODY_BYTES above Django's 2.5 MB default - and
# ``request.body`` raises ``RequestDataTooBig`` inside the middleware, *before*
# ``get_response``, so the ``except`` around it never runs. Measured on the shipped code:
# three requests, two ledger entries, and the 400 the auditor itself caused had no entry
# at all. The README's own Django snippet sets ``"CAPTURE_BODY": True``.
# ======================================================================================


def read_the_stream_first(get_response: Any) -> Any:
    """A middleware that consumes part of the request stream before the auditor does.

    This is what any streaming-upload middleware looks like. Once ``request.read()`` has
    started, Django refuses ``request.body`` with ``RawPostDataException`` for the rest of
    the request - and the auditor asks for it unconditionally.
    """

    def middleware(request: Any) -> Any:
        request.read(1)
        return get_response(request)

    return middleware


def test_a_body_django_refuses_to_hand_over_is_still_audited() -> None:
    """The headline. The audit middleware was the cause of an unaudited failed request."""
    with audited(CAPTURE_BODY=True), override_settings(DATA_UPLOAD_MAX_MEMORY_SIZE=1024):
        Client(raise_request_exception=False).post(
            "/echo",
            data=json.dumps({"blob": "a" * 2000}),
            content_type="application/json",
        )
        assert len(entries()) == 1


def test_a_body_django_refuses_to_hand_over_is_recorded_as_a_marker() -> None:
    """The established pattern for "capture was refused and here is why", and the reason
    names the limit that refused rather than SignLedger's own, which admitted it."""
    with audited(CAPTURE_BODY=True), override_settings(DATA_UPLOAD_MAX_MEMORY_SIZE=1024):
        Client(raise_request_exception=False).post(
            "/echo",
            data=json.dumps({"blob": "a" * 2000}),
            content_type="application/json",
        )
        assert entries()[0].data["body"] == "[OMITTED:over-request-limit]"


def test_the_recorded_status_is_the_one_django_actually_served() -> None:
    """Django's limit is real policy and is still honoured: the view asks for the body,
    Django refuses it, the client gets its 400 - and now the ledger says so."""
    with audited(CAPTURE_BODY=True), override_settings(DATA_UPLOAD_MAX_MEMORY_SIZE=1024):
        response = Client(raise_request_exception=False).post(
            "/echo",
            data=json.dumps({"blob": "a" * 2000}),
            content_type="application/json",
        )
        assert (response.status_code, entries()[0].data["status"]) == (400, 400)


def test_a_view_that_never_reads_the_body_is_served_normally() -> None:
    """The auditor must be invisible to the outcome. ``DATA_UPLOAD_MAX_MEMORY_SIZE``
    only ever applies to code that asks for the body, and an audit middleware asking on
    the view's behalf turned requests that would have succeeded into 400s."""
    with audited(CAPTURE_BODY=True), override_settings(DATA_UPLOAD_MAX_MEMORY_SIZE=1024):
        response = Client(raise_request_exception=False).post(
            "/login",
            data=json.dumps({"blob": "a" * 2000}),
            content_type="application/json",
        )
        assert response.status_code == 200


def test_a_body_that_was_already_read_is_recorded_rather_than_raised() -> None:
    """``RawPostDataException`` is not a 400, it is a 500 - and it was raised from inside
    the middleware, so the request that produced it went unrecorded too."""
    with audited(CAPTURE_BODY=True), override_settings(
        MIDDLEWARE=["test_contrib_django.read_the_stream_first", AUDIT_MIDDLEWARE]
    ):
        Client(raise_request_exception=False).post(
            "/login",
            data=json.dumps({"user": "ada"}),
            content_type="application/json",
        )
        assert entries()[0].data["body"] == "[OMITTED:unreadable]"


def test_a_body_that_could_not_be_read_says_so_in_the_log(caplog: Any) -> None:
    """Counted and named, never swallowed - and the type only, never the payload.

    Filtered to this library's own logger. Django logs the 500 it used to produce here
    with a full traceback, so an assertion against the whole of ``caplog`` passes on the
    broken code and proves nothing.
    """
    with audited(CAPTURE_BODY=True), override_settings(
        MIDDLEWARE=["test_contrib_django.read_the_stream_first", AUDIT_MIDDLEWARE]
    ):
        with caplog.at_level(logging.WARNING, logger="signledger"):
            Client(raise_request_exception=False).post(
                "/login",
                data=json.dumps({"user": "ada", "password": "hunter2"}),
                content_type="application/json",
            )
    ours = [
        record.getMessage()
        for record in caplog.records
        if record.name.startswith("signledger")
    ]
    assert any("RawPostDataException" in message for message in ours)
    assert "hunter2" not in " ".join(ours)


def test_a_chunked_upload_is_not_recorded_as_if_nothing_was_sent() -> None:
    """``Transfer-Encoding: chunked`` with no declared length is how a client suppresses
    body capture, and it recorded ``[OMITTED:no-content-length]`` - the same marker a
    plain GET gets, so the entry read like a policy decision rather than a withheld
    length. Reachable on gunicorn and uWSGI, which dechunk without setting
    ``CONTENT_LENGTH``."""
    with audited(CAPTURE_BODY=True):
        Client().post(
            "/login",
            data=json.dumps({"user": "ada"}),
            content_type="application/json",
            CONTENT_LENGTH="",
            HTTP_TRANSFER_ENCODING="chunked",
        )
        assert entries()[0].data["body"] == "[OMITTED:undeclared-length]"


def test_a_request_with_no_body_at_all_still_says_no_content_length() -> None:
    """And the marker for "nothing was sent" does not move: it is the one the other two
    integrations write for the same traffic."""
    with audited(CAPTURE_BODY=True):
        Client().get("/items")
        assert entries()[0].data["body"] == "[OMITTED:no-content-length]"


# --------------------------------------------------------------------------------------
# ... and a system check, so the two limits are reconciled before any traffic arrives
# --------------------------------------------------------------------------------------


def test_manage_py_check_warns_when_the_two_body_limits_disagree() -> None:
    from signledger.contrib.django.apps import check_body_capture_limits

    with audited(CAPTURE_BODY=True, MAX_BODY_BYTES=8192), override_settings(
        DATA_UPLOAD_MAX_MEMORY_SIZE=1024
    ):
        assert [warning.id for warning in check_body_capture_limits(None)] == [
            "signledger_django.W001"
        ]


@pytest.mark.parametrize("bad", ["not-a-number", None, [8192]], ids=["text", "none", "list"])
def test_a_body_limit_that_is_not_a_number_does_not_crash_the_check(bad: Any) -> None:
    """The check runs inside ``manage.py check`` and therefore inside ``manage.py
    migrate``, which declares ``requires_system_checks = "__all__"``. An unguarded
    ``int()`` on a settings value turns a typo into a traceback out of the deploy command -
    the exact blast radius this module's docstring argues a warning-only check must never
    have. Its two sibling checks each type-check and return an ``Error``; this one did not.
    """
    from signledger.contrib.django.apps import check_body_capture_limits

    with audited(CAPTURE_BODY=True, MAX_BODY_BYTES=bad), override_settings(
        DATA_UPLOAD_MAX_MEMORY_SIZE=1024
    ):
        results = check_body_capture_limits(None)

    assert [result.id for result in results] == ["signledger_django.E006"]


def test_a_django_limit_that_is_not_a_number_does_not_crash_the_check() -> None:
    """The same hazard from the other side: DATA_UPLOAD_MAX_MEMORY_SIZE is a Django
    setting this app does not own, so it cannot assume the shape of it either."""
    from signledger.contrib.django.apps import check_body_capture_limits

    with audited(CAPTURE_BODY=True, MAX_BODY_BYTES=8192), override_settings(
        DATA_UPLOAD_MAX_MEMORY_SIZE="unlimited"
    ):
        results = check_body_capture_limits(None)

    assert [result.id for result in results] == ["signledger_django.E006"]


def test_a_refused_body_limit_names_the_setting_but_never_its_value() -> None:
    """Contract section 10: a settings value can carry a credential, so a check reports
    which setting is wrong and never what was in it."""
    from signledger.contrib.django.apps import check_body_capture_limits

    with audited(CAPTURE_BODY=True, MAX_BODY_BYTES="s3cr3t-looking-typo"), override_settings(
        DATA_UPLOAD_MAX_MEMORY_SIZE=1024
    ):
        message = check_body_capture_limits(None)[0]

    assert "MAX_BODY_BYTES" in message.msg
    assert "s3cr3t-looking-typo" not in message.msg + (message.hint or "")


def test_the_body_limit_warning_names_both_numbers_to_reconcile() -> None:
    from signledger.contrib.django.apps import check_body_capture_limits

    with audited(CAPTURE_BODY=True, MAX_BODY_BYTES=8192), override_settings(
        DATA_UPLOAD_MAX_MEMORY_SIZE=1024
    ):
        rendered = " ".join(str(warning) for warning in check_body_capture_limits(None))
    assert "8192" in rendered
    assert "1024" in rendered


def test_the_body_limit_check_is_silent_when_the_limits_are_compatible() -> None:
    from signledger.contrib.django.apps import check_body_capture_limits

    with audited(CAPTURE_BODY=True, MAX_BODY_BYTES=1024), override_settings(
        DATA_UPLOAD_MAX_MEMORY_SIZE=8192
    ):
        assert check_body_capture_limits(None) == []


def test_the_body_limit_check_is_silent_when_no_body_is_captured() -> None:
    """Nothing is read, so there is nothing to reconcile."""
    from signledger.contrib.django.apps import check_body_capture_limits

    with audited(MAX_BODY_BYTES=8192), override_settings(DATA_UPLOAD_MAX_MEMORY_SIZE=1024):
        assert check_body_capture_limits(None) == []


def test_the_body_limit_check_is_silent_when_django_has_no_limit() -> None:
    """``DATA_UPLOAD_MAX_MEMORY_SIZE = None`` disables Django's limit entirely."""
    from signledger.contrib.django.apps import check_body_capture_limits

    with audited(CAPTURE_BODY=True, MAX_BODY_BYTES=8192), override_settings(
        DATA_UPLOAD_MAX_MEMORY_SIZE=None
    ):
        assert check_body_capture_limits(None) == []


def test_the_body_limit_warning_does_not_fail_manage_py_check() -> None:
    """It is a Warning and not an Error on purpose: the configuration is survivable now -
    the body is recorded as a marker - so it must not stop a deploy."""
    with audited(CAPTURE_BODY=True, MAX_BODY_BYTES=8192), override_settings(
        DATA_UPLOAD_MAX_MEMORY_SIZE=1024
    ):
        call_command("check")


def test_the_body_limit_check_is_registered_under_the_compatibility_tag() -> None:
    from django.core.checks.registry import registry

    from signledger.contrib.django.apps import check_body_capture_limits

    assert check_body_capture_limits in registry.get_checks(
        include_deployment_checks=False
    )


# ======================================================================================
# ASGI
#
# ``grep -rE "asgi|ASGI|async" tests/test_contrib_django.py`` used to return nothing, on
# an integration whose middleware is installed by every Django project template and whose
# supported Django versions have all shipped ASGI. Two things were true of it:
#
# * an async ``StreamingHttpResponse`` has an *async* iterator, and ``SettlingBody`` is a
#   sync iterable. Django's ``_set_streaming_content`` decides which kind it has by
#   trying ``iter(value)``, which succeeds on ``SettlingBody``, so the response was
#   flipped to ``is_async = False`` and ``ASGIHandler`` then died with
#   ``TypeError: 'async_generator' object is not iterable`` - after the 200 status line
#   had already gone out and before a single byte of the body did;
# * before that, the same response produced one entry saying
#   ``status=200, completion=final, error=None``: the identical overclaim the WSGI path
#   was rebuilt to remove.
#
# The middleware stays sync-only. Under ASGI Django wraps it in
# ``sync_to_async(thread_sensitive=True)``, and that was measured not to serialise
# requests - four concurrent async views sleeping 0.3s finish in 0.31s - so there is no
# throughput problem here to fix, only a correctness one.
# ======================================================================================


async def async_view(request: Any) -> HttpResponse:
    return JsonResponse({"ok": True})


async def async_stream_view(request: Any) -> HttpResponse:
    async def body() -> AsyncIterator[bytes]:
        yield b'{"row": 0}\n'
        yield b'{"row": 1}\n'

    return StreamingHttpResponse(body(), content_type="application/x-ndjson")


async def broken_async_stream_view(request: Any) -> HttpResponse:
    async def body() -> AsyncIterator[bytes]:
        yield b'{"row": 0}\n'
        raise RuntimeError("async feed died mid-stream")

    return StreamingHttpResponse(body(), content_type="application/x-ndjson")


urlpatterns += [
    path("async", async_view),
    path("async-stream", async_stream_view),
    path("broken-async-stream", broken_async_stream_view),
]


def run_asgi(path_info: str, method: str = "GET") -> List[Dict[str, Any]]:
    """Drive one request through Django's real ``ASGIHandler`` and return what it sent.

    ``django.test.AsyncClient`` does not go near ``ASGIHandler``: it builds an
    ``ASGIRequest`` by hand and calls the middleware chain, so the entire response-sending
    half - which is where an async streamed body is consumed, and where every defect this
    section covers lives - is not exercised by it at all. This drives the same application
    object ``asgi.py`` hands to daphne or uvicorn.

    The application's own exception is swallowed for the same reason :func:`run_wsgi`
    swallows one: a stream that dies is the scenario under test, not a failure of the
    harness. What was sent before it died is returned, because that is what the client
    got and what the ledger has to agree with.
    """
    from asgiref.testing import ApplicationCommunicator
    from django.core.handlers.asgi import ASGIHandler

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.1"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path_info,
        "raw_path": path_info.encode(),
        "root_path": "",
        "query_string": b"",
        "headers": [(b"host", b"testserver")],
        "client": ("127.0.0.1", 54321),
        "server": ("testserver", 80),
    }

    async def drive() -> List[Dict[str, Any]]:
        communicator = ApplicationCommunicator(ASGIHandler(), scope)
        await communicator.send_input(
            {"type": "http.request", "body": b"", "more_body": False}
        )
        messages: List[Dict[str, Any]] = []
        with contextlib.suppress(RuntimeError, TypeError, asyncio.TimeoutError):
            while True:
                message = await communicator.receive_output(timeout=1)
                messages.append(message)
                if message["type"] == "http.response.body" and not message.get(
                    "more_body"
                ):
                    break
        # Waited on separately, and after the body: ``handle()`` closes the response -
        # which is one of the two moments a settlement can happen in - only once
        # ``send_response`` has returned, so returning at the last chunk would read the
        # ledger before it had been written.
        with contextlib.suppress(RuntimeError, TypeError, asyncio.TimeoutError):
            await communicator.wait(timeout=1)
        return messages

    # ``handle()`` closes the request's ``SpooledTemporaryFile`` on its very last line,
    # which an exception escaping the response body skips - so a dying stream leaks a file
    # object that emits a ResourceWarning from ``__del__`` at whatever later moment the
    # collector happens to run. Under ``filterwarnings = ["error"]`` that surfaces as an
    # unraisable exception charged to whichever test was unlucky enough to be running.
    # It is Django's own error path and not this library's, so it is made deterministic
    # here rather than left to land on a stranger.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        messages = asyncio.run(drive())
        gc.collect()
    return messages


def body_of(messages: List[Dict[str, Any]]) -> bytes:
    return b"".join(message.get("body", b"") for message in messages)


def test_a_plain_async_view_is_audited(audited_ledger: Ledger) -> None:
    run_asgi("/async")

    assert [row["path"] for row in data_of(audited_ledger)] == ["/async"]


def test_a_plain_async_view_is_recorded_final(audited_ledger: Ledger) -> None:
    """Nothing is promised, so nothing is provisional."""
    run_asgi("/async")

    assert data_of(audited_ledger)[0]["completion"] == "final"


def test_a_plain_async_view_still_answers(audited_ledger: Ledger) -> None:
    messages = run_asgi("/async")

    assert body_of(messages) == b'{"ok": true}'


def test_an_async_stream_still_delivers_its_whole_body(audited_ledger: Ledger) -> None:
    """The blocker, and it is not subtle: wrapping the body in the sync ``SettlingBody``
    convinced Django the response was synchronous, and ``ASGIHandler`` then raised
    ``TypeError: 'async_generator' object is not iterable`` with the 200 already sent and
    no body behind it. An audit middleware that breaks streaming has stopped being an
    audit middleware."""
    messages = run_asgi("/async-stream")

    assert body_of(messages) == b'{"row": 0}\n{"row": 1}\n'


def test_a_completed_async_stream_is_recorded_provisional_then_settled(
    audited_ledger: Ledger,
) -> None:
    run_asgi("/async-stream")

    assert [row["kind"] for row in data_of(audited_ledger)] == [
        "request",
        "response.completion",
    ]


def test_a_completed_async_stream_settles_as_complete(audited_ledger: Ledger) -> None:
    run_asgi("/async-stream")

    assert data_of(audited_ledger)[1]["outcome"] == "complete"


def test_an_async_stream_that_dies_delivers_only_what_it_managed(
    audited_ledger: Ledger,
) -> None:
    messages = run_asgi("/broken-async-stream")

    assert body_of(messages) == b'{"row": 0}\n'


def test_an_async_stream_that_dies_is_not_recorded_as_a_finished_200(
    audited_ledger: Ledger,
) -> None:
    """The same overclaim the WSGI path was rebuilt to remove, on the path nobody tested:
    ``{'status': 200, 'completion': 'final', 'error': None}`` for a response the client
    received one chunk of."""
    run_asgi("/broken-async-stream")

    rows = data_of(audited_ledger)
    assert rows[0]["completion"] == "provisional"


def test_an_async_stream_that_dies_settles_as_incomplete(
    audited_ledger: Ledger,
) -> None:
    run_asgi("/broken-async-stream")

    assert data_of(audited_ledger)[1]["outcome"] == "incomplete"


def test_an_async_stream_that_dies_names_the_exception(audited_ledger: Ledger) -> None:
    run_asgi("/broken-async-stream")

    assert data_of(audited_ledger)[1]["error"] == "RuntimeError"


def test_an_async_stream_settles_exactly_once(audited_ledger: Ledger) -> None:
    """``ASGIHandler`` closes the response *and* ``aclosing()`` closes the iterator, so
    there are two plausible moments to settle in and a second settlement entry would be a
    second claim rather than a repeat."""
    run_asgi("/async-stream")

    assert [row["kind"] for row in data_of(audited_ledger)].count("response.completion") == 1


def test_an_async_settlement_reaches_the_orm_mirror(audited_ledger: Ledger) -> None:
    """The settlement is recorded from inside the event loop, and the mirror is the ORM.
    Writing to it from there raises ``SynchronousOnlyOperation`` unless the settle is put
    back on a thread Django considers safe."""
    run_asgi("/async-stream")

    assert list(AuditIndex.objects.order_by("sequence").values_list("kind", flat=True)) == [
        "request",
        "response.completion",
    ]
