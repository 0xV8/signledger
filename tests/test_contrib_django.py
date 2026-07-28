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
import json
import pathlib
from typing import Any, Dict, Iterator, List

import django
import pytest
from django.conf import settings

if not settings.configured:
    settings.configure(
        DEBUG=False,
        DEBUG_PROPAGATE_EXCEPTIONS=False,
        SECRET_KEY="signledger-test-key",
        ALLOWED_HOSTS=["*"],
        USE_TZ=True,
        DATABASES={
            "default": {"ENGINE": "django.db.backends.sqlite3", "NAME": ":memory:"}
        },
        INSTALLED_APPS=["signledger.contrib.django"],
        MIDDLEWARE=["signledger.contrib.django.middleware.SignLedgerAuditMiddleware"],
        ROOT_URLCONF="test_contrib_django",
        DEFAULT_AUTO_FIELD="django.db.models.BigAutoField",
    )
    django.setup()

from django.core.exceptions import ImproperlyConfigured  # noqa: E402
from django.core.management import call_command  # noqa: E402
from django.core.management.base import SystemCheckError  # noqa: E402
from django.http import HttpResponse, JsonResponse  # noqa: E402
from django.test import Client, override_settings  # noqa: E402
from django.urls import path  # noqa: E402

from signledger.backends.memory import MemoryBackend  # noqa: E402
from signledger.contrib import FAIL_OPEN  # noqa: E402
from signledger.contrib.django import (  # noqa: E402
    SignLedgerDjangoConfigurationError,
    get_ledger,
    reset_ledger,
)
from signledger.contrib.django.apps import SignLedgerConfig  # noqa: E402
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
    with audited():
        Client().get("/items?page=2&api_key=sk-live-1")
        query = entries()[0].data["query"]
        assert (query["api_key"], query["page"]) == (REDACTED, ["2"])


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
    """The auth app is not installed here, so ``request.user`` does not exist."""
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


AUDIT_MIDDLEWARE = "signledger.contrib.django.middleware.SignLedgerAuditMiddleware"


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


def test_an_index_write_failure_under_fail_closed_fails_the_request() -> None:
    """A real unique-constraint violation from real SQLite, not a simulated one."""
    with audited():
        Client().get("/items")
        entry = entries()[0]
        AuditIndex.objects.all().delete()
        AuditIndex.objects.create(
            ledger_id=entry.ledger_id,
            sequence=1,
            entry_id="pre-existing",
            entry_hash="0" * 64,
            timestamp=entry.timestamp,
            method="GET",
            path="/items",
            status=200,
            actor=None,
        )
        assert Client(raise_request_exception=False).get("/items").status_code == 500


def test_an_index_write_failure_under_fail_open_still_serves_the_request() -> None:
    """The evidence is already in the ledger. A convenience table must not 500 a request."""
    with audited(AUDIT_FAILURE_POLICY=FAIL_OPEN):
        Client().get("/items")
        entry = entries()[0]
        AuditIndex.objects.all().delete()
        AuditIndex.objects.create(
            ledger_id=entry.ledger_id,
            sequence=1,
            entry_id="pre-existing",
            entry_hash="0" * 64,
            timestamp=entry.timestamp,
            method="GET",
            path="/items",
            status=200,
            actor=None,
        )
        assert Client().get("/items").status_code == 200


def test_an_index_write_failure_is_counted_rather_than_swallowed() -> None:
    with audited(AUDIT_FAILURE_POLICY=FAIL_OPEN):
        Client().get("/items")
        entry = entries()[0]
        AuditIndex.objects.all().delete()
        AuditIndex.objects.create(
            ledger_id=entry.ledger_id,
            sequence=1,
            entry_id="pre-existing",
            entry_hash="0" * 64,
            timestamp=entry.timestamp,
            method="GET",
            path="/items",
            status=200,
            actor=None,
        )
        Client().get("/items")
        assert get_ledger().count() == 2


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
