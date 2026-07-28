"""The Django ``AppConfig`` that makes this a real, installable application."""

from django.apps import AppConfig

__all__ = ["SignLedgerConfig"]


class SignLedgerConfig(AppConfig):  # type: ignore[misc]
    """Registers the audit index model and its shipped migration.

    ``label`` is set explicitly. Django otherwise derives it from the last component of
    ``name``, which here is ``django`` - a label that collides with anything else called
    the same and reads as nonsense in ``migrate`` output. Every model in this app pins
    the same value in its ``Meta``, so the two can never drift apart.
    """

    name = "signledger.contrib.django"
    label = "signledger_django"
    verbose_name = "SignLedger audit"
    default_auto_field = "django.db.models.BigAutoField"
