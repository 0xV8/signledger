"""The Django admin for the audit index.

:mod:`~signledger.contrib.django.models` says the index exists "so an operator can use the
ORM and the Django admin to find a record". The ORM half shipped and the admin half did
not: ``AuditIndex`` was registered with no site and its URL 404ed. An operator with a UI
is the reason a mirror table is worth having at all, so the promise is kept here.

**Read-only, and not out of caution.** Every row mirrors an append-only log. A form that
could add, change or delete one would let this site produce an index that disagrees with
the evidence it claims to point at - and unlike the ledger, an ordinary Django table
cannot detect that it happened. Rows appear here because the middleware wrote an entry,
and for no other reason.

Autodiscovery imports this module, and only when ``django.contrib.admin`` is installed.
It stays importable when it is not: touching ``admin.site`` in a project without that app
raises ``LookupError: No installed app with label 'admin'``, and every module in this
package must import in every project that installs it.
"""

from typing import Any, List, Optional

from django.apps import apps as django_apps
from django.contrib import admin

from signledger.contrib.django.models import AuditIndex

__all__ = ["AuditIndexAdmin", "register"]


class AuditIndexAdmin(admin.ModelAdmin):  # type: ignore[misc]
    """A read-only window onto the audit index.

    ``list_display`` deliberately carries no payload column, because the model
    deliberately has none: the admin is the widest audience this data ever gets.
    """

    list_display = (
        "ledger_id",
        "sequence",
        "timestamp",
        "method",
        "path",
        "status",
        "actor",
    )
    list_filter = ("ledger_id", "method", "status")
    #: What an operator holding a ledger entry, a log line or an alert actually has.
    search_fields = ("entry_id", "entry_hash", "path", "actor")
    #: Contract section 3: ``sequence`` is the ordering authority. ``timestamp`` is shown
    #: because humans read it, and is not sortable because sorting by it is the v1 defect
    #: this table was rebuilt to avoid - equal timestamps came back reversed, and one NTP
    #: step-back made a pristine ledger look reordered.
    sortable_by = ("ledger_id", "sequence", "method", "path", "status", "actor")
    #: The bulk-action bar exists to change rows in bulk. There is nothing to offer.
    actions = None

    def get_readonly_fields(self, request: Any, obj: Any = None) -> List[str]:
        """Every field, always.

        Belt and braces with the permissions below: a field that is not named here
        renders as an editable widget the moment someone restores a change permission.
        """
        return [field.name for field in self.model._meta.fields]

    def has_add_permission(self, request: Any) -> bool:
        """No. There is no such thing as an audit entry that was not audited."""
        return False

    def has_change_permission(self, request: Any, obj: Any = None) -> bool:
        """No, for anyone, on any row: the entry it mirrors cannot change either."""
        return False

    def has_delete_permission(self, request: Any, obj: Any = None) -> bool:
        """No. Losing the mirror is recoverable; being able to prune it quietly, by
        hand, one row at a time, is exactly the capability an audit trail exists to
        deny."""
        return False


def register(site: Optional[Any] = None) -> bool:
    """Register :class:`AuditIndex` with ``site``, or with the default admin site.

    Returns whether the model is registered on that site afterwards. ``False`` means
    ``django.contrib.admin`` is not installed, which is a supported way to deploy this
    package and not an error - the index is still queryable through the ORM.

    Idempotent, because ``AdminSite.register`` raises ``AlreadyRegistered`` and there are
    several ways to arrive here: autodiscovery, an explicit call from a project's own
    ``admin.py`` that wants the model on a custom site, and an autoreload doing both
    again.
    """
    if not django_apps.is_installed("django.contrib.admin"):
        return False
    target = admin.site if site is None else site
    if not target.is_registered(AuditIndex):
        target.register(AuditIndex, AuditIndexAdmin)
    return True


register()
