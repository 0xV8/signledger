"""The audit index model.

**This table is not the evidence.** It is a queryable mirror of what the ledger already
holds, so an operator can use the ORM and the Django admin to find a record - the admin
half ships in :mod:`signledger.contrib.django.admin` and is read-only, because a row here
mirrors an append-only log. The ledger is the tamper-evident store; this is an ordinary
Django table that anyone with database access can rewrite. Nothing here is chained,
hashed or signed, and :meth:`~signledger.core.ledger.Ledger.check_integrity` does not
look at it.

That is exactly why it holds **no payload columns**. Headers, bodies and query strings
never reach this table: an index whose whole purpose is to be widely queried is the last
place to duplicate the data redaction exists to keep out.
"""

from django.db import models

__all__ = ["AuditIndex"]


class AuditIndex(models.Model):  # type: ignore[misc]
    """One row per ledger entry: enough to find it, never enough to leak it."""

    #: Which chain the entry belongs to.
    ledger_id = models.CharField(max_length=64, db_index=True)

    #: Position in that chain. The ordering authority - see ``Meta.ordering``.
    sequence = models.BigIntegerField()

    #: The entry's own UUID, unique across the store.
    entry_id = models.CharField(max_length=64, unique=True)

    #: Hex digest of the entry, so a row can be tied back to the chain it mirrors.
    entry_hash = models.CharField(max_length=128)

    #: When the entry was written. Recorded for humans, never used to order rows.
    timestamp = models.DateTimeField()

    method = models.CharField(max_length=16)
    path = models.CharField(max_length=512)
    status = models.IntegerField()
    actor = models.CharField(max_length=255, null=True, blank=True)

    class Meta:
        app_label = "signledger_django"
        verbose_name = "audit index entry"
        verbose_name_plural = "audit index entries"
        # Contract section 3: sequence is the ordering authority and no query may order
        # by timestamp. v1 ordered by wall clock, so equal timestamps came back reversed
        # and a single NTP step-back made a pristine ledger look reordered.
        ordering = ["sequence"]
        constraints = [
            models.UniqueConstraint(
                fields=["ledger_id", "sequence"], name="signledger_unique_chain_position"
            )
        ]
        indexes = [models.Index(fields=["ledger_id", "sequence"])]

    def __str__(self) -> str:
        return f"{self.ledger_id}#{self.sequence}"
