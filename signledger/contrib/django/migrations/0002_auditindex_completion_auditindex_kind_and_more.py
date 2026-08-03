"""Columns the audit index needed to express what a v2.0.0 entry says.

``kind``, ``completion`` and ``request_id`` are what let the ORM and the read-only
admin - the query surface Django users are told to use - answer "which entries are
still only promises" and "which settlement belongs to which request". ``status``
becomes nullable because ``build_audit_record`` types it as optional, and a NOT NULL
column would turn that into a database error inside the one method that promises never
to raise.
"""

from django.db import migrations, models


class Migration(migrations.Migration):  # type: ignore[misc]

    dependencies = [
        ('signledger_django', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='auditindex',
            name='completion',
            field=models.CharField(db_index=True, default='final', max_length=16),
        ),
        migrations.AddField(
            model_name='auditindex',
            name='kind',
            field=models.CharField(db_index=True, default='request', max_length=32),
        ),
        migrations.AddField(
            model_name='auditindex',
            name='request_id',
            field=models.CharField(db_index=True, default='', max_length=64),
        ),
        migrations.AlterField(
            model_name='auditindex',
            name='status',
            field=models.IntegerField(blank=True, null=True),
        ),
    ]
