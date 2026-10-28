# dsbf/store/migrations/env.py
"""Alembic environment: runs on the connection the repository passes in."""

from alembic import context

from dsbf.store.tables import metadata

connection = context.config.attributes["connection"]
context.configure(connection=connection, target_metadata=metadata)
with context.begin_transaction():
    context.run_migrations()
