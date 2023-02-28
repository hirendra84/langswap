import os

from logging.config import fileConfig

from sqlalchemy import engine_from_config
from sqlalchemy import pool

from alembic import context

PG_HOST = os.environ.get('PG_HOST', '127.0.0.1')
PG_PORT = os.environ.get('PG_PORT', 5432)
PG_USER = os.environ.get('PG_USER', 'joint_user')
PG_PASS = os.environ.get('PG_PASS', 'pass123')
PG_DBNAME = os.environ.get('PG_DBNAME', 'joint_dev')

SQLALCHEMY_DATABASE_URL = f"postgresql://{PG_USER}:{PG_PASS}@{PG_HOST}/{PG_DBNAME}"


# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
import api

config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
fileConfig(config.config_file_name)

# add your model's MetaData object here
# for 'autogenerate' support
# from myapp import mymodel
# target_metadata = mymodel.Base.metadata
target_metadata = api.models.Base.metadata


# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.

exclude_tables = ['spatial_ref_sys']


def include_object(object, name, type_, reflected, compare_to):
    if type_ == "table" and name in exclude_tables:
        return False
    else:
        return True


def run_migrations_offline():
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        include_object=include_object,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online():
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    connectable = engine_from_config(
        # config.get_section(config.config_ini_section),
        {
                'sqlalchemy.url': SQLALCHEMY_DATABASE_URL
        },
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata, include_object=include_object
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
