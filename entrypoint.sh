#!/bin/sh
set -eu

echo "Waiting for postgres..."

while ! nc -z $POSTGRES_HOST 5432; do
  sleep 0.1
done

echo "PostgreSQL started"

poetry run alembic upgrade head

exec poetry run "$@"
