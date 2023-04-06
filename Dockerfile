FROM python:3.11

ENV PYTHONDONTWRITEBYTECODE 1 \
    PYTHONUNBUFFERED 1

RUN apt-get update \
    && apt-get install curl python3-dev libpq-dev netcat -y \
    && curl -sSL https://install.python-poetry.org | python -

ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

COPY pyproject.toml poetry.lock ./

RUN poetry config virtualenvs.create false \
    && poetry install --no-dev --no-interaction --no-ansi

COPY src/ alembic/ alembic.ini entrypoint.sh .

# run entrypoint.sh
ENTRYPOINT ["/app/entrypoint.sh"]
