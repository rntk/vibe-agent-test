FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VERSION=1.8.3 \
    POETRY_NO_INTERACTION=1

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

RUN pip install "poetry==${POETRY_VERSION}"

RUN groupadd -r appuser -g 1000 && \
    useradd -r -g appuser -u 1000 -d /app -s /bin/bash appuser

WORKDIR /app

COPY --chown=appuser:appuser pyproject.toml poetry.lock ./

RUN poetry config virtualenvs.create false && \
    poetry install --no-root

COPY --chown=appuser:appuser . .

USER appuser

ENTRYPOINT [ "python" ]
