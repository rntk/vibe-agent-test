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

# Install dependencies first for better layer caching
COPY --chown=appuser:appuser pyproject.toml poetry.lock ./
RUN poetry config virtualenvs.create false && \
    poetry install --no-root

# Copy source and install the package itself (registers the `cagent` console script)
COPY --chown=appuser:appuser . .
RUN poetry install

USER appuser

ENTRYPOINT ["cagent"]
