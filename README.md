# cagent

A Python CLI application managed by **Poetry**, linted and formatted with
**Ruff**, tested with **pytest** — all running inside Docker.

## Quick start

```bash
# Build and run the production image
docker build -t cagent:latest -f Dockerfile .
docker run --rm -it cagent:latest

# Build and run the dev image
docker build -t cagent:dev -f Dockerfile.dev .
docker run --rm -it cagent:dev
```

## Development commands

```bash
# Run tests
docker run --rm cagent:dev pytest

# Run linter
docker run --rm cagent:dev ruff check .

# Run formatter
docker run --rm cagent:dev ruff format .

# Check formatting without changes
docker run --rm cagent:dev ruff format --check .
```

## Project layout

```
cagent/
├── cagent/          # Application source
│   ├── __init__.py
│   ├── config.py
│   ├── tools.py
│   ├── tracing.py
│   └── llm/         # LLM client adapters
├── tests/           # pytest suite
├── main.py          # Entry-point
├── Dockerfile       # Production image
├── Dockerfile.dev   # Development image
└── pyproject.toml   # Poetry + Ruff + pytest config
```
