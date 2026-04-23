# cagent

`cagent` is an experimental vibe coding agent. It employs a dual-model architecture where a small, fast model executes the heavy lifting, while a larger, smart model acts as an overseer to guide the process and ensure quality.

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
```
