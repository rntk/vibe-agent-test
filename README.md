# cagent

`cagent` is an experimental vibe coding agent. It employs a dual-model architecture where a small, fast model executes the heavy lifting, while a larger, smart model acts as an overseer to guide the process and ensure quality.

## Quick start

### Build the image

```bash
docker build -t cagent:latest -f Dockerfile .
```

### Run against any project

Mount the target project to `/app` inside the container and pass the required API configuration via environment variables:

```bash
docker run --rm -it \
  -v "$(pwd):/app" \
  -e FAST_API_TYPE=openai \
  -e FAST_API_HOST=https://api.openai.com \
  -e FAST_API_TOKEN="sk-..." \
  -e FAST_API_MODEL="gpt-4o-mini" \
  -e SMART_API_TYPE=openai \
  -e SMART_API_TOKEN="sk-..." \
  -e SMART_API_MODEL="gpt-4o" \
  cagent:latest -p task.md
```

Or use an `.env` file:

```bash
docker run --rm -it \
  -v "$(pwd):/app" \
  --env-file .env \
  cagent:latest -p task.md
```

### Modes

**Plan mode** — analyzes a task file and produces an implementation plan saved to `plans/plan_<timestamp>.md`:

```bash
docker run --rm -it -v "$(pwd):/app" --env-file .env cagent:latest -p task.md
```

**Implementation mode** — executes a plan or prompt file using read, bash, and write tools:

```bash
docker run --rm -it -v "$(pwd):/app" --env-file .env cagent:latest -i plan.md
```

**Default mode** (no flags) — sends a simple "Hello, World!" prompt to the configured model:

```bash
docker run --rm -it --env-file .env cagent:latest
```

### Enable bash advisor

Add `--bash-advisor fast` or `--bash-advisor smart` to pre-review bash commands before execution:

```bash
docker run --rm -it -v "$(pwd):/app" --env-file .env cagent:latest -i plan.md --bash-advisor smart
```

## Configuration

Copy `.env.example` to `.env` and fill in your provider details:

| Variable | Purpose |
|----------|---------|
| `FAST_API_TYPE` | Provider for the fast model: `openai`, `anthropic`, `llamacpp`, or `gemini` |
| `FAST_API_HOST` | Base URL (required for `llamacpp`; optional for others) |
| `FAST_API_TOKEN` | API key |
| `FAST_API_MODEL` | Model name |
| `SMART_API_TYPE` | Provider for the smart / advisor model |
| `SMART_API_HOST` | Base URL for the smart model |
| `SMART_API_TOKEN` | API key for the smart model |
| `SMART_API_MODEL` | Model name for the smart model |

## Development commands

```bash
# Build the dev image
docker build -t cagent:dev -f Dockerfile.dev .

# Run tests
docker run --rm cagent:dev pytest

# Run linter
docker run --rm cagent:dev ruff check .

# Run formatter
docker run --rm cagent:dev ruff format .

# Check formatting without changes
docker run --rm cagent:dev ruff format --check .
```
