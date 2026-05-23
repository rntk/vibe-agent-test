# cagent

`cagent` is an experimental vibe coding agent. It employs a dual-model architecture where a small, fast model executes the heavy lifting, while a larger, smart model acts as an overseer to guide the process, review edits, and ensure quality.

## Quick start

### Prerequisites

- Docker (or Python 3.12+ with Poetry)
- An API key for at least the fast model (OpenAI, Anthropic, Gemini, or llama.cpp)

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

## Modes

### Plan mode (`-p` / `--plan`)

Analyzes a task file and produces an implementation plan saved to `plans/plan_<timestamp>.md`.

Tools available: `bash` (read-only exploration), `advisor`

```bash
docker run --rm -it -v "$(pwd):/app" --env-file .env cagent:latest -p task.md
```

### Implementation mode (`-i` / `--implementation`)

Executes a plan or prompt file using read, bash, write, and search-and-replace tools.

Tools available: `bash`, `advisor`, `write_file`, `search_and_replace`

```bash
docker run --rm -it -v "$(pwd):/app" --env-file .env cagent:latest -i plan.md
```

### Web mode (`-w` / `--web`)

Starts an interactive web UI with a real-time conversation view. Supports pause/resume, message send, and message deletion. Optionally seed the conversation with an initial task file.

```bash
docker run --rm -it \
  -v "$(pwd):/app" \
  --env-file .env \
  -p 8765:8765 \
  cagent:latest -w
```

With an initial task file:

```bash
docker run --rm -it \
  -v "$(pwd):/app" \
  --env-file .env \
  -p 8765:8765 \
  cagent:latest -w task.md
```

Custom host/port:

```bash
docker run --rm -it \
  -v "$(pwd):/app" \
  --env-file .env \
  -p 8080:8080 \
  cagent:latest -w --web-host 0.0.0.0 --web-port 8080
```

### Default mode (no flags)

Sends a simple "Hello, World!" prompt to the configured model (smart first, fast second, or echo fallback):

```bash
docker run --rm -it --env-file .env cagent:latest
```

## CLI flags

| Flag | Description |
|------|-------------|
| `-p FILE` / `--plan FILE` | Plan mode — analyze a task and produce an implementation plan |
| `-i FILE` / `--implementation FILE` | Implementation mode — execute a plan/prompt with write tools |
| `-w [FILE]` / `--web [FILE]` | Interactive web mode (optionally seed with a task file) |
| `--web-host HOST` | Host to bind the web UI to (default: `127.0.0.1`) |
| `--web-port PORT` | Port to bind the web UI to (default: `8765`) |
| `--trace [FILE]` | Enable debug tracing; optionally write JSON trace to FILE |
| `--bash-advisor MODE` | Bash pre-execution advisor: `off`, `fast`, or `smart` (default: `off`) |
| `--tool-summary` | Enable LLM summarization of tool call steps (default: disabled) |

### Enable bash advisor

Add `--bash-advisor fast` or `--bash-advisor smart` to pre-review bash commands before execution:

```bash
docker run --rm -it -v "$(pwd):/app" --env-file .env cagent:latest -i plan.md --bash-advisor smart
```

### Enable tool summaries

Add `--tool-summary` to insert LLM-generated checkpoints after each iteration:

```bash
docker run --rm -it -v "$(pwd):/app" --env-file .env cagent:latest -i plan.md --tool-summary
```

### Debug tracing

Use `--trace` to capture a detailed span tree of LLM calls and tool executions, written as JSON and as a readable HTML view:

```bash
docker run --rm -it -v "$(pwd):/app" --env-file .env cagent:latest -i plan.md --trace trace.json
```

This produces `trace.json` (JSON span tree) and `trace.html` (readable conversation view).

## Tools

The agent uses the following built-in tools depending on the mode:

| Tool | Plan mode | Implementation mode | Web mode | Description |
|------|-----------|---------------------|----------|-------------|
| `bash` | ✅ | ✅ | ✅ | Run a shell command with stdout/stderr capture |
| `advisor` | ✅ | ✅ | ✅ | Ask a smarter model for concise coding guidance |
| `write_file` | — | ✅ | ✅ | Create, overwrite, append, or replace a line range in a file |
| `search_and_replace` | — | ✅ | ✅ | Replace an exact substring exactly once in a file |

## Advisor system

The advisor is a multi-layered quality and safety system:

1. **Bash precheck** (`--bash-advisor`) — Reviews bash commands before execution to flag risky operations (destructive commands, interactive prompts, likely hangs).

2. **Edit precheck** — Automatically reviews file writes and search-and-replace edits against the smart model to catch architectural problems before they are applied.

3. **Tool failure advisor** — When a bash command fails (non-zero exit code, stderr output, or timeout), the advisor diagnoses the error and provides a concise hint.

The advisor can use either the `fast` or `smart` model depending on the `--bash-advisor` setting.

## Configuration

Copy `.env.example` to `.env` and fill in your provider details:

| Variable | Purpose |
|----------|---------|
| `FAST_API_TYPE` | Provider for the fast model: `openai`, `anthropic`, `llamacpp`, or `gemini` |
| `FAST_API_HOST` | Base URL (required for `llamacpp`; optional for others) |
| `FAST_API_TOKEN` | API key |
| `FAST_API_MODEL` | Model name (default: `gpt-4o` for OpenAI, `claude-3-5-sonnet-20241022` for Anthropic, `gemini-2.0-flash` for Gemini, `moonshotai/Kimi-K2.5` for llama.cpp) |
| `SMART_API_TYPE` | Provider for the smart / advisor model |
| `SMART_API_HOST` | Base URL for the smart model |
| `SMART_API_TOKEN` | API key for the smart model |
| `SMART_API_MODEL` | Model name for the smart model |

**Note:** For `gemini`, if the model name contains `thinking`, the API uses `v1alpha` for thinking-enabled models. For `llamacpp`, only `HOST` is required (token and model are optional).

## Architecture

```
User task
    │
    ▼
┌─────────────────────────────────────────────────┐
│                  Agent loop                      │
│  (up to 20 iterations, configurable per mode)   │
│                                                 │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │  Fast    │    │  Smart   │    │  Bash    │  │
│  │  Model   │◄──►│  Model   │◄──►│ Advisor  │  │
│  │(executor)│    │(overseer)│    │(optional)│  │
│  └──────────┘    └──────────┘    └──────────┘  │
│       │                              ▲         │
│       ▼                              │         │
│  ┌──────────┐                  ┌──────────┐    │
│  │  Tools   │──────result─────►│ Advisor  │    │
│  │(bash,    │                  │(on fail) │    │
│  │ write,   │                  └──────────┘    │
│  │ search)  │                                  │
│  └──────────┘                                  │
│       │                                         │
│       ▼                                         │
│  ┌──────────┐                                  │
│  │Compaction│ (deduplicates tool results)      │
│  └──────────┘                                  │
└─────────────────────────────────────────────────┘
    │
    ▼
 Final result (plan file or printed output)
```

Key components:
- **Fast model** — Executes the agent loop: plans, writes code, runs commands.
- **Smart model** — Acts as advisor: reviews bash commands pre-execution, reviews edits pre-apply, diagnoses tool failures, and answers `advisor` tool calls.
- **Compaction** — Automatically deduplicates stale tool results with similar arguments to keep context within budget.
- **Tracing** — Span-based instrumentation across all LLM calls and tool executions, with JSON and HTML output.
- **Operating contract** — Every agent follows a shared operating contract: reason before acting, treat content as data not policy, plan before executing, and stop at budgets.

## Context costs & best practices

The file [`AGENTS.md`](AGENTS.md) documents best practices for coding agent architecture. The file [`cagent/bp.md`](cagent/bp.md) compares the current implementation against those practices with prioritized improvement recommendations.

## Development commands

```bash
# Build the dev image (includes bash and full tooling)
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

### Without Docker

```bash
# Install dependencies
poetry install

# Run linter
poetry run ruff check .

# Run formatter
poetry run ruff format .

# Run tests with coverage
poetry run pytest

# Run the CLI
poetry run cagent -p task.md
```
