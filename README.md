# Agentic Code RAG

> **Note from the author:** This project is a sandbox for educational purposes to experiment with building agentic RAGs for codebases. The goal is to solve problems of managing changes of knowledge and lazy insertion.

Code documentation tool that uses LLMs to generate descriptions and enable natural language queries.

## Quick Start

### 1. Start Infrastructure

Start required services (Ollama, Qdrant, and Aspire Dashboard):

```bash
docker compose up -d
```

This starts:
- **Ollama** (port 11434) - LLM model and embeddings
- **Qdrant** (port 6333) - Vector database
- **Aspire Dashboard** (port 18888) - OpenTelemetry observability dashboard

Pull required models:

```bash
docker exec ollama ollama pull qwen3-coder:30b
docker exec ollama ollama pull qwen3-embedding
```

### 2. Install and Run

Run with `uv run`:

```bash
# Scan a codebase
uv run agentic-code-rag scan -p myproject -s /path/to/code

# Query the codebase
uv run agentic-code-rag query -p myproject "How does authentication work?"
```

## Usage

### Commands

#### scan

Scan source code and generate descriptions for embedding in the vector database.

```bash
agentic-code-rag scan [OPTIONS]
```

**Examples:**

```bash
# Scan current directory with default project name
agentic-code-rag scan

# Scan specific directory with custom project name
agentic-code-rag scan -p myproject -s /path/to/code

# With uv run
uv run agentic-code-rag scan -p myproject -s /path/to/code
```

#### query

Query the codebase using natural language.

```bash
agentic-code-rag query "your question here" [OPTIONS]
```

**Examples:**

```bash
# Query current project
agentic-code-rag query "How does authentication work?"

# Query specific project
agentic-code-rag query "What are the main components?" -p myproject

# With uv run
uv run agentic-code-rag query "How does authentication work?" -p myproject
```

#### destroy

Delete a collection from the vector database.

```bash
agentic-code-rag destroy [OPTIONS]
```

#### list-collections

List all collections in the vector database with their entry counts.

```bash
agentic-code-rag list-collections
```

### Global Options

| Option | Description |
|--------|-------------|
| `-v, --verbose` | Enable verbose output (INFO level) |
| `-q, --quiet` | Enable quiet mode (WARNING level) |
| `-d, --debug` | Enable debug mode (DEBUG level) |
| `-p, --project_name TEXT` | Project name for vector database collection (defaults to source directory name) |
| `-s, --source_directory PATH` | Source directory to scan (default: current directory) |

**Note:** If `--project_name` is not provided, the project name is automatically inferred from the source directory name.

## Requirements

- Python 3.12+
- Docker Compose (for infrastructure services)
- uv (for dependency management)

## Developing

This project uses [Poethepoet](https://poethepoet.nat-n.net/) for task automation and `uv` for dependency management.

### Setup

Install dependencies including dev tools:

```bash
uv sync
```

### Available Tasks

| Command | Description |
|---------|-------------|
| `poe lint` | Run ruff check |
| `poe format-check` | Check code formatting |
| `poe format` | Format code with ruff |
| `poe mypy` | Run type checking |
| `poe build` | Build package |
| `poe release` | Run lint, format-check, mypy, and build |

### Running During Development

After `uv sync`, run via `uv run`:

```bash
uv run agentic-code-rag scan -s /path/to/code
```

### Code Style

- Format: `poe format`
- Check: `poe format-check && poe lint && poe mypy`

## Configuration

Default settings in `src/agentic_code_rag/config.py`:
- Ollama host: `http://localhost:11434`
- Qdrant host: `http://localhost:6333`
- Model: `qwen3-coder:30b`
- Embedding model: `qwen3-embedding`
- Scanned extensions: `.py`, `.rs`, `.ts`
