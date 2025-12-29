# Agentic Code RAG

> **Note from the author:** This project is a sandbox for educational purposes to experiment with building agentic RAGs for codebases. The goal is to solve problems of managing changes of knowledge and lazy insertion.

Code documentation tool that uses LLMs to generate descriptions and enable natural language queries.

## Requirements

- Python 3.12+
- Docker Compose (for infrastructure)

## Infrastructure

This project includes Docker Compose configuration for required services:

| Service | Port | Purpose |
|---------|------|---------|
| Ollama | 11434 | LLM model and embeddings |
| Qdrant | 6333 | Vector database |

### Starting Services

```bash
docker compose up -d
```

### Stopping Services

```bash
docker compose down
```

### Pulling Models

After starting Ollama, pull the required models:

```bash
docker exec ollama ollama pull qwen3-coder:30b
docker exec ollama ollama pull qwen3-embedding
```

## Developing

This project uses [Poethepoet](https://poethepoet.nat-n.net/) for task automation.

| Command | Description |
|---------|-------------|
| `poe lint` | Run ruff check |
| `poe format-check` | Check code formatting |
| `poe format` | Format code with ruff |
| `poe mypy` | Run type checking |
| `poe build` | Build package |
| `poe release` | Run lint, format-check, mypy, and build |

### Installation for Development

```bash
uv install --extra dev
```

## Installation

```bash
pip install -e .
```

## Usage

### Global Options

| Option | Description |
|--------|-------------|
| `-d, --debug` | Enable debug logging |
| `-p, --project_name` | Project name for vector database collection |
| `--source_directory PATH` | Source directory to scan (default: current directory) |

### Commands

#### scan

Scan source code and generate descriptions for embedding in the vector database.

```bash
uv run main.py scan [--project_name NAME] [--source_directory PATH]
```

**Examples:**

```bash
# Scan current directory with default project name
uv run main.py scan

# Scan specific directory with custom project name
uv run main.py scan --project_name myproject --source_directory /path/to/code
```

#### query

Query the codebase using natural language.

```bash
uv run main.py query "your question here" [--project_name NAME] [--source_directory PATH]
```

**Examples:**

```bash
# Query current project
uv run main.py query "How does authentication work?"

# Query specific project
uv run main.py query "What are the main components?" --project_name myproject
```

## Configuration

Default settings in `main.py`:
- Ollama host: `http://localhost:11434`
- Qdrant host: `http://localhost:6333`
- Model: `qwen3-coder:30b`
- Embedding model: `qwen3-embedding`
- Scanned extensions: `.py`, `.rs`, `.ts`
