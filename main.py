import logging
import os
import pathlib
import uuid
from dataclasses import dataclass

import click
from qdrant_client import QdrantClient
from qdrant_client.conversions.common_types import PointStruct, VectorParams
from qdrant_client.conversions.common_types import QueryResponse as QdrantQueryResponse
from qdrant_client.models import Distance, Filter, IsEmptyCondition, PayloadField
from rich.console import Console
from rich.logging import RichHandler
from rich.markdown import Markdown
from strands import Agent
from strands.models.ollama import OllamaModel

logger = logging.getLogger("agentic-code-rag")

OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL_ID = "qwen3-coder:30b"
OLLAMA_NUM_CTX = 16000
OLLAMA_EMBEDDING_MODEL = "qwen3-embedding"
QDRANT_HOST = "http://localhost:6333"
VECTOR_SIZE = 4096
SCAN_EXTENSIONS = [".py", ".rs", ".ts"]
QUERY_LIMIT = 5
LIST_LIMIT = 100
LIST_DESCRIPTION_PREVIEW = 200
SYSTEM_PROMPT_DESCRIPTION = """
You are transcribing a code into human readable description. The description is technical but you
aim to capture both the technical aspect as well as the intent behind the code.
"""
SYSTEM_PROMPT_ANSWER = """
You are answering user questions based on the knowledge provided in the context.
"""


@dataclass
class CLIContext:
    project_name: str
    directory_path: pathlib.Path
    vectordb: QdrantClient


def setup_logging(log_level: str) -> None:
    valid_levels = set(logging.getLevelNamesMapping().keys())
    if log_level.upper() not in valid_levels:
        valid_level_list = ", ".join(valid_levels)
        raise click.BadParameter(
            f"Invalid log level '{log_level}'. Valid levels are: {valid_level_list}"
        )

    console = Console()
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(message)s",
        handlers=[RichHandler(console=console, show_time=False, show_level=True)],
    )

    logger.debug(f"Log level set to {log_level.upper()}")


def create_description_agent() -> Agent:
    ollama_model = OllamaModel(
        host=OLLAMA_HOST,
        model_id=OLLAMA_MODEL_ID,
        options={"num_ctx": OLLAMA_NUM_CTX},
    )
    agent = Agent(
        model=ollama_model,
        system_prompt=SYSTEM_PROMPT_DESCRIPTION,
        callback_handler=None,
    )
    return agent


def create_answering_agent() -> Agent:
    ollama_model = OllamaModel(
        host=OLLAMA_HOST,
        model_id=OLLAMA_MODEL_ID,
        options={"num_ctx": OLLAMA_NUM_CTX},
    )
    agent = Agent(
        model=ollama_model, system_prompt=SYSTEM_PROMPT_ANSWER, callback_handler=None
    )
    return agent


def agent_response_to_text(response) -> str:
    first_message = response.message["content"][0]

    response_text = first_message.get("text") or ""
    if not response_text:
        logger.warning("No text in response")

    return response_text


def describe(file_path: pathlib.Path) -> str:
    agent = create_description_agent()

    file_contents = file_path.read_text()
    logger.info(f"Read file {file_path}, length: {len(file_contents)}")

    response = agent(prompt=file_contents)

    for i, block in enumerate(response.message["content"]):
        print(f"Block {i}: {block.keys()}")

    response_text = agent_response_to_text(response)

    latency = response.metrics.accumulated_metrics["latencyMs"]
    input_tokens = response.metrics.accumulated_usage["inputTokens"]
    output_tokens = response.metrics.accumulated_usage["outputTokens"]
    total_tokens = input_tokens + output_tokens
    logger.info(
        f"Response: {output_tokens} output tokens, {input_tokens} input tokens, {total_tokens} total"
    )
    logger.info(f"Response time: {latency}; length: {len(response_text)}")

    return response_text


def calculate_vector(text: str):
    import ollama

    embedding_response = ollama.embed(model=OLLAMA_EMBEDDING_MODEL, input=text)
    embeddings = embedding_response["embeddings"][0]
    logger.debug(f"Calculated embedding, len={len(embeddings)}")

    return embeddings


def scan_project(
    project_name: str, scan_directory: pathlib.Path, vectordb: QdrantClient
):
    logger.info(f"Scanning {scan_directory}")

    # Iterate over all files with specified extensions
    for file_path in scan_directory.rglob("*"):
        if file_path.is_file() and file_path.suffix in SCAN_EXTENSIONS:
            # Skip hidden files and directories
            if any(part.startswith(".") for part in file_path.parts):
                continue
            logger.info(f"Found file: {file_path}")
            description = describe(file_path)
            vector = calculate_vector(description)
            vectordb.upsert(
                collection_name=project_name,
                points=[
                    PointStruct(
                        id=uuid.uuid4(),
                        vector=vector,
                        payload={
                            "file": file_path,
                            "description": description,
                        },
                    )
                ],
            )
            logger.info("Entry successfully added to the database")


def init_vector_database(project_name: str) -> QdrantClient:
    vectordb = QdrantClient(QDRANT_HOST)

    try:
        vectordb.get_collection(project_name)
    except Exception:
        logger.warning(f"Collection {project_name} does not exists, creating")
        vectordb.create_collection(
            collection_name=project_name,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        logger.info(f"Created collection {project_name}, vector size {VECTOR_SIZE}")

    return vectordb


def answer_question(query: str, knowledge: QdrantQueryResponse) -> str:
    agent = create_answering_agent()

    knowledge_text = "\n".join(
        [point.payload["description"] for point in knowledge.points if point.payload]
    )

    agent_response = agent(
        prompt=f'USER_QUESTION: "{query}\\n\n KNOWLEDGE: \n{knowledge_text}'
    )

    answer = agent_response_to_text(agent_response)

    return answer


@click.group()
@click.option(
    "-v", "--verbose", is_flag=True, help="Enable verbose output (INFO level)"
)
@click.option("-q", "--quiet", is_flag=True, help="Enable quiet mode (WARNING level)")
@click.option("--project_name", "-p", default=None, help="Set project name")
@click.option(
    "-s",
    "--source_directory",
    type=click.Path(exists=True, file_okay=False),
    default=".",
)
@click.pass_context
def cli(ctx, verbose, quiet, project_name, source_directory):
    if verbose and quiet:
        raise click.BadParameter("Cannot use both -v/--verbose and -q/--quiet")

    if quiet:
        log_level = "WARNING"
    elif verbose:
        log_level = "INFO"
    else:
        log_level = "ERROR"

    setup_logging(log_level)

    # Normalize the directory path to CWD if it's "."
    if source_directory == ".":
        source_directory = os.getcwd()

    directory_path = pathlib.Path(source_directory)

    safe_project_name: str = project_name or directory_path.name
    vectordb = init_vector_database(project_name)

    ctx.obj = CLIContext(
        project_name=safe_project_name,
        directory_path=directory_path,
        vectordb=vectordb,
    )


@cli.command()
@click.pass_context
def scan(ctx) -> None:
    context: CLIContext = ctx.obj
    project_name = context.project_name
    directory_path = context.directory_path
    vectordb = context.vectordb

    if project_name is None:
        project_name = directory_path.name
    logger.info(f"Project name: {project_name}")
    logger.info(f"Main called with directory: {directory_path}")

    scan_project(project_name, directory_path, vectordb)


@cli.command()
@click.argument("query", required=True)
@click.pass_context
def query(ctx, query: str) -> None:
    context: CLIContext = ctx.obj
    project_name = context.project_name
    vectordb = context.vectordb

    logger.info(f"Called query ({query}) on project {project_name}")

    vector = calculate_vector(text=query)

    search_results = vectordb.query_points(
        collection_name=project_name,
        query=vector,
        limit=QUERY_LIMIT,
        query_filter=Filter(
            must_not=[IsEmptyCondition(is_empty=PayloadField(key="description"))]
        ),
    )

    logger.info(f"Found {len(search_results.points)} results.")

    for i, result in enumerate(search_results.points, 1):
        print(f"\n--- Result {i} (Score: {result.score:.4f}) ---")
        print(f"ID: {result.id}")
        if result.payload:
            print(f"File: {result.payload.get('file_path', 'N/A')}")
            print(
                f"Description: {result.payload.get('description', 'N/A')[:LIST_DESCRIPTION_PREVIEW]}..."
            )

    answer = answer_question(query, search_results)

    console = Console()
    markdown = Markdown(answer)
    console.print(markdown)


@cli.command()
@click.pass_context
def destroy(ctx) -> None:
    context: CLIContext = ctx.obj
    project_name = context.project_name
    vectordb = context.vectordb

    logger.info(f"Destroying collection {project_name}")

    vectordb.delete_collection(collection_name=project_name)
    logger.info(f"Collection {project_name} destroyed")


@cli.command()
@click.pass_context
def list_collections(ctx) -> None:
    context: CLIContext = ctx.obj
    vectordb = context.vectordb

    logger.info("Listing collections")

    collections = vectordb.get_collections()
    for collection in collections.collections:
        info = vectordb.get_collection(collection.name)
        points_count = info.points_count
        print(f"\n=== {collection.name}: {points_count} entries ===")

        points = vectordb.scroll(
            collection_name=collection.name,
            limit=LIST_LIMIT,
        )[0]

        for point in points:
            if point.payload:
                file_path = point.payload.get("file", "N/A")
                description = point.payload.get("description", "")
                desc_preview = (
                    description[:LIST_DESCRIPTION_PREVIEW].replace("\n", " ") + "..."
                    if len(description) > LIST_DESCRIPTION_PREVIEW
                    else description.replace("\n", " ")
                )
                print(f"  - {file_path}")
                print(f"    {desc_preview}")
                print()


if __name__ == "__main__":
    cli()
