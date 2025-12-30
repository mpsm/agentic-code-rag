import logging
import os
import pathlib
import uuid

import click
from qdrant_client import QdrantClient
from qdrant_client.conversions.common_types import (
    PointStruct,
)
from qdrant_client.models import Filter, IsEmptyCondition, PayloadField
from rich.console import Console
from rich.logging import RichHandler
from strands.telemetry import StrandsTelemetry

from .agents import answer_question, describe
from .config import (
    LIST_DESCRIPTION_PREVIEW,
    LIST_LIMIT,
    QUERY_LIMIT,
    SCAN_EXTENSIONS,
)
from .models import CLIContext
from .vectorstore import calculate_vector, init_vector_database

logger = logging.getLogger("agentic-code-rag")
console = Console()
StrandsTelemetry().setup_otlp_exporter().setup_meter(enable_otlp_exporter=True)


def setup_logging(log_level: str) -> None:
    """Configure Python logging with the specified level and Rich handler for pretty output."""
    valid_levels = set(logging.getLevelNamesMapping().keys())
    if log_level.upper() not in valid_levels:
        valid_level_list = ", ".join(valid_levels)
        raise click.BadParameter(
            f"Invalid log level '{log_level}'. Valid levels are: {valid_level_list}"
        )

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(message)s",
        handlers=[RichHandler(console=console, show_time=False, show_level=True)],
    )

    logger.debug(f"Log level set to {log_level.upper()}")


def scan_project(
    project_name: str, scan_directory: pathlib.Path, vectordb: QdrantClient
) -> None:
    """Scan all files in the directory, generate descriptions, compute vectors, and store in the database."""
    logger.info(f"Scanning {scan_directory}")

    for file_path in scan_directory.rglob("*"):
        if file_path.is_file() and file_path.suffix in SCAN_EXTENSIONS:
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


@click.group()
@click.option(
    "-v", "--verbose", is_flag=True, help="Enable verbose output (INFO level)"
)
@click.option("-q", "--quiet", is_flag=True, help="Enable quiet mode (WARNING level)")
@click.option("-d", "--debug", is_flag=True, help="Enable debug mode (DEBUG level)")
@click.option("--project_name", "-p", default=None, help="Set project name")
@click.option(
    "-s",
    "--source_directory",
    type=click.Path(exists=True, file_okay=False),
    default=".",
)
@click.pass_context
def cli(
    ctx,
    verbose: bool,
    quiet: bool,
    debug: bool,
    project_name: str | None,
    source_directory: str,
) -> None:
    """Main CLI entry point. Sets up logging and initializes the CLI context."""
    if verbose and quiet:
        raise click.BadParameter("Cannot use both -v/--verbose and -q/--quiet")

    if quiet:
        log_level = "WARNING"
    elif verbose:
        log_level = "INFO"
    elif debug:
        log_level = "DEBUG"
    else:
        log_level = "ERROR"

    setup_logging(log_level)

    if source_directory == ".":
        source_directory = os.getcwd()

    directory_path = pathlib.Path(source_directory)

    safe_project_name: str = project_name or directory_path.name
    vectordb = init_vector_database(safe_project_name)

    ctx.obj = CLIContext(
        project_name=safe_project_name,
        directory_path=directory_path,
        vectordb=vectordb,
    )


@cli.command()
@click.pass_context
def scan(ctx) -> None:
    """Scan the source directory and index all code files into the vector database."""
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
    """Search the vector database for relevant code descriptions and answer the query."""
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

    knowledge_text = "\n".join(
        [
            point.payload["description"]
            for point in search_results.points
            if point.payload
        ]
    )

    answer = answer_question(query, knowledge_text)

    from rich.markdown import Markdown

    markdown = Markdown(answer)
    console.print(markdown)


@cli.command()
@click.pass_context
def destroy(ctx) -> None:
    """Delete the project's collection from the vector database."""
    context: CLIContext = ctx.obj
    project_name = context.project_name
    vectordb = context.vectordb

    logger.info(f"Destroying collection {project_name}")

    vectordb.delete_collection(collection_name=project_name)
    logger.info(f"Collection {project_name} destroyed")


@cli.command()
@click.pass_context
def list_collections(ctx) -> None:
    """List all collections in the database with their entry counts and previews."""
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
