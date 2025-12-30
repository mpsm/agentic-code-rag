import pathlib
from dataclasses import dataclass

from qdrant_client import QdrantClient


@dataclass
class CLIContext:
    """Context passed to CLI commands containing project configuration and database connection."""

    project_name: str
    directory_path: pathlib.Path
    vectordb: QdrantClient
