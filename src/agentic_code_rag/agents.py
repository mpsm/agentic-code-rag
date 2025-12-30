import logging
import pathlib

from strands import Agent
from strands.models.ollama import OllamaModel

from .config import (
    OLLAMA_HOST,
    OLLAMA_MODEL_ID,
    OLLAMA_NUM_CTX,
)

logger = logging.getLogger("agentic-code-rag")

SYSTEM_PROMPT_DESCRIPTION = """
You are transcribing a code into human readable description. The description is technical but you
aim to capture both the technical aspect as well as the intent behind the code.
"""

SYSTEM_PROMPT_ANSWER = """
You are answering user questions based on the knowledge provided in the context.
"""


def create_model() -> OllamaModel:
    return OllamaModel(
        host=OLLAMA_HOST,
        model_id=OLLAMA_MODEL_ID,
        options={"num_ctx": OLLAMA_NUM_CTX},
    )


def create_description_agent() -> Agent:
    """Create and return an Ollama agent configured for code description tasks."""

    ollama_model = create_model()
    agent = Agent(
        model=ollama_model,
        system_prompt=SYSTEM_PROMPT_DESCRIPTION,
        callback_handler=None,
    )
    return agent


def create_answering_agent() -> Agent:
    """Create and return an Ollama agent configured for answering user questions based on knowledge."""

    ollama_model = create_model()
    agent = Agent(
        model=ollama_model, system_prompt=SYSTEM_PROMPT_ANSWER, callback_handler=None
    )
    return agent


def agent_response_to_text(response) -> str:
    """Extract and return the text content from an agent response message."""
    first_message = response.message["content"][0]

    response_text = first_message.get("text") or ""
    if not response_text:
        logger.warning("No text in response")

    return response_text


def describe(file_path: pathlib.Path) -> str:
    """Read a file, generate a description using the description agent, and return the description."""
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


def answer_question(query: str, knowledge_text: str) -> str:
    """Generate an answer to the user's question using the answering agent and retrieved knowledge."""
    agent = create_answering_agent()

    agent_response = agent(
        prompt=f'USER_QUESTION: "{query}\n\n KNOWLEDGE: \n{knowledge_text}'
    )

    answer = agent_response_to_text(agent_response)

    return answer
