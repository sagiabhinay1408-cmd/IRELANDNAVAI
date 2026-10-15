
"""
Ireland Navigator AI — Shared Settings

Every phase imports configuration from this file.
This avoids repeating API keys and model settings.
"""

# os reads operating-system environment variables.
import os

# Path helps us locate files on macOS, Windows or Linux.
from pathlib import Path

# load_dotenv reads values from the project's .env file.
from dotenv import load_dotenv


# Find the project root.
#
# __file__ is the path to this settings.py file.
# parents[1] moves from config/ to IRELANDNAVAI/.

ROOT_DIR = Path(__file__).resolve().parents[1]


# Load the .env file from the project root.
#
# Existing environment variables take precedence.
# This is useful for deployment environments.

load_dotenv(ROOT_DIR / ".env")


# Qdrant Cloud connection settings.
QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")

# Name of the vector collection.
QDRANT_COLLECTION = os.getenv(
    "QDRANT_COLLECTION",
    "ireland_public_services",
)


# The same embedding model must be used when
# indexing documents and searching questions.

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)

# all-MiniLM-L6-v2 generates 384-dimensional vectors.
EMBEDDING_DIMENSION = 384


# NVIDIA's OpenAI-compatible API settings.
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")

NVIDIA_BASE_URL = os.getenv(
    "NVIDIA_BASE_URL",
    "https://integrate.api.nvidia.com/v1",
)

NVIDIA_MODEL = os.getenv(
    "NVIDIA_MODEL",
    "nvidia/nemotron-3-ultra-550b-a55b",
)


# Retrieve at most five chunks for each question.
RETRIEVAL_TOP_K = 5

# This is an initial similarity heuristic.
# It is not proof that an answer is factually supported.
MIN_SIMILARITY = 0.45

# Keep the last six user-assistant exchanges.
MAX_HISTORY_MESSAGES = 12


def validate_configuration(require_nvidia=True):
    """
    Verify that required credentials exist.

    Parameters:
        require_nvidia:
            False for document ingestion, which only
            needs Qdrant.
            True for the RAG and agent phases.

    Raises:
        RuntimeError if required values are missing.
    """

    required = {
        "QDRANT_URL": QDRANT_URL,
        "QDRANT_API_KEY": QDRANT_API_KEY,
    }

    if require_nvidia:
        required["NVIDIA_API_KEY"] = NVIDIA_API_KEY

    # Collect variable names whose values are empty.
    missing = [
        name
        for name, value in required.items()
        if not value
    ]

    if missing:
        raise RuntimeError(
            "Missing .env settings: " + ", ".join(missing)
        )

    return True
