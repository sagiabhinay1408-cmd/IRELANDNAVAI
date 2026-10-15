
"""
Phase 1 — Verify Project Setup
"""

from config.settings import (
    ROOT_DIR,
    QDRANT_COLLECTION,
    EMBEDDING_MODEL,
    NVIDIA_MODEL,
    validate_configuration,
)


def main():
    """Check imports and configuration."""

    print("=" * 50)
    print("IRELAND NAVIGATOR AI — PHASE 1")
    print("=" * 50)

    # Import each major library to verify installation.
    import requests
    import bs4
    import openai
    import qdrant_client
    import sentence_transformers
    import langgraph
    import streamlit

    print("[PASS] Dependencies imported")

    # Never print actual API keys.
    print("Project root:", ROOT_DIR)
    print("Qdrant collection:", QDRANT_COLLECTION)
    print("Embedding model:", EMBEDDING_MODEL)
    print("NVIDIA model:", NVIDIA_MODEL)

    # Check required environment variables.
    validate_configuration()

    print("[PASS] Configuration loaded")
    print("[PASS] Phase 1 completed")


# Execute main() only when running this file directly.
if __name__ == "__main__":
    main()
