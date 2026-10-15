
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models


# Project configuration
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

COLLECTION_NAME = "ireland_public_services"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    print("=" * 55)
    print("IRELAND NAVIGATOR AI")
    print("Official Document Semantic Search")
    print("=" * 55)

    # Load embedding model
    print("\nLoading embedding model...")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    # Connect to Qdrant Cloud
    client = QdrantClient(
        url=os.environ["QDRANT_URL"],
        api_key=os.environ["QDRANT_API_KEY"],
        timeout=120,
    )

    # Connect to the existing vector collection
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
        content_payload_key="page_content",
        metadata_payload_key="metadata",
    )

    question = input(
        "\nEnter your question about Irish public services: "
    ).strip()

    if not question:
        print("Please enter a question.")
        return

    # Filter out the sample records from Step 5
    official_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="metadata.source_type",
                match=models.MatchValue(
                    value="official_website"
                ),
            )
        ]
    )

    print("\nSearching official documents...\n")

    results = vector_store.similarity_search_with_score(
        query=question,
        k=5,
        filter=official_filter,
    )

    if not results:
        print("No matching official documents found.")
        return

    for index, (document, score) in enumerate(results, 1):
        print(f"\n{'-' * 55}")
        print(f"RESULT {index}")
        print(f"{'-' * 55}")

        print("Similarity score:", round(score, 4))
        print("Title:", document.metadata.get("title"))
        print("Topic:", document.metadata.get("topic"))
        print("Source:", document.metadata.get("source"))
        print(
            "Retrieved:",
            document.metadata.get("retrieved_at")
        )

        print("\nContent:")
        print(document.page_content[:700])

    print("\nSearch completed successfully!")


if __name__ == "__main__":
    main()
