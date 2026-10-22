
"""
Inspect documents stored in the existing
Qdrant Cloud collection.

This script is read-only:
    - No ingestion
    - No collection deletion
    - No vector updates

It displays document IDs, topics, sources,
chunk indexes, and text previews.
"""

from phase3_vectorstore.vector_store import (
    create_services,
)

from config.settings import QDRANT_COLLECTION


def inspect_documents():
    _, client = create_services()

    offset = None
    total = 0

    while True:
        points, offset = client.scroll(
            collection_name=QDRANT_COLLECTION,
            limit=50,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        for point in points:
            payload = point.payload or {}
            metadata = payload.get("metadata", {})

            total += 1

            print("\n" + "-" * 60)
            print("ID:", point.id)
            print("Topic:", metadata.get("topic"))
            print("Source:", metadata.get("source"))
            print(
                "Chunk index:",
                metadata.get("chunk_index"),
            )
            print(
                "Text preview:",
                payload.get(
                    "page_content",
                    "",
                )[:300],
            )

        if offset is None:
            break

    print(f"\nTotal inspected chunks: {total}")


if __name__ == "__main__":
    inspect_documents()
