
"""
Phase 8 — Qdrant Record Classification

Read-only diagnostic.

Classifies stored points into:
1. Complete
2. Missing chunk_index only
3. Missing other required fields

No records are deleted or modified.
"""

from collections import Counter

from qdrant_client import QdrantClient

from config.settings import (
    QDRANT_URL,
    QDRANT_API_KEY,
    QDRANT_COLLECTION,
)


def main():
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=30,
    )

    counts = Counter()
    examples = {
        "complete": [],
        "missing_chunk_index_only": [],
        "other_incomplete": [],
    }

    offset = None

    while True:
        points, offset = client.scroll(
            collection_name=QDRANT_COLLECTION,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        for point in points:
            payload = point.payload or {}
            metadata = payload.get("metadata") or {}

            if not isinstance(metadata, dict):
                metadata = {}

            source = metadata.get("source")
            topic = metadata.get("topic")
            source_type = metadata.get("source_type")
            chunk_index = metadata.get("chunk_index")
            content = payload.get("page_content")

            other_fields_valid = (
                bool(source)
                and bool(topic)
                and source_type == "official_website"
                and isinstance(content, str)
                and bool(content.strip())
            )

            if other_fields_valid:
                if chunk_index is None:
                    category = "missing_chunk_index_only"
                else:
                    category = "complete"
            else:
                category = "other_incomplete"

            counts[category] += 1

            if len(examples[category]) < 5:
                examples[category].append({
                    "id": str(point.id),
                    "topic": topic,
                    "source": source,
                    "chunk_index": chunk_index,
                    "source_type": source_type,
                    "text_preview": (
                        content[:120]
                        if isinstance(content, str)
                        else ""
                    ),
                })

        if offset is None:
            break

    print("\n=== QDRANT RECORD CLASSIFICATION ===")

    for category in examples:
        print(f"\n{category}: {counts[category]}")

        for example in examples[category]:
            print(example)

    print("\nTotal:", sum(counts.values()))
    print("No records modified.")


if __name__ == "__main__":
    main()
