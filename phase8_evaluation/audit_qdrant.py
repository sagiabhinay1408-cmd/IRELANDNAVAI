
"""
Phase 8 — Qdrant Metadata Audit

Read-only audit of the existing Qdrant collection.
Does not delete, modify, or upload any documents.
"""

from collections import Counter

from qdrant_client import QdrantClient

from config.settings import (
    QDRANT_URL,
    QDRANT_API_KEY,
    QDRANT_COLLECTION,
)


def audit_qdrant():
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=30,
    )

    counts = Counter()
    topics = Counter()
    incomplete_ids = []

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
            counts["total"] += 1

            payload = point.payload or {}
            metadata = payload.get("metadata")

            if not isinstance(metadata, dict):
                metadata = {}

            source = metadata.get("source")
            topic = metadata.get("topic")
            chunk_index = metadata.get("chunk_index")
            source_type = metadata.get("source_type")
            text = payload.get("page_content")

            if topic:
                topics[str(topic)] += 1

            missing = []

            if not source:
                missing.append("source")

            if not topic:
                missing.append("topic")

            if chunk_index is None:
                missing.append("chunk_index")

            if source_type != "official_website":
                missing.append("official_source_type")

            if not isinstance(text, str) or not text.strip():
                missing.append("page_content")

            if missing:
                counts["incomplete"] += 1
                incomplete_ids.append({
                    "id": str(point.id),
                    "missing": missing,
                })
            else:
                counts["complete"] += 1

        if offset is None:
            break

    print("\n=== QDRANT METADATA AUDIT ===")
    print("Total points:", counts["total"])
    print("Complete points:", counts["complete"])
    print("Incomplete points:", counts["incomplete"])

    print("\nTopics:")
    for topic, count in sorted(topics.items()):
        print(f"  {topic}: {count}")

    print("\nFirst 10 incomplete records:")
    for item in incomplete_ids[:10]:
        print(item)

    print("\nAudit finished. No data was modified.")


if __name__ == "__main__":
    audit_qdrant()
