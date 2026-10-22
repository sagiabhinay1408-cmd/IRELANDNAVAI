
"""
Phase 8 — Review Retrieved Documents

Displays the complete text of documents retrieved during
the first retrieval evaluation.

Reads the existing retrieval report and Qdrant collection.

Read-only: does not modify Qdrant.
"""

import json
from pathlib import Path

from qdrant_client import QdrantClient

from config.settings import (
    QDRANT_URL,
    QDRANT_API_KEY,
    QDRANT_COLLECTION,
)


ROOT = Path(__file__).resolve().parents[1]

REPORT_PATH = (
    ROOT
    / "reports"
    / "phase8"
    / "retrieval_report.json"
)


def main():
    if not REPORT_PATH.exists():
        raise FileNotFoundError(
            f"Retrieval report not found: {REPORT_PATH}"
        )

    with REPORT_PATH.open("r", encoding="utf-8") as file:
        report = json.load(file)

    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=30,
    )

    for result in report["results"]:
        print("\n" + "=" * 75)
        print("QUESTION:", result["question"])
        print("=" * 75)

        document_ids = result["retrieved_document_ids"]

        points = client.retrieve(
            collection_name=QDRANT_COLLECTION,
            ids=document_ids,
            with_payload=True,
            with_vectors=False,
        )

        points_by_id = {
            str(point.id): point
            for point in points
        }

        for rank, document_id in enumerate(document_ids, start=1):
            point = points_by_id.get(document_id)

            if point is None:
                print(f"\nRank {rank}: Document not found")
                continue

            payload = point.payload or {}
            metadata = payload.get("metadata") or {}

            print(f"\n{'-' * 75}")
            print("RANK:", rank)
            print("ID:", document_id)
            print("TOPIC:", metadata.get("topic"))
            print("SOURCE:", metadata.get("source"))
            print("CHUNK INDEX:", metadata.get("chunk_index"))
            print("\nDOCUMENT TEXT:\n")
            print(payload.get("page_content", ""))

    print("\nReview complete. No Qdrant data modified.")


if __name__ == "__main__":
    main()
